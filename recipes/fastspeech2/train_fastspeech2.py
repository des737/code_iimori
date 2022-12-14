import sys
from functools import partial
import warnings

import hydra
import torch
from omegaconf import DictConfig
import matplotlib.pyplot as plt

sys.path.append("../..")
from vc_tts_template.fastspeech2.collate_fn import (
    collate_fn_fastspeech2, fastspeech2_get_data_loaders)
from vc_tts_template.train_utils import setup, get_vocoder, vocoder_infer, free_tensors_memory
from recipes.common.train_loop import train_loop
from recipes.fastspeech2.utils import plot_mel_with_prosody
from recipes.common.fit_scaler import MultiSpeakerStandardScaler  # noqa: F401

warnings.simplefilter('ignore', UserWarning)


def fastspeech2_train_step(
    model,
    optimizer,
    lr_scheduler,
    train,
    loss,
    batch,
    logger,
    scaler,
    grad_checker,
):
    """dev時にはpredしたp, eで計算してほしいので, オリジナルのtrain_stepに.
    """
    optimizer.zero_grad()

    # Run forwaard
    with torch.cuda.amp.autocast():
        if train is True:
            output = model(*batch)
        else:
            output = model(
                *batch[:-3],
                p_targets=None,
                e_targets=None,
                d_targets=batch[-1],
            )

        loss, loss_values = loss(batch, output)

    # Update
    if train:
        scaler.scale(loss).backward()
        grad_checker.set_params(model.named_parameters())
        free_tensors_memory([loss])
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            grad_checker.report(loss_values)
            if scaler.is_enabled() is True:
                logger.info("grad norm is NaN. Will Skip updating")
            else:
                logger.error("grad norm is NaN. check your model grad flow.")
                raise ValueError("Please check log.")
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

    return loss_values


@torch.no_grad()
def fastspeech2_eval_model(
    phase, step, model, writer, batch, is_inference, vocoder_infer, sampling_rate
):
    # 最大3つまで
    N = min(len(batch[0]), 3)

    if is_inference:
        # pitch, energyとして正解データを与えない.
        output = model(
            *batch[:-6],
            mels=None,
            mel_lens=None,
            max_mel_len=None,
            p_targets=None,
            e_targets=None,
            d_targets=None,
        )
    else:
        output = model(*batch)

    for idx in range(N):  # 一個ずつsummary writerしていく.
        file_name = batch[0][idx]
        if phase == 'train':
            file_name = f"utt_{idx}"

        speakers = [batch[0][idx].split("_")[0]]
        mel_post = output[1][idx].cpu().data.numpy().T
        pitch = output[2][idx].cpu().data.numpy()
        energy = output[3][idx].cpu().data.numpy()
        duration = output[5][idx].cpu().data.numpy().astype("int16")
        mel_len_pre = output[9][idx].item()
        mel_gt = batch[-6][idx].cpu().data.numpy().T
        mel_len_gt = batch[-5][idx].item()
        pitch_gt = batch[-3][idx].cpu().data.numpy()
        energy_gt = batch[-2][idx].cpu().data.numpy()
        duration_gt = batch[-1][idx].cpu().data.numpy()
        audio_recon = vocoder_infer(batch[-6][idx][:mel_len_gt].unsqueeze(0), speakers)[0]
        audio_synth = vocoder_infer(output[1][idx][:mel_len_pre].unsqueeze(0), speakers)[0]

        mel_gts = [mel_gt, pitch_gt, energy_gt, duration_gt]
        if is_inference:
            group = f"{phase}/inference"
            mel_posts = [mel_post, pitch, energy, duration]
        else:
            group = f"{phase}/teacher_forcing"
            mel_posts = [mel_post, pitch_gt, energy_gt, duration_gt]

        fig = plot_mel_with_prosody([mel_posts, mel_gts], ["out_after_postnet", "out_ground_truth"])
        writer.add_figure(f"{group}/{file_name}", fig, step)
        if is_inference:
            writer.add_audio(f"{group}/{file_name}_reconstruct", audio_recon/max(abs(audio_recon)), step, sampling_rate)
        writer.add_audio(f"{group}/{file_name}_synthesis", audio_synth/max(abs(audio_synth)), step, sampling_rate)
        plt.close()


def to_device(data, phase, device):
    (
        ids,
        speakers,
        emotions,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        pitches,
        energies,
        durations,
    ) = data
    speakers = torch.from_numpy(speakers).long().to(device, non_blocking=True)
    emotions = torch.from_numpy(emotions).long().to(device, non_blocking=True)
    texts = torch.from_numpy(texts).long().to(device, non_blocking=True)
    src_lens = torch.from_numpy(src_lens).to(device, non_blocking=True)
    mels = torch.from_numpy(mels).float().to(device, non_blocking=True)
    mel_lens = torch.from_numpy(mel_lens).to(device, non_blocking=True)
    pitches = torch.from_numpy(pitches).float().to(device, non_blocking=True)
    energies = torch.from_numpy(energies).to(device, non_blocking=True)
    durations = torch.from_numpy(durations).long().to(device, non_blocking=True)

    return (
        ids,
        speakers,
        emotions,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        pitches,
        energies,
        durations,
    )


@hydra.main(config_path="conf/train_fastspeech2", config_name="config")
def my_app(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 以下自由
    collate_fn = partial(
        collate_fn_fastspeech2, batch_size=config.data.batch_size,
        speaker_dict=config.model.netG.speakers, emotion_dict=config.model.netG.emotions,
    )

    model, optimizer, lr_scheduler, loss, data_loaders, writers, logger, last_epoch, last_train_iter = setup(
        config, device, collate_fn, fastspeech2_get_data_loaders  # type: ignore
    )
    # set_vocoder
    vocoder = get_vocoder(
        device, config.train.vocoder_name, config.train.vocoder_config, config.train.vocoder_weight_path
    )
    _vocoder_infer = partial(
        vocoder_infer, vocoder_dict={config.train.vocoder_name: vocoder},
        mel_scaler_path=config.train.mel_scaler_path,
        max_wav_value=config.train.max_wav_value
    )
    eval_model = partial(
        fastspeech2_eval_model, vocoder_infer=_vocoder_infer, sampling_rate=config.train.sampling_rate
    )

    # 以下固定
    to_device_ = partial(to_device, device=device)
    train_loop(config, to_device_, model, optimizer, lr_scheduler, loss,
               data_loaders, writers, logger, eval_model, fastspeech2_train_step,
               last_epoch=last_epoch, last_train_iter=last_train_iter)


if __name__ == "__main__":
    my_app()

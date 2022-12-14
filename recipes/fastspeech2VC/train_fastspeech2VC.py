import sys
from functools import partial
import warnings

import hydra
import torch
from omegaconf import DictConfig
import matplotlib.pyplot as plt
sys.path.append("../..")
from vc_tts_template.fastspeech2VC.collate_fn import (
    collate_fn_fastspeech2VC, fastspeech2VC_get_data_loaders)
from vc_tts_template.train_utils import setup, get_vocoder, vocoder_infer, free_tensors_memory
from recipes.common.train_loop import train_loop
from recipes.fastspeech2VC.utils import plot_mel_with_prosody

warnings.simplefilter('ignore', UserWarning)


def fastspeech2VC_train_step(
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
            if len(batch) == 16:
                # fastspeech2VC
                output = model(
                    *batch[:13],
                    t_pitches=None,
                    t_energies=None,
                    t_durations=batch[15],
                )
            elif len(batch) == 18:
                # fastspeech2VCwGMM
                output = model(
                    *batch[:13],
                    t_pitches=None,
                    t_energies=None,
                    t_durations=batch[15],
                    s_snt_durations=batch[16],
                    t_snt_durations=batch[17],
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
def fastspeech2VC_eval_model(
    phase, step, model, writer, batch, is_inference, vocoder_infer, sampling_rate, reduction_factor
):
    # 最大3つまで
    N = min(len(batch[0]), 3)

    if is_inference:
        # pitch, energyとして正解データを与えない.
        if len(batch) == 16:
            # fastspeech2VC
            output = model(
                *batch[:10],
                t_mels=None,
                t_mel_lens=None,
                max_t_mel_len=None,
                t_pitches=None,
                t_energies=None,
                t_durations=None,
            )
        elif len(batch) == 18:
            # fastspeech2VCwGMM
            output = model(
                *batch[:10],
                t_mels=None,
                t_mel_lens=None,
                max_t_mel_len=None,
                t_pitches=None,
                t_energies=None,
                t_durations=None,
                s_snt_durations=batch[16],
            )
    else:
        output = model(*batch)

    for idx in range(N):  # 一個ずつsummary writerしていく.
        file_name = batch[0][idx]
        if phase == 'train':
            file_name = f"utt_{idx}"

        mel_post = output[1][idx].cpu().data.numpy().T
        pitch = output[2][idx].cpu().data.numpy()
        energy = output[3][idx].cpu().data.numpy()
        duration = output[5][idx].cpu().data.numpy().astype("int16")
        mel_len_pre = output[9][idx].item()

        mel_gt = batch[10][idx].cpu().data.numpy().T
        mel_len_gt = batch[11][idx].item()
        pitch_gt = batch[13][idx].cpu().data.numpy()
        energy_gt = batch[14][idx].cpu().data.numpy()
        duration_gt = batch[15][idx].cpu().data.numpy()
        audio_recon = vocoder_infer(batch[10][idx][:mel_len_gt].unsqueeze(0))[0]
        audio_synth = vocoder_infer(output[1][idx][:mel_len_pre].unsqueeze(0))[0]

        mel_gts = [mel_gt, pitch_gt, energy_gt, duration_gt]
        if is_inference:
            group = f"{phase}/inference"
            mel_posts = [mel_post, pitch, energy, duration]
        else:
            group = f"{phase}/teacher_forcing"
            mel_posts = [mel_post, pitch_gt, energy_gt, duration_gt]

        fig = plot_mel_with_prosody([mel_posts, mel_gts], ["out_after_postnet", "out_ground_truth"],
                                    reduction_factor, False)
        writer.add_figure(f"{group}/{file_name}", fig, step)
        if is_inference:
            writer.add_audio(f"{group}/{file_name}_reconstruct", audio_recon/max(abs(audio_recon)), step, sampling_rate)
        writer.add_audio(f"{group}/{file_name}_synthesis", audio_synth/max(abs(audio_synth)), step, sampling_rate)
        plt.close()


def to_device(data, phase, device):
    (
        ids,
        s_speakers,
        t_speakers,
        s_emotions,
        t_emotions,
        s_mels,
        s_mel_lens,
        max_src_len,
        s_pitches,
        s_energies,
        t_mels,
        t_mel_lens,
        max_mel_len,
        t_pitches,
        t_energies,
        t_durations,
        s_snt_durations,
        t_snt_durations,
    ) = data

    s_speakers = torch.from_numpy(s_speakers).long().to(device)
    t_speakers = torch.from_numpy(t_speakers).long().to(device)
    s_emotions = torch.from_numpy(s_emotions).long().to(device)
    t_emotions = torch.from_numpy(t_emotions).long().to(device)
    s_mels = torch.from_numpy(s_mels).float().to(device)
    s_mel_lens = torch.from_numpy(s_mel_lens).long().to(device)
    s_pitches = torch.from_numpy(s_pitches).float().to(device)
    s_energies = torch.from_numpy(s_energies).float().to(device)
    t_mels = torch.from_numpy(t_mels).float().to(device)
    t_mel_lens = torch.from_numpy(t_mel_lens).long().to(device)
    t_pitches = torch.from_numpy(t_pitches).float().to(device)
    t_energies = torch.from_numpy(t_energies).float().to(device)
    t_durations = torch.from_numpy(t_durations).long().to(device)
    if s_snt_durations is not None:
        s_snt_durations = torch.from_numpy(s_snt_durations).long().to(device)
        t_snt_durations = torch.from_numpy(t_snt_durations).long().to(device)
        return (
            ids,
            s_speakers,
            t_speakers,
            s_emotions,
            t_emotions,
            s_mels,
            s_mel_lens,
            max_src_len,
            s_pitches,
            s_energies,
            t_mels,
            t_mel_lens,
            max_mel_len,
            t_pitches,
            t_energies,
            t_durations,
            s_snt_durations,
            t_snt_durations,
        )
    return (
        ids,
        s_speakers,
        t_speakers,
        s_emotions,
        t_emotions,
        s_mels,
        s_mel_lens,
        max_src_len,
        s_pitches,
        s_energies,
        t_mels,
        t_mel_lens,
        max_mel_len,
        t_pitches,
        t_energies,
        t_durations,
    )


@hydra.main(config_path="conf/train_fastspeech2VC", config_name="config")
def my_app(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 以下自由
    collate_fn = partial(
        collate_fn_fastspeech2VC, batch_size=config.data.batch_size,
        speaker_dict=config.model.netG.speakers, emotion_dict=config.model.netG.emotions,
        sentence_duration=config.data.sentence_duration
    )

    model, optimizer, lr_scheduler, loss, data_loaders, writers, logger, last_epoch, last_train_iter = setup(
        config, device, collate_fn, fastspeech2VC_get_data_loaders  # type: ignore
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
        fastspeech2VC_eval_model, vocoder_infer=_vocoder_infer, sampling_rate=config.train.sampling_rate,
        reduction_factor=config.model.netG.reduction_factor
    )

    # 以下固定
    to_device_ = partial(to_device, device=device)
    train_loop(config, to_device_, model, optimizer, lr_scheduler, loss,
               data_loaders, writers, logger, eval_model, fastspeech2VC_train_step,
               last_epoch=last_epoch, last_train_iter=last_train_iter)


if __name__ == "__main__":
    my_app()

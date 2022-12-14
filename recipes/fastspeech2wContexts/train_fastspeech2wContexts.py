import sys
from functools import partial
import warnings

import hydra
import torch
from omegaconf import DictConfig

sys.path.append("../..")
from vc_tts_template.fastspeech2wContexts.collate_fn import (
    collate_fn_fastspeech2wContexts, fastspeech2wContexts_get_data_loaders
)
from vc_tts_template.fastspeech2wContexts.collate_fn_PEProsody import (
    collate_fn_fastspeech2wPEProsody, fastspeech2wPEProsody_get_data_loaders
)
from vc_tts_template.train_utils import setup, get_vocoder, vocoder_infer
from recipes.common.train_loop import train_loop
from recipes.fastspeech2.train_fastspeech2 import fastspeech2_train_step, fastspeech2_eval_model
from recipes.common.fit_scaler import MultiSpeakerStandardScaler  # noqa: F401

warnings.simplefilter('ignore', UserWarning)


def to_device(data, phase, device):
    if len(data) == 22:
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
            c_txt_embs,
            h_txt_embs,
            h_txt_emb_lens,
            h_speakers,
            h_emotions,
            h_prosody_emb,
            h_prosody_lens,
            h_g_prosody_embs,
            h_prosody_speakers,
            h_prosody_emotions,
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
        c_txt_embs = torch.from_numpy(c_txt_embs).float().to(device, non_blocking=True)
        h_txt_embs = torch.from_numpy(h_txt_embs).float().to(device, non_blocking=True)
        h_txt_emb_lens = torch.from_numpy(h_txt_emb_lens).long().to(device, non_blocking=True)
        h_speakers = torch.from_numpy(h_speakers).long().to(device, non_blocking=True)
        h_emotions = torch.from_numpy(h_emotions).long().to(device, non_blocking=True)

        if h_prosody_emb is not None:
            h_prosody_emb = torch.from_numpy(h_prosody_emb).float().to(device, non_blocking=True)
            h_prosody_lens = torch.from_numpy(h_prosody_lens).long().to(device, non_blocking=True)
            h_prosody_speakers = torch.from_numpy(h_prosody_speakers).long().to(device, non_blocking=True)
            h_prosody_emotions = torch.from_numpy(h_prosody_emotions).long().to(device, non_blocking=True)
        if h_g_prosody_embs is not None:
            h_g_prosody_embs = torch.from_numpy(h_g_prosody_embs).float().to(device, non_blocking=True)

        return (
            ids,
            speakers,
            emotions,
            texts,
            src_lens,
            max_src_len,
            c_txt_embs,
            h_txt_embs,
            h_txt_emb_lens,
            h_speakers,
            h_emotions,
            h_prosody_emb,
            h_prosody_lens,
            h_g_prosody_embs,
            h_prosody_speakers,
            h_prosody_emotions,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )
    elif len(data) == 29:
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
            c_txt_embs,
            c_txt_embs_lens,
            h_txt_embs,
            h_txt_emb_lens,
            h_speakers,
            h_emotions,
            c_prosody_embs,
            c_prosody_embs_lens,
            c_prosody_embs_duration,
            c_prosody_embs_phonemes,
            h_prosody_embs,
            h_prosody_embs_lens,
            h_prosody_embs_len,
            h_local_prosody_emb,
            h_local_prosody_emb_lens,
            h_local_prosody_speakers,
            h_local_prosody_emotions,
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
        c_txt_embs = torch.from_numpy(c_txt_embs).float().to(device, non_blocking=True)
        if c_txt_embs_lens is not None:
            c_txt_embs_lens = torch.from_numpy(c_txt_embs_lens).long().to(device, non_blocking=True)
        h_txt_embs = torch.from_numpy(h_txt_embs).float().to(device, non_blocking=True)
        h_txt_emb_lens = torch.from_numpy(h_txt_emb_lens).long().to(device, non_blocking=True)
        h_speakers = torch.from_numpy(h_speakers).long().to(device, non_blocking=True)
        h_emotions = torch.from_numpy(h_emotions).long().to(device, non_blocking=True)

        c_prosody_embs = torch.from_numpy(c_prosody_embs).float().to(device, non_blocking=True)
        c_prosody_embs_lens = torch.from_numpy(c_prosody_embs_lens).long().to(device, non_blocking=True)
        if c_prosody_embs_duration is not None:
            c_prosody_embs_duration = torch.from_numpy(c_prosody_embs_duration).long().to(device, non_blocking=True)
        if c_prosody_embs_phonemes is not None:
            c_prosody_embs_phonemes = torch.from_numpy(c_prosody_embs_phonemes).long().to(device, non_blocking=True)
        h_prosody_embs = torch.from_numpy(h_prosody_embs).float().to(device, non_blocking=True)
        h_prosody_embs_lens = torch.from_numpy(h_prosody_embs_lens).long().to(device, non_blocking=True)
        h_prosody_embs_len = torch.from_numpy(h_prosody_embs_len).long().to(device, non_blocking=True)

        if h_local_prosody_emb is not None:
            h_local_prosody_emb = torch.from_numpy(h_local_prosody_emb).float().to(device, non_blocking=True)
            h_local_prosody_emb_lens = torch.from_numpy(h_local_prosody_emb_lens).long().to(device, non_blocking=True)
            h_local_prosody_speakers = torch.from_numpy(h_local_prosody_speakers).long().to(device, non_blocking=True)
            h_local_prosody_emotions = torch.from_numpy(h_local_prosody_emotions).long().to(device, non_blocking=True)

        return (
            ids,
            speakers,
            emotions,
            texts,
            src_lens,
            max_src_len,
            c_txt_embs,
            c_txt_embs_lens,
            h_txt_embs,
            h_txt_emb_lens,
            h_speakers,
            h_emotions,
            c_prosody_embs,
            c_prosody_embs_lens,
            c_prosody_embs_duration,
            c_prosody_embs_phonemes,
            h_prosody_embs,
            h_prosody_embs_lens,
            h_prosody_embs_len,
            h_local_prosody_emb,
            h_local_prosody_emb_lens,
            h_local_prosody_speakers,
            h_local_prosody_emotions,
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
    if ("PEProsody" in config.data.acoustic_model) or ("EmotionPredictor" in config.data.acoustic_model):
        _collate_fn = collate_fn_fastspeech2wPEProsody
        _get_data_loader = fastspeech2wPEProsody_get_data_loaders
    else:
        _collate_fn = collate_fn_fastspeech2wContexts
        _get_data_loader = fastspeech2wContexts_get_data_loaders

    collate_fn = partial(
        _collate_fn, batch_size=config.data.batch_size,
        speaker_dict=config.model.netG.speakers, emotion_dict=config.model.netG.emotions,
    )

    model, optimizer, lr_scheduler, loss, data_loaders, writers, logger, last_epoch, last_train_iter = setup(
        config, device, collate_fn, _get_data_loader  # type: ignore
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

import json
import sys
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pyworld as pw
from scipy.interpolate import interp1d

sys.path.append('..')
from vc_tts_template.dsp import logmelspectrogram
from vc_tts_template.pretrained import retrieve_pretrained_model
from vc_tts_template.utils import StandardScaler


class FastSpeech2VC(object):
    """FastSpeech 2 based text-to-speech

    Args:
        model_dir (str): model directory. A pre-trained model (ID: ``fastspeech2``)
            is used if None.
        device (str): cpu or cuda.

    Examples:

        >>> from vc_tts_template.fastspeech2 import FastSpeech2VC
        >>> engine = FastSpeech2VC()
        >>> wav, sr = engine.tts("一貫学習にチャレンジしましょう！")
    """

    def __init__(self, model_dir=None, model_name=None, device=None):
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_dir is None:
            if model_name is not None:
                model_dir = retrieve_pretrained_model(model_name)
            else:
                model_dir = retrieve_pretrained_model("fastspeech2VC")
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        # search for config.yaml
        config = OmegaConf.load(model_dir / "config.yaml")
        self.sample_rate = config.sample_rate
        self.is_continuous_pitch = config.is_continuous_pitch > 0
        self.get_mel = partial(
            logmelspectrogram, sr=self.sample_rate, n_fft=config.filter_length,
            hop_length=config.hop_length, win_length=config.win_length, n_mels=config.n_mel_channels,
            fmin=config.mel_fmin, fmax=config.mel_fmax, clip=config.clip, log_base=config.log_base, need_energy=True
        )
        self.get_pitch = partial(
            pw.dio, fs=self.sample_rate,
            frame_period=config.hop_length / self.sample_rate * 1000
        )

        # 音響モデル
        self.acoustic_config = OmegaConf.load(model_dir / "acoustic_model.yaml")
        self.acoustic_model = instantiate(self.acoustic_config.netG).to(self.device)
        checkpoint = torch.load(
            model_dir / "acoustic_model.pth",
            map_location=self.device,
        )
        self.acoustic_model.load_state_dict(checkpoint["state_dict"])
        self.acoustic_in_mel_scaler = StandardScaler(
            np.load(model_dir / "in_fastspeech2VC_mel_scaler_mean.npy"),
            np.load(model_dir / "in_fastspeech2VC_mel_scaler_var.npy"),
            np.load(model_dir / "in_fastspeech2VC_mel_scaler_scale.npy"),
        )
        self.acoustic_in_pitch_scaler = StandardScaler(
            np.load(model_dir / "in_fastspeech2VC_pitch_scaler_mean.npy"),
            np.load(model_dir / "in_fastspeech2VC_pitch_scaler_var.npy"),
            np.load(model_dir / "in_fastspeech2VC_pitch_scaler_scale.npy"),
        )
        self.acoustic_in_energy_scaler = StandardScaler(
            np.load(model_dir / "in_fastspeech2VC_energy_scaler_mean.npy"),
            np.load(model_dir / "in_fastspeech2VC_energy_scaler_var.npy"),
            np.load(model_dir / "in_fastspeech2VC_energy_scaler_scale.npy"),
        )
        self.acoustic_out_scaler = StandardScaler(
            np.load(model_dir / "out_fastspeech2VC_mel_scaler_mean.npy"),
            np.load(model_dir / "out_fastspeech2VC_mel_scaler_var.npy"),
            np.load(model_dir / "out_fastspeech2VC_mel_scaler_scale.npy"),
        )
        self.acoustic_model.eval()

        # vocoder
        self.vocoder_config = OmegaConf.load(model_dir / "vocoder_model.yaml")
        self.vocoder_model = instantiate(self.vocoder_config.netG).to(self.device)
        checkpoint = torch.load(
            model_dir / "vocoder_model.pth",
            map_location=self.device,
        )
        self.vocoder_model.load_state_dict(checkpoint["state_dict"]["netG"])
        self.vocoder_model.eval()
        self.vocoder_model.remove_weight_norm()

    def __repr__(self):
        acoustic_str = json.dumps(
            OmegaConf.to_container(self.acoustic_config["netG"]),
            sort_keys=False,
            indent=4,
        )
        wavenet_str = json.dumps(
            OmegaConf.to_container(self.vocoder_config["netG"]),
            sort_keys=False,
            indent=4,
        )

        return f"""Fastspeech2 VC (sampling rate: {self.sample_rate})

Acoustic model: {acoustic_str}
Vocoder model: {wavenet_str}
"""

    def set_device(self, device):
        """Set device for the VC models

        Args:
            device (str): cpu or cuda.
        """
        self.device = device
        self.acoustic_model.to(device)
        self.vocoder_model.to(device)

    @torch.no_grad()
    def vc(self,
           wav, wav_sr,
           s_speaker=None, t_speaker=None,
           s_emotion=None, t_emotion=None,
           ):
        """Run VC

        Args:
            text (str): Input text
            speaker (str): you can select speaker if you train with it.
            tqdm (object, optional): tqdm object. Defaults to None.

        Returns:
            tuple: audio array (np.int16) and sampling rate (int)
        """
        # OpenJTalkを用いて言語特徴量の抽出
        if wav.dtype in [np.int16, np.int32]:
            wav = (wav / np.iinfo(wav.dtype).max).astype(np.float64)
        wav = librosa.resample(wav, wav_sr, self.sample_rate)
        s_mel, s_energy = self.get_mel(wav)
        s_pitch, t = self.get_pitch(wav.astype(np.float64))
        s_pitch = pw.stonemask(wav.astype(np.float64),
                               s_pitch, t, self.sample_rate)
        s_energy = np.log(s_energy+1e-6)

        if self.is_continuous_pitch is True:
            nonzero_ids = np.where(s_energy > -5.0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                s_pitch[nonzero_ids],
                fill_value=(s_pitch[nonzero_ids[0]], s_pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            s_pitch = interp_fn(np.arange(0, len(s_pitch)))

        s_pitch = np.log(s_pitch+1e-6)
        s_mel = self.acoustic_in_mel_scaler.transform(s_mel)
        s_pitch = self.acoustic_in_pitch_scaler.transform(s_pitch)
        s_energy = self.acoustic_in_energy_scaler.transform(s_energy)
        if s_speaker is None:
            s_speakers = np.array([0])
            t_speakers = np.array([0])
        else:
            s_speakers = np.array([self.acoustic_model.speakers[s_speaker]])
            t_speakers = np.array([self.acoustic_model.speakers[t_speaker]])
        if s_emotion is None:
            s_emotions = np.array([0])
            t_emotions = np.array([0])
        else:
            s_emotions = np.array([self.acoustic_model.emotions[s_emotion]])
            t_emotions = np.array([self.acoustic_model.emotions[t_emotion]])
        s_mels = np.array([s_mel])
        s_pitches = np.array([s_pitch])
        s_energies = np.array([s_energy])
        s_mel_lens = np.array([s_mel.shape[0]])
        max_s_mel_len = max(s_mel_lens)

        s_speakers = torch.tensor(s_speakers).long().to(self.device)
        t_speakers = torch.tensor(t_speakers).long().to(self.device)
        s_emotions = torch.tensor(s_emotions).long().to(self.device)
        t_emotions = torch.tensor(t_emotions).long().to(self.device)
        s_mels = torch.tensor(s_mels).float().to(self.device)
        s_mel_lens = torch.tensor(s_mel_lens).long().to(self.device)
        s_pitches = torch.tensor(s_pitches).float().to(self.device)
        s_energies = torch.tensor(s_energies).float().to(self.device)

        output = self.acoustic_model(
            None, s_speakers, t_speakers, s_emotions, t_emotions,
            s_mels, s_mel_lens, max_s_mel_len, s_pitches, s_energies
        )

        mel_post = output[1]
        mels = [self.acoustic_out_scaler.inverse_transform(mel.cpu().data.numpy()) for mel in mel_post]  # type: ignore
        mels = torch.Tensor(np.array(mels)).to(self.device)
        wav = self.vocoder_model(mels.transpose(1, 2)).squeeze(1).cpu().data.numpy()[0]

        return self.post_process(wav), self.sample_rate

    def post_process(self, wav):
        wav = np.clip(wav, -1.0, 1.0)
        wav = (wav * 32767.0).astype(np.int16)
        return wav


def randomize_tts_engine_(engine: FastSpeech2VC) -> FastSpeech2VC:
    # アテンションのパラメータの一部を強制的に乱数で初期化することで、学習済みモデルを破壊する
    torch.nn.init.normal_(engine.acoustic_model.decoder.attention.mlp_dec.weight.data)
    return engine

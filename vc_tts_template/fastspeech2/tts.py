import json
from pathlib import Path
import sys

import numpy as np
import pyopenjtalk
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.append('..')
from vc_tts_template.pretrained import retrieve_pretrained_model
from vc_tts_template.frontend.openjtalk import text_to_sequence
from vc_tts_template.utils import StandardScaler


class FastSpeech2TTS(object):
    """FastSpeech 2 based text-to-speech

    Args:
        model_dir (str): model directory. A pre-trained model (ID: ``fastspeech2``)
            is used if None.
        device (str): cpu or cuda.

    Examples:

        >>> from vc_tts_template.fastspeech2 import FastSpeech2TTS
        >>> engine = FastSpeech2TTS()
        >>> wav, sr = engine.tts("一貫学習にチャレンジしましょう！")
    """
    def __init__(self, model_dir=None, device=None):
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_dir is None:
            model_dir = retrieve_pretrained_model("fastspeech2")
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        # search for config.yaml
        if (model_dir / "config.yaml").exists():
            config = OmegaConf.load(model_dir / "config.yaml")
            self.sample_rate = config.sample_rate
        else:
            self.sample_rate = 22050

        # 音響モデル
        self.acoustic_config = OmegaConf.load(model_dir / "acoustic_model.yaml")
        self.acoustic_model = instantiate(self.acoustic_config.netG).to(self.device)
        checkpoint = torch.load(
            model_dir / "acoustic_model.pth",
            map_location=self.device,
        )
        self.acoustic_model.load_state_dict(checkpoint["state_dict"])
        self.acoustic_out_scaler = StandardScaler(
            np.load(model_dir / "out_fastspeech2_mel_scaler_mean.npy"),
            np.load(model_dir / "out_fastspeech2_mel_scaler_var.npy"),
            np.load(model_dir / "out_fastspeech2_mel_scaler_scale.npy"),
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

        return f"""Tacotron2 TTS (sampling rate: {self.sample_rate})

Acoustic model: {acoustic_str}
Vocoder model: {wavenet_str}
"""

    def set_device(self, device):
        """Set device for the TTS models

        Args:
            device (str): cpu or cuda.
        """
        self.device = device
        self.acoustic_model.to(device)
        self.vocoder_model.to(device)

    @torch.no_grad()
    def tts(self, text, accents=None, speaker=None, emotion=None):
        """Run TTS

        Args:
            text (str): Input text
            accents (str): Accents corresponding to text
            speaker (str): you can select speaker if you train with it.
            tqdm (object, optional): tqdm object. Defaults to None.

        Returns:
            tuple: audio array (np.int16) and sampling rate (int)
        """
        # OpenJTalkを用いて言語特徴量の抽出
        phonemes = pyopenjtalk.g2p(text).split(" ")
        if speaker is None:
            speakers = np.array([0])
        else:
            speakers = np.array([self.acoustic_model.speakers[speaker]])
        if emotion is None:
            emotions = np.array([0])
        else:
            emotions = np.array([self.acoustic_model.emotions[emotion]])

        if accents is not None:
            if len(accents) != len(phonemes):
                raise ValueError(
                    f"""Error! アクセントの文字列は、音素に一致するようにしてください。\n
                    今テキストの音素は以下の通りです。\n
                    {phonemes}
                    """
                )
            phonemes = [
                p + acc for p, acc in zip(phonemes, accents)
            ]
        in_feats = np.array(text_to_sequence(phonemes))
        src_lens = [in_feats.shape[0]]
        max_src_len = max(src_lens)

        speakers = torch.tensor(speakers, dtype=torch.long).to(self.device)
        emotions = torch.tensor(emotions, dtype=torch.long).to(self.device)
        in_feats = torch.tensor(in_feats, dtype=torch.long).unsqueeze(0).to(self.device)
        src_lens = torch.tensor(src_lens, dtype=torch.long).to(self.device)

        output = self.acoustic_model(
            ids=None,
            speakers=speakers,
            emotions=emotions,
            texts=in_feats,
            src_lens=src_lens,
            max_src_len=max_src_len,
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


def randomize_tts_engine_(engine: FastSpeech2TTS) -> FastSpeech2TTS:
    # アテンションのパラメータの一部を強制的に乱数で初期化することで、学習済みモデルを破壊する
    torch.nn.init.normal_(engine.acoustic_model.decoder.attention.mlp_dec.weight.data)
    return engine

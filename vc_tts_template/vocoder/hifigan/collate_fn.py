import random
from pathlib import Path
from typing import Callable, Dict
import warnings
import sys

import librosa
import numpy as np
import torch
from hydra.utils import to_absolute_path
from librosa.filters import mel as librosa_mel_fn
from scipy.io import wavfile
from torch.utils import data as data_utils

sys.path.append("../..")
from vc_tts_template.utils import load_utt_list

warnings.simplefilter('ignore', UserWarning)


class hifigan_Dataset(data_utils.Dataset):
    def __init__(self, in_feats_paths, sampling_rate, n_cashe_reuse, split, segment_size=None):
        self.audio_files = in_feats_paths  # wav_path
        self.sampling_rate = sampling_rate
        self._cache_ref_count = 0
        self.split = split
        self.segment_size = segment_size
        # validationでは特にn_cashe_reuse = 0 推奨.
        self.n_cache_reuse = n_cashe_reuse

    def __getitem__(self, index):
        wav_path = self.audio_files[index]
        filename = wav_path.name.replace(".wav", "")
        if self._cache_ref_count == 0:
            _sr, x = wavfile.read(wav_path)
            if x.dtype in [np.int16, np.int32]:
                x = (x / np.iinfo(x.dtype).max).astype(np.float64)
            audio = librosa.resample(x, _sr, self.sampling_rate)

            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start:audio_start+self.segment_size]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        return filename, audio.squeeze(0)

    def __len__(self):
        return len(self.audio_files)


def hifigan_get_data_loaders(data_config: Dict, collate_fn: Callable) -> Dict[str, data_utils.DataLoader]:
    """Get data loaders for training and validation.

    Args:
        data_config: Data configuration.
        collate_fn: Collate function.

    Returns:
        dict: Data loaders.
    """
    data_loaders = {}

    for phase in ["train", "dev"]:
        utt_ids = load_utt_list(to_absolute_path(data_config[phase].utt_list))
        in_dir = Path(to_absolute_path(data_config[phase].in_dir))

        in_feats_paths = [in_dir / f"{utt_id}.wav" for utt_id in utt_ids]

        if phase == 'train':
            dataset = hifigan_Dataset(
                in_feats_paths,
                data_config.sampling_rate,  # type: ignore
                data_config.n_cache_reuse,  # type: ignore
                split=True,
                segment_size=data_config.segment_size  # type: ignore
            )
        else:
            dataset = hifigan_Dataset(
                in_feats_paths,
                data_config.sampling_rate,  # type: ignore
                data_config.n_cache_reuse,  # type: ignore
                split=False
            )

        batch_size = data_config.batch_size * data_config.group_size  # type: ignore
        data_loaders[phase] = data_utils.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=data_config.num_workers,  # type: ignore
            shuffle=phase.startswith("train"),  # trainならTrue
        )

    return data_loaders


mel_basis = {}
hann_window = {}


def spectral_normalize_torch(magnitudes):
    def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax,
                    center=False, max_audio_len=None):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
    if max_audio_len is not None:
        y = y[:, :max_audio_len]
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def collate_fn_hifigan(batch, config):
    ids = []
    audios = []
    mels = []
    mel_losses = []
    for data in batch:
        filename, audio = data
        audio = audio.unsqueeze(0)

        mel = mel_spectrogram(audio, config.n_fft, config.num_mels,
                              config.sampling_rate, config.hop_size, config.win_size, config.fmin, config.fmax,
                              center=False)

        mel_loss = mel_spectrogram(audio, config.n_fft, config.num_mels,
                                   config.sampling_rate, config.hop_size, config.win_size,
                                   config.fmin, config.fmax_loss)

        ids.append(filename)
        audios.append(audio.squeeze(0).numpy())
        mels.append(mel.squeeze().numpy())
        mel_losses.append(mel_loss.squeeze().numpy())

    # group化をdefaultとしているので.
    return [[ids, audios, mels, mel_losses]]

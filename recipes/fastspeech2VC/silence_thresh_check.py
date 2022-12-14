"""preprocessにおいて利用するsilence threshとして
良さそうな閾値を探すためのコード.
"""
from preprocess import delete_novoice
from pathlib import Path
import numpy as np
import librosa
import sys
from scipy.io import wavfile

sys.path.append("../..")
from vc_tts_template.dsp import logmelspectrogram
from vc_tts_template.train_utils import plot_mels


################################################
utt_list = "data/train.list"
src_wav_root = "downloads/jsut_jsss/source"
tgt_wav_root = "downloads/jsut_jsss/target"
silence_thresh_h = -50
silence_thresh_t = -100
shuffle = True
num_plot = 5
sr = 22050
################################################

with open(utt_list) as f:
    utt_ids = [utt_id.strip() for utt_id in f]

src_wav_files = [Path(src_wav_root) / f"{utt_id}.wav" for utt_id in utt_ids]
tgt_wav_files = [Path(tgt_wav_root) / f"{utt_id}.wav" for utt_id in utt_ids]


if shuffle is True:
    indexes = np.random.permutation(len(src_wav_files))[:num_plot]
    source_input_paths = np.array(src_wav_files)[indexes]
    target_input_paths = np.array(tgt_wav_files)[indexes]
else:
    source_input_paths = src_wav_files[:num_plot]
    target_input_paths = tgt_wav_files[:num_plot]


def read_wav(wav_path, sr):
    _sr, x = wavfile.read(wav_path)
    if x.dtype in [np.int16, np.int32]:
        x = (x / np.iinfo(x.dtype).max).astype(np.float64)
    wav = librosa.resample(x, _sr, sr)
    return wav


src_mels = []
tgt_mels = []
titles = []
for src_wav_file, tgt_wav_file in zip(source_input_paths, target_input_paths):
    src_wav, sr_src = delete_novoice(src_wav_file, silence_thresh_h, silence_thresh_t)
    tgt_wav, sr_tgt = delete_novoice(tgt_wav_file, silence_thresh_h, silence_thresh_t)

    src_wav_org = read_wav(src_wav_file, sr)
    src_wav = librosa.resample(src_wav, sr_src, sr)
    tgt_wav_org = read_wav(tgt_wav_file, sr)
    tgt_wav = librosa.resample(tgt_wav, sr_tgt, sr)

    src_mel_org = logmelspectrogram(
        src_wav_org, sr, 1024, 256, 1024,
        80, 0, 8000, 0.00001, "natural"
    )
    src_mel = logmelspectrogram(
        src_wav, sr, 1024, 256, 1024,
        80, 0, 8000, 0.00001, "natural"
    )
    tgt_mel_org = logmelspectrogram(
        tgt_wav_org, sr, 1024, 256, 1024,
        80, 0, 8000, 0.00001, "natural"
    )
    tgt_mel = logmelspectrogram(
        tgt_wav, sr, 1024, 256, 1024,
        80, 0, 8000, 0.00001, "natural"
    )
    titles.append(src_wav_file.name)
    titles.append(src_wav_file.name)
    src_mels.append(src_mel_org.T)
    src_mels.append(src_mel.T)
    tgt_mels.append(tgt_mel_org.T)
    tgt_mels.append(tgt_mel.T)

src_fig = plot_mels(src_mels, titles)
src_fig.savefig("sorce_mels.png")

tgt_fig = plot_mels(tgt_mels, titles)
tgt_fig.savefig("target_mels.png")

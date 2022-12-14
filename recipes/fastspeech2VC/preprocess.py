from typing import Tuple
import argparse
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import librosa
import numpy as np
import pyworld as pw
from scipy.io import wavfile
import pydub
from pydub import AudioSegment, silence

sys.path.append("../..")
from recipes.fastspeech2VC.utils import get_alignment_model, get_alignment
from recipes.fastspeech2VC.duration_preprocess import (
    get_duration, get_sentence_duration
)
from scipy.interpolate import interp1d
from tqdm import tqdm
from vc_tts_template.dsp import logmelspectrogram


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess for Fastspeech2VC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("utt_list", type=str, help="utternace list")
    parser.add_argument("src_wav_root", type=str, help="source wav root")
    parser.add_argument("tgt_wav_root", type=str, help="target wav root")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
    parser.add_argument("--sample_rate", type=int)
    parser.add_argument("--silence_thresh_h", type=int, help="silence thresh of head")
    parser.add_argument("--silence_thresh_t", type=int, help="silence thresh of tail")
    parser.add_argument("--chunk_size", type=int, help="silence chunk size")
    parser.add_argument("--filter_length", type=int)
    parser.add_argument("--hop_length", type=int)
    parser.add_argument("--win_length", type=int)
    parser.add_argument("--n_mel_channels", type=int)
    parser.add_argument("--mel_fmin", type=int)
    parser.add_argument("--mel_fmax", type=int)
    parser.add_argument("--clip", type=float)
    parser.add_argument("--log_base", type=str)
    parser.add_argument("--is_continuous_pitch", type=int)
    parser.add_argument("--reduction_factor", type=int)
    parser.add_argument("--sentence_duration", type=int)
    parser.add_argument("--min_silence_len", type=int)
    parser.add_argument("--model_config_path", type=str, nargs='?')
    parser.add_argument("--pretrained_checkpoint", type=str, nargs='?')
    parser.add_argument("--in_scaler_path", type=str, nargs='?')
    parser.add_argument("--out_scaler_path", type=str, nargs='?')
    parser.add_argument("--batch_size", type=int, nargs='?')
    parser.add_argument("--length_thresh", type=int, nargs='?')
    return parser


def make_novoice_to_zero(audio: AudioSegment, silence_thresh: float, min_silence_len: int) -> AudioSegment:
    """無音判定をくらった部分を, 0にしてしまう.
    """
    silences = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    audio_new = AudioSegment.empty()
    s_index = 0
    for silence_ in silences:
        audio_new += audio[s_index:silence_[0]]
        audio_new += AudioSegment.silent(duration=silence_[1]-silence_[0])
        s_index = silence_[1]

    audio_new += audio[s_index:]

    return audio_new


def pydub_to_np(audio: pydub.AudioSegment) -> Tuple[np.ndarray, int]:
    """Converts pydub audio segment into float32 np array of shape [channels, duration_in_seconds*sample_rate],
    where each value is in range [-1.0, 1.0]. Returns tuple (audio_np_array, sample_rate)"""
    # get_array_of_samples returns the data in format:
    # [sample_1_channel_1, sample_1_channel_2, sample_2_channel_1, sample_2_channel_2, ....]
    # where samples are integers of sample_width bytes.
    return np.array(audio.get_array_of_samples(), dtype=np.float32) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate


def delete_novoice(input_path, silence_thresh_h, silence_thresh_t, chunk_size):
    """無音区間を先頭と末尾から削除します.
    Args:
      input_path: wavファイルへのpath
      output_path: wavファイルを貯めたい場所. ファイル名はinput_pathのbasenameからとる.
      chunk_size: 削除に用いる音声の最小単位. 基本defaultのままで良さそう.
    """
    audio = AudioSegment.from_wav(input_path)
    assert len(audio) > 0, f"{input_path}は音声が入っていないようです"

    # 参考: https://stackoverflow.com/questions/29547218/
    # remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
    def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
        trim_ms = 0  # ms

        assert chunk_size > 0  # to avoid infinite loop
        while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
            trim_ms += chunk_size

        return trim_ms

    start_trim = detect_leading_silence(audio, silence_threshold=silence_thresh_h, chunk_size=chunk_size)
    end_trim = detect_leading_silence(audio.reverse(),
                                      silence_threshold=silence_thresh_t, chunk_size=chunk_size)

    audio_cut = audio[start_trim:len(audio)-end_trim]

    audio_cut = make_novoice_to_zero(audio_cut, silence_thresh_t, chunk_size)

    assert len(audio_cut) > 0, f"{input_path}はすべてcutされてしまいました. 閾値を下げてください"
    return pydub_to_np(audio_cut)


def continuous_pitch(pitch: np.ndarray) -> np.ndarray:
    # 0の値をとったらnan扱いとして, 線形補完を行ってみる.
    nonzero_ids = np.where(pitch > 1e-6)[0]
    interp_fn = interp1d(
        nonzero_ids,
        pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,
    )
    pitch = interp_fn(np.arange(0, len(pitch)))

    return pitch


def process_utterance(wav, sr, n_fft, hop_length, win_length,
                      n_mels, fmin, fmax, clip_thresh, log_base,
                      is_continuous_pitch):
    pitch, t = pw.dio(
        wav.astype(np.float64),
        sr,
        frame_period=hop_length / sr * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64),
                         pitch, t, sr)
    if np.sum(pitch != 0) <= 1:
        return None, None, None
    mel_spectrogram, energy = logmelspectrogram(
        wav,
        sr,
        n_fft,
        hop_length,
        win_length,
        n_mels,
        fmin,
        fmax,
        clip=clip_thresh,
        log_base=log_base,
        need_energy=True
    )
    energy = np.log(energy+1e-6)

    if is_continuous_pitch is True:
        no_voice_indexes = np.where(energy < -5.0)
        pitch[no_voice_indexes] = 0.0
        pitch = continuous_pitch(pitch)

    pitch = np.log(pitch+1e-6)

    if (len(mel_spectrogram) != len(pitch)) or (len(mel_spectrogram) != len(energy)):
        return None, None, None

    return (
        mel_spectrogram,  # (T, mel)
        pitch,
        energy
    )


def preprocess(
    src_wav_file,
    tgt_wav_file,
    sr,
    silence_thresh_h,
    silence_thresh_t,
    chunk_size,
    n_fft,
    hop_length,
    win_length,
    n_mels,
    fmin,
    fmax,
    clip_thresh,
    log_base,
    is_continuous_pitch,
    reduction_factor,
    sentence_duration,
    min_silence_len,
    in_dir,
    out_dir,
    get_duration,
):
    assert src_wav_file.stem == tgt_wav_file.stem

    src_wav, sr_src = delete_novoice(src_wav_file, silence_thresh_h, silence_thresh_t, chunk_size)
    tgt_wav, sr_tgt = delete_novoice(tgt_wav_file, silence_thresh_h, silence_thresh_t, chunk_size)

    src_wav = librosa.resample(src_wav, sr_src, sr)
    tgt_wav = librosa.resample(tgt_wav, sr_tgt, sr)

    utt_id = src_wav_file.stem

    src_mel, src_pitch, src_energy = process_utterance(
        src_wav, sr, n_fft, hop_length, win_length,
        n_mels, fmin, fmax, clip_thresh, log_base,
        is_continuous_pitch
    )
    if src_pitch is None:
        return src_wav_file, None
    tgt_mel, tgt_pitch, tgt_energy = process_utterance(
        tgt_wav, sr, n_fft, hop_length, win_length,
        n_mels, fmin, fmax, clip_thresh, log_base,
        is_continuous_pitch
    )
    if tgt_pitch is None:
        return None, tgt_wav_file

    np.save(
        in_dir / "mel" / f"{utt_id}-feats.npy",
        src_mel.astype(np.float32),
        allow_pickle=False
    )
    np.save(
        in_dir / "pitch" / f"{utt_id}-feats.npy",
        src_pitch.astype(np.float32),
        allow_pickle=False
    )
    np.save(
        in_dir / "energy" / f"{utt_id}-feats.npy",
        src_energy.astype(np.float32),
        allow_pickle=False
    )
    np.save(
        out_dir / "mel" / f"{utt_id}-feats.npy",
        tgt_mel.astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        out_dir / "pitch" / f"{utt_id}-feats.npy",
        tgt_pitch.astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        out_dir / "energy" / f"{utt_id}-feats.npy",
        tgt_energy.astype(np.float32),
        allow_pickle=False,
    )
    if get_duration is not None:
        duration = get_duration(utt_id, src_wav, tgt_wav, sr, n_fft, hop_length, win_length,
                                fmin, fmax, clip_thresh, log_base, reduction_factor)
        np.save(
            out_dir / "duration" / f"{utt_id}-feats.npy",
            duration.astype(np.int16),
            allow_pickle=False,
        )
        if sentence_duration is True:
            src_sent_durations, tgt_sent_durations = get_sentence_duration(
                utt_id, tgt_wav, sr, hop_length, reduction_factor,
                min_silence_len, silence_thresh_t, duration
            )
            np.save(
                in_dir / "sent_duration" / f"{utt_id}-feats.npy",
                src_sent_durations.astype(np.int16),
                allow_pickle=False
            )
            np.save(
                out_dir / "sent_duration" / f"{utt_id}-feats.npy",
                tgt_sent_durations.astype(np.int16),
                allow_pickle=False,
            )
    else:
        # 後で利用するので, tgt_wavを保存しておく.
        (out_dir / "tgt_wavs").mkdir(parents=True, exist_ok=True)
        wavfile.write(out_dir / "tgt_wavs" / f"{utt_id}.wav", sr, tgt_wav)
    return None, None


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    with open(args.utt_list) as f:
        utt_ids = [utt_id.strip() for utt_id in f]

    src_wav_files = [Path(args.src_wav_root) / f"{utt_id}.wav" for utt_id in utt_ids]
    tgt_wav_files = [Path(args.tgt_wav_root) / f"{utt_id}.wav" for utt_id in utt_ids]

    in_dir = Path(args.out_dir) / "in_fastspeech2VC"
    out_dir = Path(args.out_dir) / "out_fastspeech2VC"

    in_mel_dir = Path(args.out_dir) / "in_fastspeech2VC" / "mel"
    in_pitch_dir = Path(args.out_dir) / "in_fastspeech2VC" / "pitch"
    in_energy_dir = Path(args.out_dir) / "in_fastspeech2VC" / "energy"
    in_sent_duration_dir = Path(args.out_dir) / "in_fastspeech2VC" / "sent_duration"
    out_mel_dir = Path(args.out_dir) / "out_fastspeech2VC" / "mel"
    out_pitch_dir = Path(args.out_dir) / "out_fastspeech2VC" / "pitch"
    out_energy_dir = Path(args.out_dir) / "out_fastspeech2VC" / "energy"
    out_duration_dir = Path(args.out_dir) / "out_fastspeech2VC" / "duration"
    out_sent_duration_dir = Path(args.out_dir) / "out_fastspeech2VC" / "sent_duration"

    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_mel_dir.mkdir(parents=True, exist_ok=True)
    in_pitch_dir.mkdir(parents=True, exist_ok=True)
    in_energy_dir.mkdir(parents=True, exist_ok=True)
    in_sent_duration_dir.mkdir(parents=True, exist_ok=True)
    out_mel_dir.mkdir(parents=True, exist_ok=True)
    out_pitch_dir.mkdir(parents=True, exist_ok=True)
    out_energy_dir.mkdir(parents=True, exist_ok=True)
    out_duration_dir.mkdir(parents=True, exist_ok=True)
    out_sent_duration_dir.mkdir(parents=True, exist_ok=True)

    failed_src_lst = []
    failed_tgt_lst = []

    _get_duration = get_duration if args.model_config_path is None else None

    with ProcessPoolExecutor(args.n_jobs) as executor:
        futures = [
            executor.submit(
                preprocess,
                src_wav_file,
                tgt_wav_file,
                args.sample_rate,
                args.silence_thresh_h,
                args.silence_thresh_t,
                args.chunk_size,
                args.filter_length,
                args.hop_length,
                args.win_length,
                args.n_mel_channels,
                args.mel_fmin,
                args.mel_fmax,
                args.clip,
                args.log_base,
                args.is_continuous_pitch > 0,
                args.reduction_factor,
                args.sentence_duration > 0,
                args.min_silence_len,
                in_dir,
                out_dir,
                _get_duration,
            )
            for src_wav_file, tgt_wav_file in zip(src_wav_files, tgt_wav_files)
        ]
        for future in tqdm(futures):
            src_wav_file, tgt_wav_file = future.result()
            if src_wav_file is not None:
                failed_src_lst.append(str(src_wav_file)+'\n')
            if tgt_wav_file is not None:
                failed_tgt_lst.append(str(tgt_wav_file)+'\n')

    with open(in_dir.parent / "failed_src_lst.txt", 'w') as f:
        f.writelines(failed_src_lst)
    with open(in_dir.parent / "failed_tgt_lst.txt", 'w') as f:
        f.writelines(failed_tgt_lst)

    if args.model_config_path is not None:
        # 最後にまとめて計算する.
        print("calc duration with ARmodel!!")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, acoustic_in_scaler, acoustic_out_scaler = get_alignment_model(
            args.model_config_path, args.pretrained_checkpoint, device,
            args.in_scaler_path, args.out_scaler_path
        )

        batch_utt_id = []
        batch_in_mel = []
        batch_out_mel = []
        in_mel_pathes = list(in_mel_dir.glob("*.npy"))
        for i, in_mel_path in tqdm(enumerate(in_mel_pathes)):
            out_mel_path = out_mel_dir / in_mel_path.name
            in_mel = np.load(in_mel_path)
            out_mel = np.load(out_mel_path)
            utt_id = in_mel_path.stem.replace("-feats", "")

            if args.src_wav_root == args.tgt_wav_root:
                duration = np.ones(len(in_mel)//args.reduction_factor)
                np.save(
                    out_dir / "duration" / f"{utt_id}-feats.npy",
                    duration.astype(np.int16),
                    allow_pickle=False,
                )
                continue

            if len(in_mel) < args.length_thresh:
                batch_utt_id.append(utt_id)
                batch_in_mel.append(in_mel)
                batch_out_mel.append(out_mel)
                if ((i+1) % args.batch_size == 0) or (i == len(in_mel_pathes)-1):
                    s_sp_ids = [utt_id.split("_")[0] for utt_id in batch_utt_id]
                    t_sp_ids = [utt_id.split("_")[1] for utt_id in batch_utt_id]
                    s_em_ids = [utt_id.split("_")[-2] for utt_id in batch_utt_id]
                    t_em_ids = [utt_id.split("_")[-1] for utt_id in batch_utt_id]
                    durations = get_alignment(
                        model, device, acoustic_in_scaler, acoustic_out_scaler,
                        batch_in_mel, batch_out_mel,
                        s_sp_ids, t_sp_ids, s_em_ids, t_em_ids
                    )
                    for i, duration in enumerate(durations):
                        utt_id = batch_utt_id[i]
                        np.save(
                            out_dir / "duration" / f"{utt_id}-feats.npy",
                            duration.astype(np.int16),
                            allow_pickle=False,
                        )
            else:
                s_sp_ids = [utt_id.split("_")[0]]
                t_sp_ids = [utt_id.split("_")[1]]
                s_em_ids = [utt_id.split("_")[-2]]
                t_em_ids = [utt_id.split("_")[-1]]
                durations = get_alignment(
                    model, device, acoustic_in_scaler, acoustic_out_scaler,
                    [in_mel], [out_mel],
                    s_sp_ids, t_sp_ids, s_em_ids, t_em_ids
                )
                np.save(
                    out_dir / "duration" / f"{utt_id}-feats.npy",
                    durations[0].astype(np.int16),
                    allow_pickle=False,
                )

        if args.sentence_duration > 0:
            print("calc sentence durations!!")
            utt_ids = []
            tgt_wavs = []
            durations = []
            in_mel_pathes = list(in_mel_dir.glob("*.npy"))
            for in_mel_path in in_mel_pathes:
                utt_id = in_mel_path.stem.replace("-feats", "")
                _, tgt_wav = wavfile.read(out_dir / "tgt_wavs" / f"{utt_id}.wav")
                if tgt_wav.dtype in [np.int16, np.int32]:
                    tgt_wav = (tgt_wav / np.iinfo(tgt_wav.dtype).max).astype(np.float64)
                duration = np.load(out_dir / "duration" / f"{utt_id}-feats.npy")

                utt_ids.append(utt_id)
                tgt_wavs.append(tgt_wav)
                durations.append(duration)

            with ProcessPoolExecutor(args.n_jobs) as executor:
                futures = [
                    executor.submit(
                        get_sentence_duration,
                        utt_id, tgt_wav, args.sample_rate, args.hop_length,
                        args.reduction_factor, args.min_silence_len,
                        args.silence_thresh_t, duration, return_utt_id=True
                    )
                    for utt_id, tgt_wav, duration in zip(utt_ids, tgt_wavs, durations)
                ]
                for future in tqdm(futures):
                    utt_id, src_sent_durations, tgt_sent_durations = future.result()
                    np.save(
                        in_dir / "sent_duration" / f"{utt_id}-feats.npy",
                        src_sent_durations.astype(np.int16),
                        allow_pickle=False
                    )
                    np.save(
                        out_dir / "sent_duration" / f"{utt_id}-feats.npy",
                        tgt_sent_durations.astype(np.int16),
                        allow_pickle=False,
                    )
            # 最後tmpフォルダは消しておく.
            shutil.rmtree(str(out_dir / "tgt_wavs"))

import sys
from typing import List

import numpy as np
from fastdtw import fastdtw
from pydub import AudioSegment, silence

sys.path.append("../..")
from scipy.spatial.distance import cityblock
from vc_tts_template.dsp import logmelspectrogram


def reduction(x: np.ndarray, reduction_factor: int, mode: str = "mean") -> np.ndarray:
    """1D or 2Dに対応.
    2Dの場合: (time, *) を想定.
    """
    n_dim = len(x.shape)

    if n_dim > 2:
        raise ValueError("次元が2以上のarrayは想定されていません.")

    if n_dim == 1:
        x = x[:(x.shape[0]//reduction_factor)*reduction_factor]
        x = x.reshape(x.shape[0]//reduction_factor, reduction_factor)

    else:
        x = x.T
        x = x[:, :(x.shape[1]//reduction_factor)*reduction_factor]
        x = x.reshape(x.shape[0], x.shape[1]//reduction_factor, reduction_factor)

    assert mode in ["mean", "sum", "discrete"]
    if mode == "mean":
        x = x.mean(-1)
    elif mode == "sum":
        x = x.sum(-1)
    elif mode == "discrete":
        if n_dim == 1:
            x = x[:, -1]
        else:
            x = x[:, :, -1]

    return x.T


def get_s_e_index_from_bools(array):
    """True, Falseのbool配列から, Trueがあるindexの最初と最後を見つけてindexの
    配列として返す.
    """
    index_list = []
    flg = 0
    s_ind = 0
    for i, v in enumerate(array):
        if (flg == 0) and v:
            s_ind = i
            flg = 1
        elif (flg == 1) and (not v):
            index_list.append([s_ind, i])
            flg = 0

    if v:
        # 最後がTrueで終わっていた場合.
        index_list.append([s_ind, i+1])

    return index_list


def calc_duration(ts_src: List[np.ndarray], target_path: str, diagonal_index: np.ndarray = None) -> np.ndarray:
    """
    Args:
      ts_src: アライメントさせたい対象.
        その中身は, (time, d)の時系列のリスト.
        最初のがtarget, 次にsorceが入っている.
      diagonal_index: 対角化させたいindex. False, Trueの値が入っている.
        対角化とは, sourceとtargetのtime indexをx, y軸とでもしたときに, 斜めになるようにすること.

    source : target = 多 : 1 のケース
        - 該当するsourceの最初を1, それ以外は0として削除してしまう.
        - 理由; -1, -2 などとして, meanをとることは可能だが, -1, -2のように連続値が出力される必要があり. その制約を課すのは難しいと感じた.
        - 理由: また, 削除は本質的な情報欠落には当たらないと思われる. targetにはない情報なのでつまり不要なので.

    source: target = 1 : 多のケース
        - これは従来通り, sourceに多を割り当てることで対応.
    """
    t_src, s_src = ts_src
    duration = np.ones(s_src.shape[0])

    # alignment開始.
    _, path = fastdtw(t_src, s_src, dist=cityblock)

    # xのpathを取り出して, 長さで正規化する.
    patht = np.array(list(map(lambda l: l[0], path)))
    paths = np.array(list(map(lambda l: l[1], path)))

    b_p_t, b_p_s = 0, 0  # 初期化.
    count = 0
    for p_t, p_s in zip(patht[1:], paths[1:]):
        if b_p_t == p_t:
            # もし, targetの方が連続しているなら, s:t=多:1なので削る.
            duration[p_s] = 0  # 消したいのは, p_tに対応しているp_sであることに注意.
        if b_p_s == p_s:
            # sourceが連続しているなら, s:t=1:多なので増やす方.
            count += 1
        elif count > 0:
            # count > 0で, 一致をしなくなったなら, それは連続が終了したとき.
            duration[b_p_s] += count
            count = 0

        b_p_t = p_t
        b_p_s = p_s

    duration[b_p_s] += count if count > 0 else 0

    if diagonal_index is not None:
        assert s_src.shape[0] == len(
            diagonal_index), f"s_src.shape: {s_src.shape}, len(diagonal_index): {len(diagonal_index)}"
        index_list = get_s_e_index_from_bools(diagonal_index)
        for index_s_t in index_list:
            duration_part = duration[index_s_t[0]:index_s_t[1]]
            if np.sum(duration_part) > len(duration_part):
                # targetの無音区間の方が長いケース.
                mean_ = np.sum(duration_part) // len(duration_part)
                rest_ = int(np.sum(duration_part) % (len(duration_part)*mean_))
                duration_part[:] = mean_
                duration_part[:rest_] += 1
            else:
                # sourceの方が長いケース.
                s_index = int(np.sum(duration_part))
                duration_part[:s_index] = 1
                duration_part[s_index:] = 0

            duration[index_s_t[0]:index_s_t[1]] = duration_part

    assert np.sum(duration) == t_src.shape[0], f"""{target_path}にてdurationの不一致がおきました\n
    duration: {duration}\n
    np.sum(duration): {np.sum(duration)}\n
    t_src.shape: {t_src.shape}"""

    return duration


def get_duration(
    utt_id, src_wav, tgt_wav, sr, n_fft, hop_length, win_length,
    fmin, fmax, clip_thresh, log_base, reduction_factor,
):
    src_mel, energy = logmelspectrogram(
        src_wav, sr, n_fft, hop_length, win_length,
        20, fmin, fmax, clip=clip_thresh, log_base=log_base,
        need_energy=True
    )
    tgt_mel = logmelspectrogram(
        tgt_wav, sr, n_fft, hop_length, win_length,
        20, fmin, fmax, clip=clip_thresh, log_base=log_base,
        need_energy=False
    )
    source_mel = reduction(src_mel, reduction_factor)
    energy = reduction(energy, reduction_factor)
    target_mel = reduction(tgt_mel, reduction_factor)
    duration = calc_duration([target_mel, source_mel], utt_id, np.log(energy+1e-6) < -5.0)
    return duration


def get_sentence_duration(
    utt_id, tgt_wav, sr, hop_length, reduction_factor,
    min_silence_len, silence_thresh, duration, return_utt_id=False
):
    src_reducted_mel_len = len(duration)
    tgt_reducted_mel_len = np.sum(duration)

    if tgt_wav.dtype in [np.float32, np.float64]:
        tgt_wav = (tgt_wav * np.iinfo(np.int16).max).astype(np.int16)

    t_audio = AudioSegment(
        tgt_wav.tobytes(),
        sample_width=2,
        frame_rate=sr,
        channels=1
    )

    t_silences = np.array(
        silence.detect_silence(t_audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    )

    src_sent_durations = []
    tgt_sent_durations = []

    t_silences = (t_silences / 1000 * sr // hop_length // reduction_factor).astype(np.int16)

    for i, (s_frame, _) in enumerate(t_silences):
        if s_frame == 0:
            continue
        if len(tgt_sent_durations) == 0:
            tgt_sent_durations.append(s_frame)
        else:
            tgt_sent_durations.append(s_frame - np.sum(tgt_sent_durations))

        if i == (len(t_silences) - 1):
            tgt_sent_durations.append(tgt_reducted_mel_len - s_frame)

    if len(tgt_sent_durations) == 0:
        tgt_sent_durations = [tgt_reducted_mel_len]
    assert (np.array(tgt_sent_durations) > 0).all(), f"""
        utt_id: {utt_id}
        t_audio_len: {len(t_audio)}
        tgt_sent_durations: {tgt_sent_durations}
        t_silences: {t_silences}
        tgt_mel_len: {tgt_reducted_mel_len}
        duration_len: {len(duration)}
        duration_sum: {np.sum(duration)}
    """
    assert np.sum(tgt_sent_durations) == tgt_reducted_mel_len

    snt_sum = 0
    s_duration_cum = duration.cumsum()
    for i, snt_d in enumerate(tgt_sent_durations):
        snt_sum += snt_d
        snt_idx = np.argmax(s_duration_cum > snt_sum)
        if i == (len(tgt_sent_durations) - 1):
            snt_idx = src_reducted_mel_len
        if len(src_sent_durations) == 0:
            src_sent_durations.append(snt_idx)
        else:
            src_sent_durations.append(snt_idx - np.sum(src_sent_durations))
    assert (np.array(src_sent_durations) > 0).all(), f"""
        src_sent_durations: {src_sent_durations}\n
        tgt_sent_durations: {tgt_sent_durations}\n
        s_duration_cum: {s_duration_cum}\n
        utt_id: {utt_id}\n
    """
    assert np.sum(src_sent_durations) == src_reducted_mel_len

    if return_utt_id is True:
        return utt_id, np.array(src_sent_durations), np.array(tgt_sent_durations)
    return np.array(src_sent_durations), np.array(tgt_sent_durations)


def get_sentence_duration_for_synthesis(
    tgt_wav, tgt_mel_len, sr, hop_length, reduction_factor,
    min_silence_len, silence_thresh
):
    tgt_reducted_mel_len = tgt_mel_len // reduction_factor

    if tgt_wav.dtype in [np.float32, np.float64]:
        tgt_wav = (tgt_wav * np.iinfo(np.int16).max).astype(np.int16)

    t_audio = AudioSegment(
        tgt_wav.tobytes(),
        sample_width=2,
        frame_rate=sr,
        channels=1
    )

    t_silences = np.array(
        silence.detect_silence(t_audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    )

    tgt_sent_durations = []

    t_silences = (t_silences / 1000 * sr // hop_length // reduction_factor).astype(np.int16)

    for i, (s_frame, _) in enumerate(t_silences):
        if s_frame == 0:
            continue
        if len(tgt_sent_durations) == 0:
            tgt_sent_durations.append(s_frame)
        else:
            tgt_sent_durations.append(s_frame - np.sum(tgt_sent_durations))

        if i == (len(t_silences) - 1):
            tgt_sent_durations.append(tgt_reducted_mel_len - s_frame)

    if len(tgt_sent_durations) == 0:
        tgt_sent_durations = [tgt_reducted_mel_len]
    assert (np.array(tgt_sent_durations) > 0).all(), f"""
        tgt_sent_durations: {tgt_sent_durations}
        t_silences: {t_silences}
        tgt_mel_len: {tgt_reducted_mel_len}
    """
    assert np.sum(tgt_sent_durations) == tgt_reducted_mel_len

    return np.array(tgt_sent_durations)

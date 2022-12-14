from typing import List, Dict, Callable, Optional
from pathlib import Path
from hydra.utils import to_absolute_path

from torch.utils import data as data_utils
import numpy as np

from vc_tts_template.utils import load_utt_list, pad_1d, pad_2d, pad_3d
from vc_tts_template.fastspeech2wContexts.collate_fn import make_dialogue_dict, get_embs


def get_peprosody_embs(
    utt_id: str, emb_paths: List[Path], utt2id: Dict, id2utt: Dict, use_hist_num: int,
    start_index: int = 0, use_local_prosody_hist_idx: int = 0,
    seg_emb_paths: Optional[List] = None,
    seg_d_emb_paths: Optional[List] = None, seg_p_emb_paths: Optional[List] = None,
    emb_dim: int = 80,
):
    # TODO: emb_dimは今ハードコーディング中．ちゃんとconfigから受け取れるようにするべき
    current_d_id, current_in_d_id = utt2id[utt_id]

    def get_path_from_uttid(utt_id, emb_paths):
        answer = None
        for path_ in emb_paths:
            answer = path_
            if utt_id in path_.name:
                break
        return answer

    if seg_emb_paths is None:
        current_emb = np.load(
            get_path_from_uttid(utt_id, emb_paths)
        ) if get_path_from_uttid(utt_id, emb_paths).exists() else None
    else:
        current_emb = np.load(
            get_path_from_uttid(utt_id, seg_emb_paths)
        ) if get_path_from_uttid(utt_id, seg_emb_paths).exists() else None
    current_emb_duration = np.load(
        get_path_from_uttid(utt_id, seg_d_emb_paths)
    ) if seg_d_emb_paths is not None else None
    current_emb_phonemes = np.load(
        get_path_from_uttid(utt_id, seg_p_emb_paths)
    ) if seg_p_emb_paths is not None else None

    range_ = range(int(current_in_d_id)-1, max(start_index-1, int(current_in_d_id)-1-use_hist_num), -1)
    hist_embs = []
    hist_emb_len = 0
    history_speakers = []
    history_emotions = []
    for hist_in_d_id in range_:
        utt_id = id2utt[(current_d_id, str(hist_in_d_id))]
        hist_embs.append(np.load(get_path_from_uttid(utt_id, emb_paths)))
        history_speakers.append(utt_id.split('_')[0])
        history_emotions.append(utt_id.split('_')[-1])
        hist_emb_len += 1

    for _ in range(use_hist_num - len(hist_embs)):
        # current len > hist len の時, paddingすると大分無駄があるので切る.
        if current_emb is not None:
            hist_embs.append(np.zeros_like(current_emb)[:1, :])
        else:
            hist_embs.append(np.zeros((1, emb_dim)))
        history_speakers.append("PAD")
        history_emotions.append("PAD")

    if use_local_prosody_hist_idx > -1:
        hist_for_local_emb = hist_embs[use_local_prosody_hist_idx]
        hist_for_local_speaker = history_speakers[use_local_prosody_hist_idx]  # type:ignore
        hist_for_local_emotion = history_emotions[use_local_prosody_hist_idx]  # type:ignore
    else:
        hist_for_local_emb = hist_for_local_speaker = hist_for_local_emotion = None  # type:ignore

    hist_embs_lens = []
    for i, emb in enumerate(hist_embs):
        if i < hist_emb_len:
            hist_embs_lens.append(emb.shape[0])
        else:
            hist_embs_lens.append(0)
    hist_embs = pad_2d(hist_embs)  # type:ignore

    return (
        current_emb, current_emb_duration, current_emb_phonemes,
        hist_embs, np.array(hist_embs_lens), hist_emb_len,
        np.array(history_speakers), np.array(history_emotions),
        hist_for_local_emb, hist_for_local_speaker, hist_for_local_emotion
    )


class fastspeech2wPEProsody_Dataset(data_utils.Dataset):  # type: ignore
    """Dataset for numpy files

    Args:
        in_paths: List of paths to input files
        out_paths: List of paths to output files
    """

    def __init__(
        self,
        in_paths: List,
        out_mel_paths: List,
        out_pitch_paths: List,
        out_energy_paths: List,
        out_duration_paths: List,
        dialogue_info: Path,
        text_emb_paths: List,
        prosody_emb_paths: List,
        text_seg_emb_paths: Optional[List],
        prosody_seg_emb_paths: Optional[List],
        prosody_seg_d_emb_paths: Optional[List],
        prosody_seg_p_emb_paths: Optional[List],
        use_hist_num: int,
        use_prosody_hist_num: int,
        use_local_prosody_hist_idx: int,
        use_situation_text: int,
    ):
        self.in_paths = in_paths
        self.out_mel_paths = out_mel_paths
        self.out_pitch_paths = out_pitch_paths
        self.out_energy_paths = out_energy_paths
        self.out_duration_paths = out_duration_paths
        self.utt2id, self.id2utt = make_dialogue_dict(dialogue_info)
        self.text_emb_paths = text_emb_paths
        self.prosody_emb_paths = prosody_emb_paths
        self.text_seg_emb_paths = text_seg_emb_paths
        self.prosody_seg_emb_paths = prosody_seg_emb_paths
        self.prosody_seg_d_emb_paths = prosody_seg_d_emb_paths
        self.prosody_seg_p_emb_paths = prosody_seg_p_emb_paths
        self.use_hist_num = use_hist_num
        self.use_prosody_hist_num = use_prosody_hist_num if type(use_prosody_hist_num) is int else use_hist_num
        self.use_local_prosody_hist_idx = use_local_prosody_hist_idx
        self.use_situation_text = use_situation_text > 0

    def __getitem__(self, idx: int):
        """Get a pair of input and target

        Args:
            idx: index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        current_txt_emb, history_txt_embs, hist_emb_len, history_speakers, history_emotions = get_embs(
            self.in_paths[idx].name.replace("-feats.npy", ""), self.text_emb_paths,
            self.utt2id, self.id2utt, self.use_hist_num, seg_emb_paths=self.text_seg_emb_paths,
            start_index=1 if self.use_situation_text is False else 0,
        )
        (
            current_prosody_emb, current_prosody_emb_duration, current_prosody_emb_phonemes,
            hist_prosody_embs, hist_prosody_embs_lens, hist_prosody_embs_len, _, _,
            hist_local_prosody_emb, hist_local_prosody_speaker, hist_local_prosody_emotion
        ) = get_peprosody_embs(
            self.in_paths[idx].name.replace("-feats.npy", ""), self.prosody_emb_paths,
            self.utt2id, self.id2utt, self.use_prosody_hist_num, start_index=1,
            use_local_prosody_hist_idx=self.use_local_prosody_hist_idx,
            seg_emb_paths=self.prosody_seg_emb_paths,
            seg_d_emb_paths=self.prosody_seg_d_emb_paths,
            seg_p_emb_paths=self.prosody_seg_p_emb_paths,
        )

        return (
            self.in_paths[idx].name,
            np.load(self.in_paths[idx]),
            np.load(self.out_mel_paths[idx]),
            np.load(self.out_pitch_paths[idx]),
            np.load(self.out_energy_paths[idx]),
            np.load(self.out_duration_paths[idx]),
            current_txt_emb,
            history_txt_embs,
            hist_emb_len,  # textのhist embの数. int.
            history_speakers,
            history_emotions,
            current_prosody_emb,
            current_prosody_emb_duration,
            current_prosody_emb_phonemes,
            hist_prosody_embs,
            hist_prosody_embs_lens,  # hist prosodyのtimeの長さ. List[int]
            hist_prosody_embs_len,  # hist prosodyのhistの長さ. int
            hist_local_prosody_emb,
            hist_local_prosody_speaker,
            hist_local_prosody_emotion
        )

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.in_paths)


def fastspeech2wPEProsody_get_data_loaders(data_config: Dict, collate_fn: Callable) -> Dict[str, data_utils.DataLoader]:
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
        out_dir = Path(to_absolute_path(data_config[phase].out_dir))

        emb_dir = Path(to_absolute_path(data_config.emb_dir))  # type:ignore
        prosody_emb_dir = Path(to_absolute_path(data_config.prosody_emb_dir))  # type:ignore
        dialogue_info = Path(to_absolute_path(data_config.dialogue_info))  # type:ignore
        in_feats_paths = [in_dir / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_mel_paths = [out_dir / "mel" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_pitch_paths = [out_dir / "pitch" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_energy_paths = [out_dir / "energy" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_duration_paths = [out_dir / "duration" / f"{utt_id}-feats.npy" for utt_id in utt_ids]

        text_emb_paths = list(emb_dir.glob("*.npy"))
        prosody_emb_paths = list(prosody_emb_dir.glob("*.npy"))

        emb_seg_dir = emb_dir / "segmented_text_emb"
        prosody_emb_seg_dir = prosody_emb_dir / "segmented_prosody_emb"
        prosody_emb_seg_d_dir = prosody_emb_dir / "segment_duration"
        prosody_emb_seg_p_dir = prosody_emb_dir / "segment_phonemes"
        text_seg_emb_paths = list(emb_seg_dir.glob("*.npy")) if emb_seg_dir.exists() else None
        prosody_seg_emb_paths = list(prosody_emb_seg_dir.glob("*.npy")) if prosody_emb_seg_dir.exists() else None
        prosody_seg_d_emb_paths = list(prosody_emb_seg_d_dir.glob("*.npy")) if prosody_emb_seg_d_dir.exists() else None
        prosody_seg_p_emb_paths = list(prosody_emb_seg_p_dir.glob("*.npy")) if prosody_emb_seg_p_dir.exists() else None

        dataset = fastspeech2wPEProsody_Dataset(
            in_feats_paths,
            out_mel_paths,
            out_pitch_paths,
            out_energy_paths,
            out_duration_paths,
            dialogue_info,
            text_emb_paths,
            prosody_emb_paths,
            text_seg_emb_paths,
            prosody_seg_emb_paths,
            prosody_seg_d_emb_paths,
            prosody_seg_p_emb_paths,
            use_hist_num=data_config.use_hist_num,  # type:ignore
            use_prosody_hist_num=data_config.use_prosody_hist_num,  # type:ignore
            use_local_prosody_hist_idx=data_config.use_local_prosody_hist_idx,  # type:ignore
            use_situation_text=data_config.use_situation_text,  # type:ignore
        )
        data_loaders[phase] = data_utils.DataLoader(
            dataset,
            batch_size=data_config.batch_size * data_config.group_size,  # type: ignore
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=data_config.num_workers,  # type: ignore
            shuffle=phase.startswith("train"),  # trainならTrue
        )

    return data_loaders


def reprocess(batch, idxs, speaker_dict, emotion_dict):
    file_names = [batch[idx][0] for idx in idxs]
    texts = [batch[idx][1] for idx in idxs]
    mels = [batch[idx][2] for idx in idxs]
    pitches = [batch[idx][3] for idx in idxs]
    energies = [batch[idx][4] for idx in idxs]
    durations = [batch[idx][5] for idx in idxs]
    c_txt_embs = [batch[idx][6] for idx in idxs]
    h_txt_embs = [batch[idx][7] for idx in idxs]
    h_txt_emb_lens = [batch[idx][8] for idx in idxs]
    h_speakers = [batch[idx][9] for idx in idxs]
    h_emotions = [batch[idx][10] for idx in idxs]
    c_prosody_embs = [batch[idx][11] for idx in idxs]
    c_prosody_embs_duration = [batch[idx][12] for idx in idxs]
    c_prosody_embs_phonemes = [batch[idx][13] for idx in idxs]
    h_prosody_embs = [batch[idx][14] for idx in idxs]  # [(10, time, 2), (10, time, 2), ...]
    h_prosody_embs_lens = [batch[idx][15] for idx in idxs]  # [[t1, t2, ...], [t1, t2, ...], ...]
    h_prosody_embs_len = [batch[idx][16] for idx in idxs]  # [hist1, hist2, ...]
    h_local_prosody_emb = [batch[idx][17] for idx in idxs]  # [(time, 2), (time, 2), ...]
    h_local_prosody_speaker = [batch[idx][18] for idx in idxs]  # [spk_id, spk_id, ...]
    h_local_prosody_emotion = [batch[idx][19] for idx in idxs]  # [emo_id, emo_id, ...]

    ids = np.array([fname.replace("-feats.npy", "") for fname in file_names])
    if speaker_dict is not None:
        speakers = np.array([speaker_dict[fname.split("_")[0]] for fname in ids])
        h_speakers = np.array([[speaker_dict[spk] for spk in speakers] for speakers in h_speakers])
        if h_local_prosody_speaker[0] is not None:
            h_local_prosody_speakers = np.array([speaker_dict[spk] for spk in h_local_prosody_speaker])
        else:
            h_local_prosody_speakers = None
    else:
        raise ValueError("You Need speaker_dict")
    if emotion_dict is not None:
        emotions = np.array([emotion_dict[fname.split("_")[-1]] for fname in ids])
        h_emotions = np.array([[emotion_dict[emo] for emo in emotions] for emotions in h_emotions])
        if h_local_prosody_emotion[0] is not None:
            h_local_prosody_emotions = np.array([emotion_dict[emo] for emo in h_local_prosody_emotion])
        else:
            h_local_prosody_emotions = None
    else:
        emotions = np.array([0 for _ in idxs])
        h_emotions = np.array([[0 for _ in range(len(h_speakers[0]))] for _ in idxs])
        if h_local_prosody_emotion[0] is not None:
            h_local_prosody_emotions = np.array([0 for _ in idxs])
        else:
            h_local_prosody_emotions = None

    # reprocessの内容をここに.

    text_lens = np.array([text.shape[0] for text in texts])
    mel_lens = np.array([mel.shape[0] for mel in mels])

    texts = pad_1d(texts)
    mels = pad_2d(mels)
    pitches = pad_1d(pitches)
    energies = pad_1d(energies)
    durations = pad_1d(durations)

    # current text embについて
    c_txt_embs_lens = np.array(
        [c_txt_emb.shape[0] for c_txt_emb in c_txt_embs]
    ) if len(c_txt_embs[0].shape) == 2 else None
    c_txt_embs = pad_2d(c_txt_embs) if len(c_txt_embs[0].shape) == 2 else np.array(c_txt_embs)

    # current prosodyについて
    c_prosody_embs_lens = [c_emb.shape[0] for c_emb in c_prosody_embs]
    c_prosody_embs = pad_2d(c_prosody_embs)
    c_prosody_embs_duration = pad_1d(c_prosody_embs_duration) if c_prosody_embs_duration[0] is not None else None
    c_prosody_embs_phonemes = pad_1d(c_prosody_embs_phonemes) if c_prosody_embs_phonemes[0] is not None else None

    # history prosodyについて
    h_prosody_embs = pad_3d(h_prosody_embs, pad_axis=1)

    # local prosodyについて
    h_local_prosody_emb_lens = np.array(
        [p_emb.shape[0] for p_emb in h_local_prosody_emb]
    ) if h_local_prosody_emb[0] is not None else None
    h_local_prosody_emb = pad_2d(h_local_prosody_emb) if h_local_prosody_emb[0] is not None else None

    return (
        ids,
        speakers,
        emotions,
        texts,
        text_lens,
        max(text_lens),
        mels,
        mel_lens,
        max(mel_lens),
        pitches,
        energies,
        durations,
        np.array(c_txt_embs),
        c_txt_embs_lens,
        np.array(h_txt_embs),
        np.array(h_txt_emb_lens),
        h_speakers,  # prosodyと共通
        h_emotions,  # prosodyと共通
        c_prosody_embs,
        np.array(c_prosody_embs_lens),
        c_prosody_embs_duration,
        c_prosody_embs_phonemes,
        h_prosody_embs,
        np.array(h_prosody_embs_lens),
        np.array(h_prosody_embs_len),
        h_local_prosody_emb,
        h_local_prosody_emb_lens,
        h_local_prosody_speakers,
        h_local_prosody_emotions,
    )


def collate_fn_fastspeech2wPEProsody(batch, batch_size, speaker_dict=None, emotion_dict=None):
    """Collate function for Tacotron.
    Args:
        batch (list): List of tuples of the form (inputs, targets).
        Datasetのreturnが1単位となって, それがbatch_size分入って渡される.
    Returns:
        tuple: Batch of inputs, input lengths, targets, target lengths and stop flags.
    """
    # shape[0]がtimeになるようなindexを指定する.
    len_arr = np.array([batch[idx][1].shape[0] for idx in range(len(batch))])
    # 以下固定
    idx_arr = np.argsort(-len_arr)
    tail = idx_arr[len(idx_arr) - (len(idx_arr) % batch_size):]
    idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % batch_size)]
    idx_arr = idx_arr.reshape((-1, batch_size)).tolist()
    if len(tail) > 0:
        idx_arr += [tail.tolist()]
    output = list()

    # 以下, reprocessへの引数が変更の余地あり.
    for idx in idx_arr:
        output.append(reprocess(batch, idx, speaker_dict, emotion_dict))

    return output

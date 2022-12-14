from typing import List, Dict, Callable, Optional
from pathlib import Path
from hydra.utils import to_absolute_path

from torch.utils import data as data_utils
import numpy as np

from vc_tts_template.utils import load_utt_list, pad_1d, pad_2d


def make_dialogue_dict(dialogue_info):
    # utt_idを投げたら, 対話IDと対話内IDを返してくれるような辞書と,
    # その逆で, 対話IDと対話内IDを投げたらそのutt_idを返してくれるような辞書を用意.
    utt2id = {}
    id2utt = {}

    with open(dialogue_info, 'r') as f:
        dialogue_data = f.readlines()

    for dialogue in dialogue_data:
        utt_id, dialogue_id, in_dialogue_id, _ = dialogue.strip().split(":")
        utt2id[utt_id] = (dialogue_id, in_dialogue_id)
        id2utt[(dialogue_id, in_dialogue_id)] = utt_id

    return utt2id, id2utt


def get_embs(
    utt_id: str, emb_paths: List[Path], utt2id: Dict, id2utt: Dict, use_hist_num: int,
    start_index: int = 0, only_latest: bool = False, use_local_prosody_hist_idx: int = 0,
    seg_emb_paths: Optional[List] = None,
):
    current_d_id, current_in_d_id = utt2id[utt_id]

    def get_path_from_uttid(utt_id, emb_paths):
        answer = None
        for path_ in emb_paths:
            answer = path_
            if utt_id in path_.name:
                break
        return answer

    if seg_emb_paths is not None:
        current_emb = np.load(get_path_from_uttid(utt_id, seg_emb_paths))
    else:
        current_emb = np.load(get_path_from_uttid(utt_id, emb_paths))

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
        hist_embs.append(np.zeros(current_emb.shape[-1]))
        history_speakers.append("PAD")
        history_emotions.append("PAD")

    if only_latest is True:
        hist_embs = hist_embs[use_local_prosody_hist_idx]
        history_speakers = [history_speakers[use_local_prosody_hist_idx]]
        history_emotions = [history_emotions[use_local_prosody_hist_idx]]
    else:
        hist_embs = np.stack(hist_embs)  # type: ignore
    return (
        np.array(current_emb), hist_embs, hist_emb_len,
        np.array(history_speakers), np.array(history_emotions)
    )


class fastspeech2wContexts_Dataset(data_utils.Dataset):  # type: ignore
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
        g_prosody_emb_paths: List,
        use_hist_num: int,
        use_local_prosody_hist_idx: int
    ):
        self.in_paths = in_paths
        self.out_mel_paths = out_mel_paths
        self.out_pitch_paths = out_pitch_paths
        self.out_energy_paths = out_energy_paths
        self.out_duration_paths = out_duration_paths
        self.utt2id, self.id2utt = make_dialogue_dict(dialogue_info)
        self.text_emb_paths = text_emb_paths
        self.prosody_emb_paths = prosody_emb_paths
        self.g_prosody_emb_paths = g_prosody_emb_paths
        self.use_hist_num = use_hist_num
        self.use_local_prosody_hist_idx = use_local_prosody_hist_idx

    def __getitem__(self, idx: int):
        """Get a pair of input and target

        Args:
            idx: index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        current_txt_emb, history_txt_embs, hist_emb_len, history_speakers, history_emotions = get_embs(
            self.in_paths[idx].name.replace("-feats.npy", ""), self.text_emb_paths,
            self.utt2id, self.id2utt, self.use_hist_num
        )
        if len(self.prosody_emb_paths) > 0:
            _, history_prosody_emb, _, history_prosody_speakers, history_prosody_emotions = get_embs(
                self.in_paths[idx].name.replace("-feats.npy", ""), self.prosody_emb_paths,
                self.utt2id, self.id2utt, self.use_hist_num, start_index=1, only_latest=True,
                use_local_prosody_hist_idx=self.use_local_prosody_hist_idx
            )
            history_prosody_speakers = history_prosody_speakers[0]
            history_prosody_emotions = history_prosody_emotions[0]
        else:
            history_prosody_speakers = None
            history_prosody_emotions = None
            history_prosody_emb = None
        if len(self.g_prosody_emb_paths) > 0:
            _, history_g_prosody_embs, _, _, _ = get_embs(
                self.in_paths[idx].name.replace("-feats.npy", ""), self.g_prosody_emb_paths,
                self.utt2id, self.id2utt, self.use_hist_num, start_index=1
            )
        else:
            history_g_prosody_embs = None

        return (
            self.in_paths[idx].name,
            np.load(self.in_paths[idx]),
            np.load(self.out_mel_paths[idx]),
            np.load(self.out_pitch_paths[idx]),
            np.load(self.out_energy_paths[idx]),
            np.load(self.out_duration_paths[idx]),
            current_txt_emb,
            history_txt_embs,
            hist_emb_len,
            history_speakers,
            history_emotions,
            history_prosody_emb,  # local prosody
            history_g_prosody_embs,  # global prosody
            history_prosody_speakers,  # speaker of local prosody
            history_prosody_emotions,  # emotion local prosody
        )

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.in_paths)


def fastspeech2wContexts_get_data_loaders(data_config: Dict, collate_fn: Callable) -> Dict[str, data_utils.DataLoader]:
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
        g_prosody_emb_dir = Path(to_absolute_path(data_config.g_prosody_emb_dir))  # type:ignore
        dialogue_info = Path(to_absolute_path(data_config.dialogue_info))  # type:ignore
        in_feats_paths = [in_dir / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_mel_paths = [out_dir / "mel" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_pitch_paths = [out_dir / "pitch" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_energy_paths = [out_dir / "energy" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_duration_paths = [out_dir / "duration" / f"{utt_id}-feats.npy" for utt_id in utt_ids]

        text_emb_paths = list(emb_dir.glob("*.npy"))
        prosody_emb_paths = list(prosody_emb_dir.glob("*.npy"))
        g_prosody_emb_paths = list(g_prosody_emb_dir.glob("*.npy"))

        dataset = fastspeech2wContexts_Dataset(
            in_feats_paths,
            out_mel_paths,
            out_pitch_paths,
            out_energy_paths,
            out_duration_paths,
            dialogue_info,
            text_emb_paths,
            prosody_emb_paths,
            g_prosody_emb_paths,
            use_hist_num=data_config.use_hist_num,  # type:ignore
            use_local_prosody_hist_idx=data_config.use_local_prosody_hist_idx,  # type:ignore
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
    h_prosody_emb = [batch[idx][11] for idx in idxs]
    h_g_prosody_embs = [batch[idx][12] for idx in idxs]
    h_prosody_speakers = [batch[idx][13] for idx in idxs]
    h_prosody_emotions = [batch[idx][14] for idx in idxs]

    ids = np.array([fname.replace("-feats.npy", "") for fname in file_names])
    if speaker_dict is not None:
        speakers = np.array([speaker_dict[fname.split("_")[0]] for fname in ids])
        h_speakers = np.array([[speaker_dict[spk] for spk in speakers] for speakers in h_speakers])
        if h_prosody_speakers[0] is not None:
            h_prosody_speakers = np.array([speaker_dict[spk] for spk in h_prosody_speakers])
        else:
            h_prosody_speakers = None
    else:
        raise ValueError("You Need speaker_dict")
    if emotion_dict is not None:
        emotions = np.array([emotion_dict[fname.split("_")[-1]] for fname in ids])
        h_emotions = np.array([[emotion_dict[emo] for emo in emotions] for emotions in h_emotions])
        if h_prosody_emotions[0] is not None:
            h_prosody_emotions = np.array([emotion_dict[emo] for emo in h_prosody_emotions])
        else:
            h_prosody_emotions = None
    else:
        emotions = np.array([0 for _ in idxs])
        h_emotions = np.array([[0 for _ in range(len(h_speakers[0]))] for _ in idxs])
        if h_prosody_emotions[0] is not None:
            h_prosody_emotions = np.array([0 for _ in idxs])
        else:
            h_prosody_emotions = None

    # reprocessの内容をここに.

    text_lens = np.array([text.shape[0] for text in texts])
    mel_lens = np.array([mel.shape[0] for mel in mels])

    texts = pad_1d(texts)
    mels = pad_2d(mels)
    pitches = pad_1d(pitches)
    energies = pad_1d(energies)
    durations = pad_1d(durations)
    h_prosody_lens = np.array([p_emb.shape[0] for p_emb in h_prosody_emb]) if h_prosody_emb[0] is not None else None
    h_prosody_emb = pad_2d(h_prosody_emb) if h_prosody_emb[0] is not None else None
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
        np.array(h_txt_embs),
        np.array(h_txt_emb_lens),
        h_speakers,
        h_emotions,
        h_prosody_emb,
        h_prosody_lens,
        np.array(h_g_prosody_embs) if h_g_prosody_embs[0] is not None else None,
        h_prosody_speakers,
        h_prosody_emotions,
    )


def collate_fn_fastspeech2wContexts(batch, batch_size, speaker_dict=None, emotion_dict=None):
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

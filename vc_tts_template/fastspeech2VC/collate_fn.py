from typing import List, Dict, Callable
from pathlib import Path
from hydra.utils import to_absolute_path

from torch.utils import data as data_utils
import numpy as np

from vc_tts_template.utils import load_utt_list, pad_1d, pad_2d


class fastspeech2VC_Dataset(data_utils.Dataset):  # type: ignore
    """Dataset for numpy files

    Args:
        in_paths: List of paths to input files
        out_paths: List of paths to output files
    """

    def __init__(
        self,
        in_mel_paths: List,
        in_pitch_paths: List,
        in_energy_paths: List,
        in_sent_duration_dir: List,
        out_mel_paths: List,
        out_pitch_paths: List,
        out_energy_paths: List,
        out_duration_paths: List,
        out_sent_duration_dir: List
    ):
        self.in_mel_paths = in_mel_paths
        self.in_pitch_paths = in_pitch_paths
        self.in_energy_paths = in_energy_paths
        self.in_sent_duration_dir = in_sent_duration_dir
        self.out_mel_paths = out_mel_paths
        self.out_pitch_paths = out_pitch_paths
        self.out_energy_paths = out_energy_paths
        self.out_duration_paths = out_duration_paths
        self.out_sent_duration_dir = out_sent_duration_dir

    def __getitem__(self, idx: int):
        """Get a pair of input and target

        Args:
            idx: index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        return (
            self.in_mel_paths[idx].name.replace("-feats.npy", ""),
            np.load(self.in_mel_paths[idx]),
            np.load(self.in_pitch_paths[idx]),
            np.load(self.in_energy_paths[idx]),
            np.load(self.in_sent_duration_dir[idx]) if self.in_sent_duration_dir[idx].exists() else None,
            np.load(self.out_mel_paths[idx]),
            np.load(self.out_pitch_paths[idx]),
            np.load(self.out_energy_paths[idx]),
            np.load(self.out_duration_paths[idx]),
            np.load(self.out_sent_duration_dir[idx]) if self.out_sent_duration_dir[idx].exists() else None,
        )

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.in_mel_paths)


def fastspeech2VC_get_data_loaders(data_config: Dict, collate_fn: Callable) -> Dict[str, data_utils.DataLoader]:
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

        in_mel_paths = [in_dir / "mel" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        in_pitch_paths = [in_dir / "pitch" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        in_energy_paths = [in_dir / "energy" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        in_sent_duration_dir = [in_dir / "sent_duration" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_mel_paths = [out_dir / "mel" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_pitch_paths = [out_dir / "pitch" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_energy_paths = [out_dir / "energy" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_duration_paths = [out_dir / "duration" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_sent_duration_dir = [out_dir / "sent_duration" / f"{utt_id}-feats.npy" for utt_id in utt_ids]

        dataset = fastspeech2VC_Dataset(
            in_mel_paths,
            in_pitch_paths,
            in_energy_paths,
            in_sent_duration_dir,
            out_mel_paths,
            out_pitch_paths,
            out_energy_paths,
            out_duration_paths,
            out_sent_duration_dir,
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


def reprocess(batch, idxs, speaker_dict, emotion_dict, sentence_duration):
    file_names = [batch[idx][0] for idx in idxs]
    s_mels = [batch[idx][1] for idx in idxs]
    s_pitches = [batch[idx][2] for idx in idxs]
    s_energies = [batch[idx][3] for idx in idxs]
    t_mels = [batch[idx][5] for idx in idxs]
    t_pitches = [batch[idx][6] for idx in idxs]
    t_energies = [batch[idx][7] for idx in idxs]
    t_durations = [batch[idx][8] for idx in idxs]

    if speaker_dict is not None:
        s_speakers = np.array([speaker_dict[fname.split("_")[0]] for fname in file_names])
        t_speakers = np.array([speaker_dict[fname.split("_")[1]] for fname in file_names])
    else:
        s_speakers = np.array([0 for _ in idxs])
        t_speakers = np.array([0 for _ in idxs])
    if emotion_dict is not None:
        s_emotions = np.array([emotion_dict[fname.split("_")[-2]] for fname in file_names])
        t_emotions = np.array([emotion_dict[fname.split("_")[-1]] for fname in file_names])
    else:
        s_emotions = np.array([0 for _ in idxs])
        t_emotions = np.array([0 for _ in idxs])

    s_mel_lens = np.array([s_mel.shape[0] for s_mel in s_mels])
    s_mel_max_len = max(s_mel_lens)
    t_mel_lens = np.array([t_mel.shape[0] for t_mel in t_mels])
    t_mel_max_len = max(t_mel_lens)

    ids = np.array(file_names)

    s_mels = pad_2d(s_mels)
    s_pitches = pad_1d(s_pitches, s_mel_max_len)
    s_energies = pad_1d(s_energies, s_mel_max_len)
    t_mels = pad_2d(t_mels)
    t_pitches = pad_1d(t_pitches, t_mel_max_len)
    t_energies = pad_1d(t_energies, t_mel_max_len)
    t_durations = pad_1d(t_durations)

    if (batch[0][4] is not None) and (sentence_duration == 1):
        s_snt_durations = pad_1d([batch[idx][4] for idx in idxs])
        t_snt_durations = pad_1d([batch[idx][9] for idx in idxs])
    else:
        s_snt_durations = None
        t_snt_durations = None

    return (
        ids,
        s_speakers,
        t_speakers,
        s_emotions,
        t_emotions,
        s_mels,
        s_mel_lens,
        s_mel_max_len,
        s_pitches,
        s_energies,
        t_mels,
        t_mel_lens,
        t_mel_max_len,
        t_pitches,
        t_energies,
        t_durations,
        s_snt_durations,
        t_snt_durations,
    )


def collate_fn_fastspeech2VC(batch, batch_size, speaker_dict=None, emotion_dict=None, sentence_duration=0):
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
        output.append(reprocess(batch, idx, speaker_dict, emotion_dict, sentence_duration))

    return output

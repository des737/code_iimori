from typing import List, Tuple, Dict, Callable
from pathlib import Path
from hydra.utils import to_absolute_path

from torch.utils import data as data_utils
import numpy as np

from vc_tts_template.utils import load_utt_list, pad_1d, pad_2d


class fastspeech2_Dataset(data_utils.Dataset):  # type: ignore
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
        out_duration_paths: List
    ):
        self.in_paths = in_paths
        self.out_mel_paths = out_mel_paths
        self.out_pitch_paths = out_pitch_paths
        self.out_energy_paths = out_energy_paths
        self.out_duration_paths = out_duration_paths

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get a pair of input and target

        Args:
            idx: index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        return (
            self.in_paths[idx].name,
            np.load(self.in_paths[idx]),
            np.load(self.out_mel_paths[idx]),
            np.load(self.out_pitch_paths[idx]),
            np.load(self.out_energy_paths[idx]),
            np.load(self.out_duration_paths[idx]),
        )

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.in_paths)


def fastspeech2_get_data_loaders(data_config: Dict, collate_fn: Callable) -> Dict[str, data_utils.DataLoader]:
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

        in_feats_paths = [in_dir / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_mel_paths = [out_dir / "mel" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_pitch_paths = [out_dir / "pitch" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_energy_paths = [out_dir / "energy" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_duration_paths = [out_dir / "duration" / f"{utt_id}-feats.npy" for utt_id in utt_ids]

        dataset = fastspeech2_Dataset(
            in_feats_paths,
            out_mel_paths,
            out_pitch_paths,
            out_energy_paths,
            out_duration_paths,
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

    ids = np.array([fname.replace("-feats.npy", "") for fname in file_names])
    if speaker_dict is not None:
        speakers = np.array([speaker_dict[fname.split("_")[0]] for fname in ids])
    else:
        speakers = np.array([0 for _ in idxs])
    if emotion_dict is not None:
        emotions = np.array([emotion_dict[fname.split("_")[-1]] for fname in ids])
    else:
        emotions = np.array([0 for _ in idxs])

    # reprocessの内容をここに.

    text_lens = np.array([text.shape[0] for text in texts])
    mel_lens = np.array([mel.shape[0] for mel in mels])

    texts = pad_1d(texts)
    mels = pad_2d(mels)
    pitches = pad_1d(pitches)
    energies = pad_1d(energies)
    durations = pad_1d(durations)

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
    )


def collate_fn_fastspeech2(batch, batch_size, speaker_dict=None, emotion_dict=None):
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

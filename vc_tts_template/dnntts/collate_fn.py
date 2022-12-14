from typing import List, Tuple
import torch

from vc_tts_template.train_utils import pad_2d


def collate_fn_dnntts(batch: List) -> Tuple:
    """Collate function for DNN-TTS.

    Args:
        batch: List of tuples of the form (inputs, targets).

    Returns:
        tuple: Batch of inputs, targets, and lengths.
    """
    lengths = [len(x[0]) for x in batch]
    max_len = max(lengths)
    x_batch = torch.stack([torch.from_numpy(pad_2d(x[0], max_len)) for x in batch])
    y_batch = torch.stack([torch.from_numpy(pad_2d(x[1], max_len)) for x in batch])
    l_batch = torch.tensor(lengths, dtype=torch.long)
    return x_batch, y_batch, l_batch

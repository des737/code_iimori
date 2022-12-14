from typing import List, Optional, Union
import torch
import random
import numpy as np
from functools import partial
import torch.nn.functional as F


def adaptive_load_state_dict(
    model, state_dict, logger=None, load_modules=None
):
    model_state_dict = model.state_dict()

    if load_modules is None:
        # 通常読み込み
        for k in state_dict.keys():
            if k in model_state_dict.keys():
                if state_dict[k].shape != model_state_dict[k].shape:
                    if logger is not None:
                        logger.info(
                            f"""Skip loading pretrained parameter!! because the shape unmatched: {k},\n
                            required shape: {model_state_dict[k].shape},\n
                            loaded shape: {state_dict[k].shape}"""
                        )
                    else:
                        print(f"Skip loading parameter because the shape unmatched: {k}, "
                              f"required shape: {model_state_dict[k].shape}, "
                              f"loaded shape: {state_dict[k].shape}")
                else:
                    # 正常読み込み.
                    model_state_dict[k] = state_dict[k]
            else:
                if logger is not None:
                    logger.info(
                        f"Skip loading pretrained parameter!! because there is no {k} in your finetuning model"
                    )
                else:
                    print(f"Skip loading pretrained parameter!! because there is no {k} in your finetuning model")
        for k in model_state_dict.keys():
            if k not in state_dict.keys():
                if logger is not None:
                    logger.info(
                        f"Finetuning model parameter {k} left unattended, because there is no {k} in pretrained model"
                    )
                else:
                    print(
                        f"Finetuning model parameter {k} left unattended, because there is no {k} in pretrained model")
    else:
        # load_modulesに記載されたもののみ読み込む
        for load_module_name in load_modules:
            loaded = 0
            for k in state_dict.keys():
                if load_module_name in k:
                    if k not in model_state_dict.keys():
                        raise ValueError(f"There is no {k} in your finetuning model!!")
                    if state_dict[k].shape != model_state_dict[k].shape:
                        raise ValueError(f"""The {k}'s shape unmatched:,\n
                                required shape: {model_state_dict[k].shape},\n
                                loaded shape: {state_dict[k].shape}"""
                                         )
                    model_state_dict[k] = state_dict[k]
                    if logger is not None:
                        logger.info(
                            f"Load successful!! key: {k}"
                        )
                    else:
                        print(f"Load successful!! key: {k}")
                    loaded = 1
            if loaded == 0:
                raise ValueError(f"There is no {load_module_name} in your pretrained model!!")

    model.load_state_dict(model_state_dict)


def optional_tqdm(tqdm_mode: str, **kwargs):
    """Get a tqdm object.

    Args:
        tqdm_mode: tqdm mode
        **kwargs: keyword arguments for tqdm

    Returns:
        callable: tqdm object or an identity function
    """
    if tqdm_mode == "tqdm":
        from tqdm import tqdm

        return partial(tqdm, **kwargs)
    elif tqdm_mode == "tqdm-notebook":
        from tqdm.notebook import tqdm

        return partial(tqdm, **kwargs)

    return lambda x: x


def make_pad_mask(lengths: Union[torch.Tensor, List], maxlen: Optional[int] = None,
                  device: Optional[torch.device] = None) -> torch.Tensor:
    """Make mask for padding frames
    つまり, padした部分がTrueになっているようなmaskを返す.

    Args:
        lengths: list of lengths
        maxlen: maximum length. If None, use max value of lengths.

    Returns:
        torch.ByteTensor: mask
    """
    if not isinstance(lengths, torch.Tensor):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lengths = torch.Tensor(lengths).to(device)
    device = lengths.device

    batch_size = lengths.shape[0]
    if maxlen is None:
        maxlen = int(torch.max(lengths).item())

    ids = torch.arange(0, maxlen).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, maxlen)

    return mask


def make_non_pad_mask(lengths: Union[torch.Tensor, List], maxlen: Optional[int] = None) -> torch.Tensor:
    """Make mask for non-padding frames

    Args:
        lengths: list of lengths
        maxlen: maximum length. If None, use max value of lengths.

    Returns:
        torch.ByteTensor: mask
    """
    return ~make_pad_mask(lengths, maxlen)


def load_utt_list(utt_list: str) -> List[str]:
    """Load a list of utterances.

    Args:
        utt_list: path to a file containing a list of utterances
        あくまでファイル名を返す.

    Returns:
        List[str]: list of utterances
    """
    utt_ids = []
    with open(utt_list) as f:
        for utt_id in f:
            utt_id = utt_id.strip()
            if len(utt_id) > 0:
                utt_ids.append(utt_id)
    return utt_ids


def init_seed(seed: int) -> None:
    """Initialize random seed.

    Args:
        seed: random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_1d(x: Union[np.ndarray, List], max_len: Optional[int] = None, constant_values: Optional[int] = 0) -> np.ndarray:
    """Pad a 1d-tensor.

    Args:
        x: tensor to pad
        max_len: maximum length of the tensor
        constant_values: value to pad with. Default: 0

    Returns:
        torch.Tensor: padded tensor
    """
    def pad(x, max_len, constant_values):
        x = np.pad(
            x,
            (0, max_len - len(x)),
            mode="constant",
            constant_values=constant_values,
        )
        return x

    if type(x) is list:
        if max_len is None:
            max_len = max((len(x_) for x_ in x))
        x = np.stack([pad(x_, max_len, constant_values) for x_ in x])
    elif type(x) is np.ndarray:
        assert max_len is not None, "you cant pad without maxlen"
        x = pad(x, max_len, constant_values)
    else:
        raise RuntimeError

    return x  # type: ignore


def pad_2d(x: Union[np.ndarray, List], max_len: Optional[int] = None, constant_values: Optional[int] = 0) -> np.ndarray:
    """Pad a 2d-tensor.

    Args:
        x: tensor to pad. shape=(T, dim)
        max_len: maximum length of the tensor
        constant_values: value to pad with. Default: 0

    Returns:
        torch.Tensor: padded tensor
    """
    def pad(x, max_len, constant_values):
        x = np.pad(
            x,
            [(0, max_len - len(x)), (0, 0)],
            mode="constant",
            constant_values=constant_values,
        )
        return x

    if type(x) is list:
        max_len = max((x_.shape[0] for x_ in x))
        x = np.stack([pad(x_, max_len, constant_values) for x_ in x])
    elif type(x) is np.ndarray:
        assert max_len is not None, "you cant pad without maxlen"
        x = pad(x, max_len, constant_values)
    else:
        raise RuntimeError("please len(x.shape) < 3")

    return x  # type: ignore


def pad_3d(
    x: Union[np.ndarray, List], pad_axis: int,
    max_len: Optional[int] = None, constant_values: Optional[int] = 0
) -> np.ndarray:
    """Pad a 3d-tensor."""
    def pad(x, max_len, constant_values, axis):
        pad_filter = []
        for i in range(3):
            if i == axis:
                pad_filter.append((0, max_len - x.shape[axis]))
            else:
                pad_filter.append((0, 0))

        x = np.pad(
            x,
            pad_filter,
            mode="constant",
            constant_values=constant_values,
        )
        return x

    if type(x) is list:
        max_len = max((x_.shape[pad_axis] for x_ in x))
        x = np.stack([pad(x_, max_len, constant_values, pad_axis) for x_ in x])
    elif type(x) is np.ndarray:
        assert max_len is not None, "you cant pad without maxlen"
        x = pad(x, max_len, constant_values, pad_axis)
    else:
        raise RuntimeError("please len(x.shape) < 3")

    return x  # type: ignore


def pad(x: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """padding. batch入力を想定した, ネットワーク内で利用できるpad.
    """
    if max_length:
        max_len = max_length
    else:
        max_len = max([x[i].size(0) for i in range(len(x))])

    out_list = list()
    for batch in x:
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        else:
            raise ValueError("3次元以上には未対応です.")
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


class StandardScaler:
    """sklearn.preprocess.StandardScaler like class with only
    transform functionality
    Args:
        mean (np.ndarray): mean
        std (np.ndarray): standard deviation
    """

    def __init__(self, mean, var, scale):
        self.mean_ = mean
        self.var_ = var
        # NOTE: scale may not exactly same as np.sqrt(var)
        self.scale_ = scale

    def transform(self, x):
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x):
        return x * self.scale_ + self.mean_

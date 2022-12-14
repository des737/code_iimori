import logging
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import hydra
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.utils import data as data_utils
from torch.utils.tensorboard import SummaryWriter

from vc_tts_template.logger import getLogger
from vc_tts_template.utils import (adaptive_load_state_dict, init_seed,
                                   load_utt_list)
from recipes.common.fit_scaler import MultiSpeakerStandardScaler  # noqa: F401


def free_tensors_memory(x: List[torch.Tensor]):
    for x_ in x:
        del x_
    torch.cuda.empty_cache()


class check_grad_flow():
    """AMP実装の為に, grad_infのparamを特定できるようにするクラス
    仮定(設計)
        - (train step内) grad norm が NaNとなっていいのは, AMP時だけ.
            それ以外はモデルのバグなのでraise error
        - (self.report内) paramsが記録されていないのにreport呼び出し(つまりgrad norm is None)は,
            lossにNaNがあったということ.
            - これに関しては, AMPだとしてもよろしくないので, raise error.
            - 但し, optuna中であれば, パラメタが悪い(lossが発散に向かっているということ)なので, pruned
            - optunaだとしても致命的なエラーの可能性は十分あるため, warningでlog報告.
                発見次第確認した方が良さそう.
    """

    def __init__(self, logger, only_inf_grad=True) -> None:
        self.model_params: Dict[str, np.ndarray] = {}
        self.logger = logger
        self.num_step = 0
        self.only_inf_grad = only_inf_grad

    def set_params(self, named_parameters):
        self.num_step += 1
        for n, p in named_parameters:
            if p.requires_grad is True:
                if p.grad is None:
                    raise AttributeError(f"{n}' grad is None!!")
                p = p.grad.abs().mean().cpu().numpy()
                if (self.only_inf_grad is False) or (p == np.inf):
                    n = f"steps: {self.num_step}, param_name: " + n
                    self.model_params[n] = p

    def report(self, loss_values=None):
        if (len(self.model_params) == 0) and (loss_values is not None):
            if bool(np.isnan(np.array(list(loss_values.values()))).any()) is True:
                self.logger.error(
                    "the loss is NaN!! debug it!"
                )
                self._report_dict(loss_values, add_step=True)
                raise ValueError("loss value error")
            else:
                self.logger.info(
                    "There is no record of grads. Maybe only the grad norm is inf."
                )

        self._report_dict(loss_values)
        self._report_dict(self.model_params)
        self._reset()

    def _reset(self):
        self.model_params = {}

    def _report_dict(self, dict_, add_step=False, mode="debug"):
        if dict_ is not None:
            for k, v in dict_.items():
                if add_step is True:
                    k = f"steps: {self.num_step}, " + k
                if mode == "debug":
                    self.logger.debug(f"{k}: {v}")
                else:
                    self.logger.info(f"{k}: {v}")


def get_epochs_with_optional_tqdm(tqdm_mode: str, nepochs: int, last_epoch: int = 0) -> Iterable:
    """Get epochs with optional progress bar.

    Args:
        tqdm_mode: Progress bar mode.
          tqdmと書くか, それ以外か.
        nepochs: Number of epochs.

    Returns:
        iterable: Epochs.
    """
    if tqdm_mode == "tqdm":
        from tqdm import tqdm
        epochs = tqdm(
            range(last_epoch+1, last_epoch+nepochs+1), total=last_epoch+nepochs, initial=last_epoch, desc="epoch"
        )
    else:
        epochs = range(last_epoch+1, last_epoch+nepochs + 1)

    return epochs


def num_trainable_params(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in the model.

    Args:
        model: Model to count the number of trainable parameters.

    Returns:
        int: Number of trainable parameters.
    """
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])


def set_epochs_based_on_max_steps_(train_config: Dict, steps_per_epoch: int, logger: logging.Logger):
    """Set epochs based on max steps.

    Args:
        train_config: Train config.
        steps_per_epoch: Number of steps per epoch.
        logger: Logger.
    """
    logger.info(f"Number of iterations per epoch: {steps_per_epoch}")

    if train_config.max_train_steps < 0:  # type: ignore
        # Set max_train_steps based on nepochs
        max_train_steps = train_config.nepochs * steps_per_epoch  # type: ignore
        train_config.max_train_steps = max_train_steps  # type: ignore
        logger.info(
            "Number of max_train_steps is set based on nepochs: {}".format(
                max_train_steps
            )
        )
    else:
        # Set nepochs based on max_train_steps
        max_train_steps = train_config.max_train_steps  # type: ignore
        epochs = int(np.ceil(max_train_steps / steps_per_epoch))
        train_config.nepochs = epochs  # type: ignore
        logger.info(
            "Number of epochs is set based on max_train_steps: {}".format(epochs)
        )

    logger.info(f"Number of epochs: {train_config.nepochs}")  # type: ignore
    logger.info(f"Number of iterations: {train_config.max_train_steps}")  # type: ignore


def save_checkpoint(
    logger: logging.Logger, out_dir: Path, model: nn.Module, optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler._LRScheduler, epoch: int, train_iter: int,
    is_best: Optional[bool] = False, postfix: Optional[str] = ""
) -> None:
    """Save a checkpoint.

    Args:
        logger: Logger.
        out_dir: Output directory.
        model: Model.
        optimizer: Optimizer.
        epoch: Current epoch.
        is_best: Whether or not the current model is the best.
            Defaults to False.
        postfix: Postfix. Defaults to "".
    """
    if isinstance(model, dict):
        for key in model.keys():
            if isinstance(model[key], nn.DataParallel):
                model[key] = model[key].module
    else:
        if isinstance(model, nn.DataParallel):
            model = model.module

    out_dir.mkdir(parents=True, exist_ok=True)
    if is_best:
        path = out_dir / f"best_loss{postfix}.pth"
    else:
        path = out_dir / "epoch{:04d}{}.pth".format(epoch, postfix)

    if isinstance(model, dict):
        model_state_dict = {
            k: v.state_dict() for k, v in model.items()
        }
        torch.save(
            {
                "state_dict": model_state_dict,
                "optimizer_state": optimizer.state_dict(),
                "lr_scheduler_state": lr_scheduler.state_dict(),
                "last_epoch": epoch,
                "last_train_iter": train_iter
            },
            path,
        )
    else:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "lr_scheduler_state": lr_scheduler.state_dict(),
                "last_epoch": epoch,
                "last_train_iter": train_iter
            },
            path,
        )

    logger.info(f"Saved checkpoint at {path}")
    if not is_best:
        shutil.copyfile(path, out_dir / f"latest{postfix}.pth")


def ensure_divisible_by(feats: np.ndarray, N: int) -> np.ndarray:
    """Ensure that the number of frames is divisible by N.
    Args:
        feats (np.ndarray): Input features.
        N (int): Target number of frames.
    Returns:
        np.ndarray: Input features with number of frames divisible by N.
    """
    if N == 1:
        return feats
    mod = len(feats) % N
    if mod != 0:
        feats = feats[: len(feats) - mod]
    return feats


def moving_average_(model, model_test, beta=0.9999):
    """Exponential moving average (EMA) of model parameters.

    Args:
        model (torch.nn.Module): Model to perform EMA on.
        model_test (torch.nn.Module): Model to use for the test phase.
        beta (float, optional): [description]. Defaults to 0.9999.
    """
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


plot_cnt = 1


def plot_grad_flow(named_parameters, fig_name="", save_cnt=1, scale="linear"):
    # insert after loss.backward()
    global plot_cnt
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            avg_grad = p.grad.abs().mean().cpu().numpy()
            if scale == "log":
                ave_grads.append(np.log10(avg_grad+1e-6))
            else:
                ave_grads.append(avg_grad)

    if plot_cnt == 1:
        plt.figure(figsize=(50, 10))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.subplots_adjust(bottom=0.5)
    if plot_cnt % save_cnt == 0:
        plt.savefig(to_absolute_path(f"gradient_flow{fig_name}.png"))
        # init
        plt.figure(figsize=(30, 10))
    plot_cnt += 1


def plot_attention(alignment: torch.Tensor) -> plt.figure:
    """Plot attention.
    Args:
        alignment (np.ndarray): Attention.
    """
    fig, ax = plt.subplots()
    alignment = alignment.cpu().data.numpy().T
    im = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    plt.xlabel("Decoder time step")
    plt.ylabel("Encoder time step")
    return fig


def plot_mels(mels, titles=None):
    """mel: shape=(time, dim)
    """
    fig, axes = plt.subplots(len(mels), 1, squeeze=False, figsize=(10, 10*len(mels)))
    if titles is None:
        titles = [None for i in range(len(mels))]

    for i in range(len(mels)):
        mel = mels[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig


def plot_2d_feats(feats: torch.Tensor, title=None) -> plt.figure:
    """Plot 2D features.
    Args:
        feats (np.ndarray): Input features.
        title (str, optional): Title. Defaults to None.
    """
    feats = feats.cpu().data.numpy().T
    fig, ax = plt.subplots()
    im = ax.imshow(
        feats, aspect="auto", origin="lower", interpolation="none", cmap="viridis"
    )
    fig.colorbar(im, ax=ax)
    if title is not None:
        ax.set_title(title)
    return fig


def get_vocoder(device: torch.device, model_name: str,
                model_config_path: str, weight_path: str
                ) -> Optional[nn.Module]:
    """
    Args:
      model_config: 利用したいconfigへのpath
    """
    assert model_name in ["hifigan"]
    if model_name == "hifigan":
        model_config = OmegaConf.load(to_absolute_path(model_config_path))
        generator = hydra.utils.instantiate(model_config.netG).to(device)
        ckpt = torch.load(to_absolute_path(weight_path), map_location=device)
        generator.load_state_dict(ckpt["state_dict"]["netG"])
        generator.eval()
        generator.remove_weight_norm()
        return generator

    return None


def vocoder_infer(mels: torch.Tensor, speakers: List[str],
                  vocoder_dict: Dict,
                  mel_scaler_path: str,
                  max_wav_value: Optional[int] = None,
                  lengths: Optional[List[int]] = None
                  ) -> Optional[List[np.ndarray]]:
    """
    Args:
      mels: (batch, time, dim)
      vocoder_dict: {model_name: vocoder}
      max_wav_value: needed if you use hifigan
      lengths: needed if you want to trim waves.
        please check if you multiple sampling_rate.
    """
    model_name = list(vocoder_dict.keys())[0]
    assert model_name in ["hifigan"]
    vocoder = vocoder_dict[model_name]

    if model_name == "hifigan":
        scaler = joblib.load(to_absolute_path(mel_scaler_path))

        if hasattr(scaler, "multi_speaker") is True:
            if scaler.multi_speaker is False:
                speakers = ["" for _ in range(len(speakers))]

            device = mels.device
            mels = np.array([
                scaler.inverse_transform(mel.cpu().data.numpy(), spk) for mel, spk in zip(mels, speakers)
            ])  # type: ignore
        else:
            # 昔に作った, 通常StandardScaler対応
            device = mels.device
            mels = np.array([scaler.inverse_transform(mel.cpu().data.numpy()) for mel in mels])  # type: ignore
        mels = torch.Tensor(mels).to(device)
        with torch.no_grad():
            # 基本(time, dim)だが, hifiganはなぜか(dim, time)で扱う.
            wavs = vocoder(mels.transpose(1, 2)).squeeze(1)

        if max_wav_value is not None:
            wavs = (
                wavs.cpu().data.numpy()
                * max_wav_value
            ).astype("int16")
        else:
            wavs = wavs.cpu().data.numpy()
        wavs = [wav for wav in wavs]

        if lengths is not None:
            for i in range(len(mels)):
                wavs[i] = wavs[i][: lengths[i]]

        return wavs

    return None


class _Dataset(data_utils.Dataset):  # type: ignore
    """Dataset for numpy files

    Args:
        in_paths: List of paths to input files
        out_paths: List of paths to output files
    """

    def __init__(self, in_paths: List, out_paths: List, fname: bool):
        self.in_paths = in_paths
        self.out_paths = out_paths
        self.fname = fname

    def __getitem__(self, idx: int):
        """Get a pair of input and target

        Args:
            idx: index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        if self.fname is True:
            return (
                self.in_paths[idx].name.replace("-feats.npy", ""),
                np.load(self.in_paths[idx]), np.load(self.out_paths[idx])
            )
        return np.load(self.in_paths[idx]), np.load(self.out_paths[idx])

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.in_paths)


def _get_data_loaders(data_config: Dict, collate_fn: Callable, fname: bool = True) -> Dict[str, data_utils.DataLoader]:
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
        out_feats_paths = [out_dir / f"{utt_id}-feats.npy" for utt_id in utt_ids]

        dataset = _Dataset(in_feats_paths, out_feats_paths, fname)
        data_loaders[phase] = data_utils.DataLoader(
            dataset,
            batch_size=data_config.batch_size * data_config.group_size,  # type: ignore
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=data_config.num_workers,  # type: ignore
            shuffle=phase.startswith("train"),  # trainならTrue
        )

    return data_loaders


def _several_model_loader(config, device, logger, checkpoint):
    model_dict = {}
    for model_name in config.model.keys():
        model = hydra.utils.instantiate(config.model[model_name]).to(device)
        logger.info(model)
        logger.info(
            "Number of trainable params of {}: {:.3f} million".format(
                model_name, num_trainable_params(model) / 1000000.0
            )
        )
        if checkpoint is not None:
            adaptive_load_state_dict(
                model, checkpoint["state_dict"][model_name], logger,
                None if "load_modules" not in config.train.pretrained.keys()  # type:ignore
                else config.train.pretrained.load_modules  # type:ignore
            )
        if config.data_parallel:
            model = nn.DataParallel(model)
        model_dict[model_name] = model

    return model_dict


def setup(
    config: Dict, device: torch.device,
    collate_fn: Callable, get_dataloader: Optional[Callable] = None,
) -> Tuple:
    """Setup for traiining

    Args:
        config: configuration for training
        device: device to use for training
        collate_fn: function to collate mini-batches
        get_dataloader: function to get dataloader. if you need original dataloader.

    Returns:
        (tuple): tuple containing model, optimizer, learning rate scheduler,
            data loaders, tensorboard writer, and logger.

    .. note::

        書籍に記載のコードは、この関数を一部簡略化しています。
    """
    # NOTE: hydra は内部で stream logger を追加するので、二重に追加しないことに注意
    logger = getLogger(config.verbose, add_stream_handler=False)  # type: ignore

    logger.info(f"PyTorch version: {torch.__version__}")

    # CUDA 周りの設定
    if torch.cuda.is_available():
        from torch.backends import cudnn

        cudnn.benchmark = config.cudnn.benchmark  # type: ignore
        cudnn.deterministic = config.cudnn.deterministic  # type: ignore
        logger.info(f"cudnn.deterministic: {cudnn.deterministic}")
        logger.info(f"cudnn.benchmark: {cudnn.benchmark}")
        if torch.backends.cudnn.version() is not None:
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")

    logger.info(f"Random seed: {config.seed}")  # type: ignore
    init_seed(config.seed)  # type: ignore

    # (optional) 学習済みモデルの読み込み
    # ファインチューニングしたい場合
    pretrained_checkpoint = config.train.pretrained.checkpoint  # type: ignore
    checkpoint = None
    if pretrained_checkpoint is not None and len(pretrained_checkpoint) > 0:
        logger.info(
            "Fine-tuning! Loading a checkpoint: {}".format(pretrained_checkpoint)
        )
        checkpoint = torch.load(to_absolute_path(pretrained_checkpoint), map_location=device)

    # モデルのインスタンス化
    if len(config.model.keys()) == 1:  # type: ignore
        model = hydra.utils.instantiate(config.model.netG).to(device)  # type: ignore
        logger.info(model)
        logger.info(
            "Number of trainable params: {:.3f} million".format(
                num_trainable_params(model) / 1000000.0
            )
        )
        if checkpoint is not None:
            adaptive_load_state_dict(
                model, checkpoint["state_dict"], logger,
                None if "load_modules" not in config.train.pretrained.keys()  # type:ignore
                else config.train.pretrained.load_modules  # type:ignore
            )

        # 複数 GPU 対応
        if config.data_parallel:  # type: ignore
            model = nn.DataParallel(model)

    else:
        model = _several_model_loader(config, device, logger, checkpoint)

        # 複数 GPU 対応
        if config.data_parallel:  # type: ignore
            for k, v in model.items():
                model[k] = nn.DataParallel(v)

    # Optimizer
    # 例: config.train.optim.optimizer.name = "Adam"なら,
    # optim.Adamをしていることと等価.
    if 'name' in list(config.train.optim.optimizer.keys()):  # type: ignore
        optimizer_class = getattr(optim, config.train.optim.optimizer.name)  # type: ignore
        optimizer = optimizer_class(
            model.parameters(), **config.train.optim.optimizer.params  # type: ignore
        )
    else:
        # 自作だと判断.
        optimizer = hydra.utils.instantiate(config.train.optim.optimizer)  # type: ignore
        optimizer._set_model(model)

    # 学習率スケジューラ
    if 'name' in list(config.train.optim.lr_scheduler.keys()):  # type: ignore
        lr_scheduler_class = getattr(
            optim.lr_scheduler, config.train.optim.lr_scheduler.name  # type: ignore
        )
        lr_scheduler = lr_scheduler_class(
            optimizer, **config.train.optim.lr_scheduler.params  # type: ignore
        )
    else:
        # 自作だと判断.
        lr_scheduler = hydra.utils.instantiate(config.train.optim.lr_scheduler)  # type: ignore
        lr_scheduler._set_optimizer(optimizer)

    last_epoch = 0
    last_train_iter = 0
    if checkpoint is not None:
        # optimizerたちをresetするとしても, last_epochは引き継いでいた方が見やすい気がする.
        if "last_epoch" in checkpoint.keys():  # 後方互換性のため.
            last_epoch = checkpoint["last_epoch"]
        if "last_train_iter" in checkpoint.keys():  # 後方互換性のため.
            last_train_iter = checkpoint["last_train_iter"]
        if config.train.pretrained.optimizer_reset is True:  # type: ignore
            logger.info(
                "skipping loading optimizer and lr_scheduler's states!"
            )
        else:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            except ValueError:
                logger.info(
                    "skipping loading optimizer's states!"
                )
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])

    # loss
    loss = hydra.utils.instantiate(config.train.criterion)  # type: ignore

    # DataLoader
    if get_dataloader is None:
        data_loaders = _get_data_loaders(config.data, collate_fn, fname=True)  # type: ignore
    else:
        data_loaders = get_dataloader(config.data, collate_fn)  # type: ignore

    set_epochs_based_on_max_steps_(
        config.train, len(data_loaders["train"])*config.data.group_size, logger  # type: ignore
    )

    # Tensorboard の設定
    writer_tr = SummaryWriter(to_absolute_path(config.train.log_dir + "/train"))  # type: ignore
    writer_dv = SummaryWriter(to_absolute_path(config.train.log_dir + "/dev"))  # type: ignore
    writers = {"train": writer_tr, "dev": writer_dv}

    # config ファイルを保存しておく
    out_dir = Path(to_absolute_path(config.train.out_dir))  # type: ignore
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "model.yaml", "w") as f:
        OmegaConf.save(config.model, f)  # type: ignore
    with open(out_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    return model, optimizer, lr_scheduler, loss, data_loaders, writers, logger, last_epoch, last_train_iter


def save_checkpoint_old(
    logger: logging.Logger, out_dir: Path, model: nn.Module, optimizer: optim.Optimizer,
    epoch: int, is_best: Optional[bool] = False, postfix: Optional[str] = ""
) -> None:
    """Save a checkpoint.

    Args:
        logger: Logger.
        out_dir: Output directory.
        model: Model.
        optimizer: Optimizer.
        epoch: Current epoch.
        is_best: Whether or not the current model is the best.
            Defaults to False.
        postfix: Postfix. Defaults to "".
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    out_dir.mkdir(parents=True, exist_ok=True)
    if is_best:
        path = out_dir / f"best_loss{postfix}.pth"
    else:
        path = out_dir / "epoch{:04d}{}.pth".format(epoch, postfix)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )

    logger.info(f"Saved checkpoint at {path}")
    if not is_best:
        shutil.copyfile(path, out_dir / f"latest{postfix}.pth")


def setup_old(
    config: Dict, device: torch.device,
    collate_fn: Callable, get_dataloader: Optional[Callable] = None
) -> Tuple:
    """Setup for traiining
    tacorton, wavenet, dnntts用の, lossを再実装しないためのset

    Args:
        config: configuration for training
        device: device to use for training
        collate_fn: function to collate mini-batches
        get_dataloader: function to get dataloader. if you need original dataloader.

    Returns:
        (tuple): tuple containing model, optimizer, learning rate scheduler,
            data loaders, tensorboard writer, and logger.

    .. note::

        書籍に記載のコードは、この関数を一部簡略化しています。
    """
    # NOTE: hydra は内部で stream logger を追加するので、二重に追加しないことに注意
    logger = getLogger(config.verbose, add_stream_handler=False)  # type: ignore

    logger.info(f"PyTorch version: {torch.__version__}")

    # CUDA 周りの設定
    if torch.cuda.is_available():
        from torch.backends import cudnn

        cudnn.benchmark = config.cudnn.benchmark  # type: ignore
        cudnn.deterministic = config.cudnn.deterministic  # type: ignore
        logger.info(f"cudnn.deterministic: {cudnn.deterministic}")
        logger.info(f"cudnn.benchmark: {cudnn.benchmark}")
        if torch.backends.cudnn.version() is not None:
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")

    logger.info(f"Random seed: {config.seed}")  # type: ignore
    init_seed(config.seed)  # type: ignore

    # モデルのインスタンス化
    # ↓これでインスタンス化できるhydraすごすぎる.
    model = hydra.utils.instantiate(config.model.netG).to(device)  # type: ignore
    logger.info(model)
    logger.info(
        "Number of trainable params: {:.3f} million".format(
            num_trainable_params(model) / 1000000.0
        )
    )

    # (optional) 学習済みモデルの読み込み
    # ファインチューニングしたい場合
    pretrained_checkpoint = config.train.pretrained.checkpoint  # type: ignore
    if pretrained_checkpoint is not None and len(pretrained_checkpoint) > 0:
        logger.info(
            "Fine-tuning! Loading a checkpoint: {}".format(pretrained_checkpoint)
        )
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    # 複数 GPU 対応
    if config.data_parallel:  # type: ignore
        model = nn.DataParallel(model)

    # Optimizer
    # 例: config.train.optim.optimizer.name = "Adam"なら,
    # optim.Adamをしていることと等価.
    optimizer_class = getattr(optim, config.train.optim.optimizer.name)  # type: ignore
    optimizer = optimizer_class(
        model.parameters(), **config.train.optim.optimizer.params  # type: ignore
    )
    # 学習率スケジューラ
    lr_scheduler_class = getattr(
        optim.lr_scheduler, config.train.optim.lr_scheduler.name  # type: ignore
    )
    lr_scheduler = lr_scheduler_class(
        optimizer, **config.train.optim.lr_scheduler.params  # type: ignore
    )
    # DataLoader
    if get_dataloader is None:
        data_loaders = _get_data_loaders(config.data, collate_fn)  # type: ignore
    else:
        data_loaders = get_dataloader(config.data, collate_fn)  # type: ignore

    set_epochs_based_on_max_steps_(config.train, len(data_loaders["train"]), logger)  # type: ignore

    # Tensorboard の設定
    writer = SummaryWriter(to_absolute_path(config.train.log_dir))  # type: ignore

    # config ファイルを保存しておく
    # Pathめっちゃ有能じゃん
    out_dir = Path(to_absolute_path(config.train.out_dir))  # type: ignore
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.yaml", "w") as f:
        OmegaConf.save(config.model, f)  # type: ignore
    with open(out_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    return model, optimizer, lr_scheduler, data_loaders, writer, logger

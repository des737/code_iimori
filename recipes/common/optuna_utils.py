import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import hydra
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

sys.path.append("../..")
from vc_tts_template.logger import getLogger
from vc_tts_template.train_utils import (_get_data_loaders,
                                         set_epochs_based_on_max_steps_,
                                         get_epochs_with_optional_tqdm,
                                         check_grad_flow)
from vc_tts_template.utils import init_seed
from recipes.common.train_loop import _train_step


def update_param_dict(default_prarams, tuning_config, trial):
    if tuning_config is None:
        return default_prarams
    for key in tuning_config.keys():
        _suggest = getattr(trial, tuning_config[key].suggest)
        default_prarams[key] = _suggest(key, **tuning_config[key].params)
    return default_prarams


def get_model_with_trial(model_config, tuning_config, trial):
    model_config = update_param_dict(model_config, tuning_config, trial)
    return hydra.utils.instantiate(model_config)


def get_several_model_with_trial(config, device, trial):
    model_dict = {}
    tuning_config = config.tuning.model
    for model_name in config.model.keys():
        model = get_model_with_trial(config.model[model_name], tuning_config[model_name], trial).to(device)
        if config.data_parallel:
            model = nn.DataParallel(model)
        model_dict[model_name] = model
    return model_dict


def optuna_setup(
    config: Dict, device: torch.device,
    collate_fn: Callable, trial: optuna.trial.Trial,
    get_dataloader: Optional[Callable] = None,
) -> Tuple:
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
    if len(config.model.keys()) == 1:  # type: ignore
        model = get_model_with_trial(config.model.netG, config.tuning.model.netG, trial).to(device)  # type: ignore
        # 複数 GPU 対応
        if config.data_parallel:  # type: ignore
            model = nn.DataParallel(model)

    else:
        model = get_several_model_with_trial(config, device, trial)

    # 複数 GPU 対応
    if config.data_parallel:  # type: ignore
        model = nn.DataParallel(model)

    # Optimizer
    # 例: config.train.optim.optimizer.name = "Adam"なら,
    # optim.Adamをしていることと等価.
    if 'name' in list(config.train.optim.optimizer.keys()):  # type: ignore
        optimizer_class = getattr(optim, config.train.optim.optimizer.name)  # type: ignore
        _params = update_param_dict(config.train.optim.optimizer.params, config.tuning.optimizer, trial)  # type: ignore
        optimizer = optimizer_class(
            model.parameters(), **_params  # type: ignore
        )
    else:
        # 自作だと判断.
        _optimizer = update_param_dict(config.train.optim.optimizer, config.tuning.optimizer, trial)  # type: ignore
        optimizer = hydra.utils.instantiate(_optimizer)  # type: ignore
        optimizer._set_model(model)

    # 学習率スケジューラ
    if 'name' in list(config.train.optim.lr_scheduler.keys()):  # type: ignore
        lr_scheduler_class = getattr(
            optim.lr_scheduler, config.train.optim.lr_scheduler.name  # type: ignore
        )
        _params = update_param_dict(
            config.train.optim.lr_scheduler.params, config.tuning.lr_scheduler, trial  # type: ignore
        )
        lr_scheduler = lr_scheduler_class(
            optimizer, **_params  # type: ignore
        )
    else:
        # 自作だと判断.
        _lr_scheduler = update_param_dict(
            config.train.optim.lr_scheduler, config.tuning.lr_scheduler, trial  # type: ignore
        )
        lr_scheduler = hydra.utils.instantiate(_lr_scheduler)  # type: ignore
        lr_scheduler._set_optimizer(optimizer)

    # loss
    _loss = update_param_dict(config.train.criterion, config.tuning.criterion, trial)  # type: ignore
    loss = hydra.utils.instantiate(_loss)  # type: ignore

    # DataLoader
    if get_dataloader is None:
        data_loaders = _get_data_loaders(config.data, collate_fn)  # type: ignore
    else:
        data_loaders = get_dataloader(config.data, collate_fn)  # type: ignore

    set_epochs_based_on_max_steps_(
        config.train, len(data_loaders["train"])*config.data.group_size, logger  # type: ignore
    )

    # config ファイルを保存しておく
    out_dir = Path(to_absolute_path(config.train.out_dir))  # type: ignore
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "model.yaml", "w") as f:
        OmegaConf.save(config.model, f)  # type: ignore
    with open(out_dir / "tuning.yaml", "w") as f:
        OmegaConf.save(config.tuning, f)  # type: ignore
    with open(out_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    return model, optimizer, lr_scheduler, loss, data_loaders, logger


def _update_running_losses_(running_losses, loss_values):
    for key, val in loss_values.items():
        try:
            running_losses[key] += val
        except KeyError:
            running_losses[key] = val


def optuna_train_loop(config, to_device, model, optimizer, lr_scheduler, loss, data_loaders,
                      logger, trial, use_loss=["total_loss"], train_step=None, epoch_step=False):
    nepochs = config.train.nepochs
    scaler = torch.cuda.amp.GradScaler()
    grad_checker = check_grad_flow(logger=logger, only_inf_grad=True)

    for epoch in get_epochs_with_optional_tqdm(config.tqdm, nepochs):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            if isinstance(model, dict):
                # hifiganのように複数modelを持つ場合, dictで管理.
                for key in model.keys():
                    model[key].train() if train else model[key].eval()
            else:
                model.train() if train else model.eval()
            running_losses = {}  # epoch毎のloss. ここでresetしてるし.
            is_first = 1
            group_size = 0
            for batchs in tqdm(
                data_loaders[phase], desc=f"{phase} iter", leave=False
            ):
                for batch in batchs:
                    batch = to_device(batch, phase)
                    train_step = _train_step if train_step is None else train_step
                    loss_values = train_step(
                        model,
                        optimizer,
                        lr_scheduler,
                        train,
                        loss,
                        batch,
                        logger,
                        scaler,
                        grad_checker,
                    )
                    # lossを一気に足してためておく. 賢い.
                    _update_running_losses_(running_losses, loss_values)

                    if is_first == 1:
                        # 一番最初のではかる. 最後だと端数になってしまう恐れ.
                        group_size = len(batchs)
                    is_first = 0

            if phase == "dev":
                target_loss = np.sum([running_losses[loss_] for loss_ in use_loss])
                ave_loss = target_loss / (len(data_loaders[phase]) * group_size)
                if bool(np.isnan(ave_loss).any()) is True:
                    logger.warning("the target loss is NaN!! it had better to check your exp.")
                    raise optuna.TrialPruned()
                trial.report(ave_loss, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if epoch_step is True:
                lr_scheduler.step()

    return ave_loss

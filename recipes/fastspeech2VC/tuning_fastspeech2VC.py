import sys
from functools import partial

import hydra
import optuna
import torch
from omegaconf import DictConfig

sys.path.append("../..")
from recipes.common.optuna_utils import optuna_setup, optuna_train_loop
from recipes.fastspeech2VC.train_fastspeech2VC import (fastspeech2VC_train_step,
                                                       to_device)
from vc_tts_template.fastspeech2VC.collate_fn import (
    collate_fn_fastspeech2VC, fastspeech2VC_get_data_loaders)


def _objective(trial, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    collate_fn = partial(
        collate_fn_fastspeech2VC, batch_size=config.data.batch_size,
        speaker_dict=config.model.netG.speakers, emotion_dict=config.model.netG.emotions,
        sentence_duration=config.data.sentence_duration
    )
    model, optimizer, lr_scheduler, loss, data_loaders, logger = optuna_setup(
        config, device, collate_fn, trial, fastspeech2VC_get_data_loaders  # type: ignore
    )
    to_device_ = partial(to_device, device=device)
    last_loss = optuna_train_loop(config, to_device_, model, optimizer, lr_scheduler, loss, data_loaders,
                                  logger, trial, use_loss=config.tuning.target_loss,
                                  train_step=fastspeech2VC_train_step)

    return last_loss


@hydra.main(config_path="conf/train_fastspeech2VC", config_name="config")
def my_app(config: DictConfig) -> None:

    sampler = getattr(optuna.samplers, config.tuning.sampler.name)
    pruner = getattr(optuna.pruners, config.tuning.pruner.name)
    config.tuning.sampler.params = {} if config.tuning.sampler.params is None else config.tuning.sampler.params
    config.tuning.pruner.params = {} if config.tuning.pruner.params is None else config.tuning.pruner.params
    study = optuna.create_study(
        study_name=config.tuning.study_name,
        storage=config.tuning.storage,
        load_if_exists=True,
        sampler=sampler(**config.tuning.sampler.params),
        pruner=pruner(**config.tuning.pruner.params),
    )

    objective = partial(_objective, config=config)
    study.optimize(objective, n_trials=config.tuning.n_trials)


if __name__ == "__main__":
    my_app()

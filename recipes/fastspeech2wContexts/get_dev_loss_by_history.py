from pathlib import Path
import sys
from functools import partial
import warnings

import hydra
from hydra.utils import to_absolute_path
import torch
import torch.nn as nn
from omegaconf import DictConfig

sys.path.append("../..")
from vc_tts_template.fastspeech2wContexts.collate_fn import (
    collate_fn_fastspeech2wContexts, fastspeech2wContexts_get_data_loaders
)
from vc_tts_template.fastspeech2wContexts.collate_fn_PEProsody import (
    collate_fn_fastspeech2wPEProsody, fastspeech2wPEProsody_get_data_loaders
)
from vc_tts_template.train_utils import setup
from recipes.fastspeech2wContexts.train_fastspeech2wContexts import to_device
from recipes.common.fit_scaler import MultiSpeakerStandardScaler  # noqa: F401

warnings.simplefilter('ignore', UserWarning)


def get_dev_output(
    model,
    batch,
):
    """dev時にはpredしたp, eで計算してほしいので, オリジナルのtrain_stepに.
    """
    # Run forwaard
    with torch.cuda.amp.autocast():
        output = model(
            *batch[:-3],
            p_targets=None,
            e_targets=None,
            d_targets=batch[-1],
        )

    return output


@hydra.main(config_path="conf/train_fastspeech2", config_name="config")
def my_app(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 以下自由
    if ("PEProsody" in config.data.acoustic_model) or ("EmotionPredictor" in config.data.acoustic_model):
        _collate_fn = collate_fn_fastspeech2wPEProsody
        _get_data_loader = fastspeech2wPEProsody_get_data_loaders
    else:
        _collate_fn = collate_fn_fastspeech2wContexts
        _get_data_loader = fastspeech2wContexts_get_data_loaders

    collate_fn = partial(
        _collate_fn, batch_size=config.data.batch_size,
        speaker_dict=config.model.netG.speakers, emotion_dict=config.model.netG.emotions,
    )

    model, _, _, _, data_loaders, _, _, _, _ = setup(
        config, device, collate_fn, _get_data_loader  # type: ignore
    )

    # 以下固定
    hist_len_num = {}  # type: ignore
    hist_len_loss = {}
    mae_loss = nn.L1Loss()

    to_device_ = partial(to_device, device=device)
    for batchs in data_loaders["dev"]:
        for batch in batchs:
            batch = to_device_(batch, "dev")
            output = get_dev_output(
                model, batch
            )
            postnet_mel = output[1]
            mel_masks = output[7]
            target_mel = batch[-6]
            h_prosody_embs_len = batch[18]
            for hist_len in set(list(h_prosody_embs_len.cpu().numpy())):
                if hist_len not in hist_len_num.keys():
                    hist_len_num[hist_len] = 0
                    hist_len_loss[hist_len] = 0.0
                _hist_len_num = int(torch.sum(h_prosody_embs_len == hist_len))
                hist_len_num[hist_len] += _hist_len_num
                _postnet_mel = postnet_mel[h_prosody_embs_len == hist_len]
                _target_mel = target_mel[h_prosody_embs_len == hist_len]
                _mel_masks = mel_masks[h_prosody_embs_len == hist_len]
                _mel_masks = ~_mel_masks

                _postnet_mel = _postnet_mel.masked_select(_mel_masks.unsqueeze(-1))
                _target_mel = _target_mel.masked_select(_mel_masks.unsqueeze(-1))
                mel_loss = float(mae_loss(_postnet_mel, _target_mel))

                hist_len_loss[hist_len] += mel_loss * _hist_len_num

    output = []
    for hist_len in sorted(hist_len_num.keys()):
        hist_len_loss[hist_len] = hist_len_loss[hist_len] / hist_len_num[hist_len]
        output.append(f"histlen: {hist_len}, loss value: {hist_len_loss[hist_len]}\n")

    with open(Path(to_absolute_path(config.train.out_dir)) / "devloss_by_hist_len.txt", 'w') as f:
        f.writelines(output)


if __name__ == "__main__":
    my_app()

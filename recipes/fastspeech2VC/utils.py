from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import joblib
import torch
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from vc_tts_template.utils import adaptive_load_state_dict, pad_2d
from vc_tts_template.train_utils import plot_attention
from recipes.common.fit_scaler import MultiSpeakerStandardScaler  # noqa: F401


def get_alignment_model(
    model_config_path: str, pretrained_checkpoint: str,
    device: torch.device, in_scaler_path: str, out_scaler_path: str
):
    checkpoint = torch.load(to_absolute_path(pretrained_checkpoint), map_location=device)
    model_config = OmegaConf.load(to_absolute_path(model_config_path))
    model = hydra.utils.instantiate(model_config.netG).to(device)
    adaptive_load_state_dict(model, checkpoint["state_dict"])
    acoustic_in_scaler = joblib.load(to_absolute_path(in_scaler_path))
    acoustic_out_scaler = joblib.load(to_absolute_path(out_scaler_path))
    model.eval()

    return model, acoustic_in_scaler, acoustic_out_scaler


def get_alignment(
    model, device, acoustic_in_scaler, acoustic_out_scaler, src_mels, tgt_mels,
    s_sp_ids=None, t_sp_ids=None, s_em_ids=None, t_em_ids=None
):
    src_mels = [acoustic_in_scaler.transform(src_mel) for src_mel in src_mels]
    src_mel_lens = torch.tensor([src_mel.shape[0] for src_mel in src_mels], dtype=torch.long).to(device)
    src_mels = torch.from_numpy(pad_2d(src_mels)).to(device)
    max_src_mel_len = int(torch.max(src_mel_lens))

    tgt_mels = [acoustic_out_scaler.transform(tgt_mel) for tgt_mel in tgt_mels]
    tgt_mel_lens = torch.tensor([tgt_mel.shape[0] for tgt_mel in tgt_mels], dtype=torch.long).to(device)
    tgt_mels = torch.from_numpy(pad_2d(tgt_mels)).to(device)
    max_tgt_mel_len = int(torch.max(tgt_mel_lens))

    sort_idx = torch.argsort(-src_mel_lens)
    inv_sort_idx = torch.argsort(sort_idx).cpu().numpy()

    s_sp_ids = [model.speakers[s_sp_id] for s_sp_id in s_sp_ids] if model.speakers is not None else [0] * len(src_mels)
    t_sp_ids = [model.speakers[t_sp_id] for t_sp_id in t_sp_ids] if model.speakers is not None else [0] * len(src_mels)
    s_em_ids = [model.speakers[s_em_id] for s_em_id in s_em_ids] if model.speakers is not None else [0] * len(src_mels)
    t_em_ids = [model.speakers[t_em_id] for t_em_id in t_em_ids] if model.speakers is not None else [0] * len(src_mels)

    s_sp_ids = torch.tensor(s_sp_ids, dtype=torch.long).to(device)
    t_sp_ids = torch.tensor(t_sp_ids, dtype=torch.long).to(device)
    s_em_ids = torch.tensor(s_em_ids, dtype=torch.long).to(device)
    t_em_ids = torch.tensor(t_em_ids, dtype=torch.long).to(device)

    with torch.no_grad():
        attn_ws = model(
            None, s_sp_ids[sort_idx], t_sp_ids[sort_idx], s_em_ids[sort_idx], t_em_ids[sort_idx],
            src_mels[sort_idx], src_mel_lens[sort_idx], max_src_mel_len,
            tgt_mels[sort_idx], tgt_mel_lens[sort_idx], max_tgt_mel_len
        )[3].cpu().data.numpy()

    # 対角にあるとかは関係なく, encoder軸から見てargmaxになったものの個数をただ数える.
    # Fastspeech1の論文と同じ手法.
    durations = []
    src_mel_lens = src_mel_lens[sort_idx].cpu().numpy().astype(np.int16)
    tgt_mel_lens = tgt_mel_lens[sort_idx].cpu().numpy().astype(np.int16)
    reduction_factor = model.reduction_factor
    for i, attn_w in enumerate(attn_ws):
        attn_w = attn_w[:tgt_mel_lens[i]//reduction_factor, :src_mel_lens[i]//reduction_factor]
        duration = [np.sum(np.argmax(attn_w, axis=-1) == i) for i in range(attn_w.shape[1])]
        durations.append(np.array(duration))

        if np.sum(duration) != (tgt_mel_lens[i] // reduction_factor):
            print(f"""
                duration: {duration}\n
                np.sum(duration): {np.sum(duration)}\n
                max_tgt_mel_len: {max_tgt_mel_len}\n
            """)
            fig = plot_attention(torch.from_numpy(attn_w))
            fig.savefig("failed_attention_weight.png")
            raise RuntimeError

    return [durations[idx] for idx in inv_sort_idx]


def expand(values, durations, reduction_factor):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * (max(0, int(d))) * reduction_factor
    return np.array(out)


def plot_mel_with_prosody(data, titles, reduction_factor=1, need_expand=True):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy, duration = data[i]
        if need_expand is True:
            pitch = expand(pitch, duration, reduction_factor)
            energy = expand(energy, duration, reduction_factor)
        mel = mel[:, :np.sum(duration)*reduction_factor]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig

import sys
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.append('.')
from vc_tts_template.utils import make_pad_mask, pad


class VarianceAdaptor(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int,
        variance_predictor_filter_size: int,
        variance_predictor_kernel_size: int,
        variance_predictor_dropout: int,
        pitch_feature_level: int,
        energy_feature_level: int,
        pitch_quantization: str,
        energy_quantization: str,
        pitch_embed_kernel_size: int,
        pitch_embed_dropout: float,
        energy_embed_kernel_size: int,
        energy_embed_dropout: float,
        n_bins: int,
        stats: Optional[Dict],
    ):
        super(VarianceAdaptor, self).__init__()
        # duration, pitch, energyで共通なのね.
        self.duration_predictor = VariancePredictor(
            encoder_hidden_dim,
            variance_predictor_filter_size,
            variance_predictor_kernel_size,
            variance_predictor_dropout,
        )
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(
            encoder_hidden_dim,
            variance_predictor_filter_size,
            variance_predictor_kernel_size,
            variance_predictor_dropout,
        )
        self.energy_predictor = VariancePredictor(
            encoder_hidden_dim,
            variance_predictor_filter_size,
            variance_predictor_kernel_size,
            variance_predictor_dropout,
        )

        self.pitch_feature_level = "phoneme_level" if pitch_feature_level > 0 else "frame_level"
        self.energy_feature_level = "phoneme_level" if energy_feature_level > 0 else "frame_level"

        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        self.stats = stats

        if stats is not None:
            pitch_quantization = pitch_quantization
            energy_quantization = energy_quantization
            assert pitch_quantization in ["linear", "log"]
            assert energy_quantization in ["linear", "log"]
            pitch_min, pitch_max = stats["pitch_min"], stats["pitch_max"]
            energy_min, energy_max = stats["energy_min"], stats["energy_max"]

            if pitch_quantization == "log":
                self.pitch_bins = nn.Parameter(
                    torch.exp(
                        torch.linspace(np.log(pitch_min),
                                       np.log(pitch_max), n_bins - 1)
                    ),
                    requires_grad=False,
                )
            else:
                self.pitch_bins = nn.Parameter(
                    torch.linspace(pitch_min, pitch_max, n_bins - 1),
                    requires_grad=False,
                )
            if energy_quantization == "log":
                self.energy_bins = nn.Parameter(
                    torch.exp(
                        torch.linspace(np.log(energy_min),
                                       np.log(energy_max), n_bins - 1)
                    ),
                    requires_grad=False,
                )
            else:
                self.energy_bins = nn.Parameter(
                    torch.linspace(energy_min, energy_max, n_bins - 1),
                    requires_grad=False,
                )

            self.pitch_embedding = nn.Embedding(
                n_bins, encoder_hidden_dim
            )
            self.energy_embedding = nn.Embedding(
                n_bins, encoder_hidden_dim
            )
        else:
            self.pitch_embedding = nn.Sequential(  # type:ignore
                nn.Conv1d(
                    in_channels=1,
                    out_channels=encoder_hidden_dim,
                    kernel_size=pitch_embed_kernel_size,
                    padding=(pitch_embed_kernel_size - 1) // 2,
                ),
                nn.Dropout(pitch_embed_dropout),
            )
            self.energy_embedding = nn.Sequential(  # type:ignore
                nn.Conv1d(
                    in_channels=1,
                    out_channels=encoder_hidden_dim,
                    kernel_size=energy_embed_kernel_size,
                    padding=(energy_embed_kernel_size - 1) // 2,
                ),
                nn.Dropout(energy_embed_dropout),
            )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            if self.stats is not None:
                embedding = self.pitch_embedding(
                    torch.bucketize(target, self.pitch_bins))
            else:
                embedding = self.pitch_embedding(
                    target.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
        else:
            prediction = prediction * control
            if self.stats is not None:
                embedding = self.pitch_embedding(
                    torch.bucketize(prediction, self.pitch_bins)
                )
            else:
                embedding = self.pitch_embedding(
                    prediction.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            if self.stats is not None:
                embedding = self.energy_embedding(
                    torch.bucketize(target, self.energy_bins))
            else:
                embedding = self.energy_embedding(
                    target.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
        else:
            prediction = prediction * control
            if self.stats is not None:
                embedding = self.energy_embedding(
                    torch.bucketize(prediction, self.energy_bins))
            else:
                embedding = self.energy_embedding(
                    prediction.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        log_duration_prediction = self.duration_predictor(x, src_mask)

        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding

        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, e_control
            )
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = make_pad_mask(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """ Length Regulator
    https://github.com/xcmyz/FastSpeech/blob/master/modules.py
    """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        duration = duration.long()
        expand_max_len = torch.max(
            torch.sum(duration, -1), -1)[0]
        alignment = torch.zeros(duration.size(0),
                                expand_max_len,
                                duration.size(1)).numpy()
        alignment = self.create_alignment(alignment,
                                          duration.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if (max_len is not None) and (max_len > output.size(1)):
            output = pad(output, max_len)

        return output, torch.sum(duration, -1)

    def forward(self, x, duration, max_len=None):
        # durationがpredictか, targetのdurationです.
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len

    def create_alignment(self, base_mat, duration_predictor_output):
        N, L = duration_predictor_output.shape
        for i in range(N):
            count = 0
            for j in range(L):
                for k in range(duration_predictor_output[i][j]):
                    base_mat[i][count+k][j] = 1
                count = count + duration_predictor_output[i][j]
        return base_mat


class VariancePredictor(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int,
        variance_predictor_filter_size: int,
        variance_predictor_kernel_size: int,
        variance_predictor_dropout: int,
    ):
        super(VariancePredictor, self).__init__()

        self.input_size = encoder_hidden_dim
        self.filter_size = variance_predictor_filter_size
        self.kernel = variance_predictor_kernel_size
        self.conv_output_size = variance_predictor_filter_size
        self.dropout = variance_predictor_dropout

        self.conv_layer = nn.Sequential(
            OrderedDict(  # なんでわざわざ? 命名のためかな?
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,  # same sizeに.
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

# https://github.com/jinhan/tacotron2-vae
from typing import List
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('.')
from vc_tts_template.fastspeech2wVAE.layers import CoordConv2d


class VAE_GST(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int,
        ref_enc_dim: int,
        ref_enc_filters: List[int],
        ref_enc_kernel_size: int,
        ref_enc_stride: int,
        ref_enc_pad: int,
        n_mel_channels: int,
        ref_enc_gru_size: int,
        z_latent_dim: int,
    ):
        super().__init__()
        self.ref_encoder = ReferenceEncoder(
            ref_enc_dim, ref_enc_filters,
            ref_enc_kernel_size, ref_enc_stride, ref_enc_pad,
            n_mel_channels
        )
        self.fc1 = nn.Linear(ref_enc_gru_size, z_latent_dim)
        self.fc2 = nn.Linear(ref_enc_gru_size, z_latent_dim)
        self.fc3 = nn.Linear(z_latent_dim, encoder_hidden_dim)

        self.z_latent_dim = z_latent_dim

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, inputs, batch_size=None, device=None):
        if inputs is not None:
            enc_out = self.ref_encoder(inputs)
            mu = self.fc1(enc_out)
            logvar = self.fc2(enc_out)
            z = self.reparameterize(mu, logvar)
        else:
            mu = logvar = None
            z = torch.randn(batch_size, self.z_latent_dim).to(device)
        style_embed = self.fc3(z)

        return style_embed, mu, logvar


class ReferenceEncoder(nn.Module):
    def __init__(
        self,
        ref_enc_dim: int,
        ref_enc_filters: List[int],
        ref_enc_kernel_size: int,
        ref_enc_stride: int,
        ref_enc_pad: int,
        n_mel_channels: int,
    ):
        super().__init__()
        K = len(ref_enc_filters)
        filters = [1] + list(ref_enc_filters)
        # 最初のレイヤーとしてCoordConvを使用すると、positional情報を保存することがよくあります。https://arxiv.org/pdf/1811.02122.pdf
        convs = [CoordConv2d(in_channels=filters[0],
                             out_channels=filters[0 + 1],
                             kernel_size=ref_enc_kernel_size,
                             stride=ref_enc_stride,
                             padding=ref_enc_pad, with_r=True)]
        convs2 = [nn.Conv2d(in_channels=filters[i],
                            out_channels=filters[i + 1],
                            kernel_size=ref_enc_kernel_size,
                            stride=ref_enc_stride,
                            padding=ref_enc_pad) for i in range(1, K)]
        convs.extend(convs2)  # type: ignore
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(
            n_mel_channels, ref_enc_kernel_size, ref_enc_stride, ref_enc_pad, K
        )
        self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_channels,
                          hidden_size=ref_enc_dim // 2,
                          batch_first=True)
        self.n_mels = n_mel_channels

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.contiguous().view(N, 1, -1, self.n_mels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        _, out = self.gru(out)  # out --- [1, N, E//2]

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

import sys
from typing import Optional, Union

import torch
import torch.nn as nn
import numpy as np

sys.path.append("../..")
from vc_tts_template.fastspeech2wGMM.prosody_model import ProsodyPredictor
from vc_tts_template.tacotron.decoder import LocationSensitiveAttention
from vc_tts_template.utils import make_pad_mask
from vc_tts_template.fastspeech2wGMM.layers import GRUwSort, ConvBNorms2d
from vc_tts_template.fastspeech2wGMM.prosody_model import encoder_init


class ProsodyPredictorwAttention(ProsodyPredictor):
    def __init__(
        self,
        d_in=1,
        d_gru=1,  # dimention of gru
        d_out=1,  # output prosody emb size
        local_prosody=True,
        conv_out_channels=1,  # hidden channel before gru
        conv_kernel_size=1,
        conv_stride=1,
        conv_padding=None,
        conv_dilation=1,
        conv_bias=True,
        conv_n_layers=2,
        conv_dropout=0.2,
        gru_layers=2,
        zoneout=0.1,
        num_gaussians=10,
        softmax_temperature=1.0,
        global_prosody=False,
        global_gru_layers=1,
        global_d_gru=256,
        global_num_gaussians=10,
        global_softmax_temperature=1.0,
        h_prosody_emb_size=256,
        prosody_attention=True,
        attention_hidden_dim=512,
        attention_conv_channels=256,
        attention_conv_kernel_size=128,
        speaker_embedding=None,
        emotion_embedding=None,
    ) -> None:
        super().__init__(
            d_in, d_gru, d_out, local_prosody,
            conv_out_channels, conv_kernel_size, conv_stride,
            conv_padding, conv_dilation, conv_bias,
            conv_n_layers, conv_dropout, gru_layers,
            zoneout, num_gaussians, softmax_temperature,
            global_prosody, global_gru_layers, global_d_gru,
            global_num_gaussians, global_softmax_temperature,
        )
        if prosody_attention is True:
            if local_prosody is False:
                raise RuntimeError(
                    "do not set prosody_attention True when local_prosody is False"
                )
            self.attention = LocationSensitiveAttention(
                h_prosody_emb_size,
                d_gru,
                attention_hidden_dim,
                attention_conv_channels,
                attention_conv_kernel_size,
            )
            spk_emo_emb_size = 0
            if speaker_embedding is not None:
                spk_emo_emb_size += speaker_embedding.embedding_dim
            if emotion_embedding is not None:
                spk_emo_emb_size += emotion_embedding.embedding_dim
            self.attn_linear = nn.Linear(h_prosody_emb_size+spk_emo_emb_size, conv_out_channels+d_out)
        self.prosody_attention = prosody_attention
        self.speaker_embedding = speaker_embedding
        self.emotion_embedding = emotion_embedding

    def forward(
        self, encoder_output, h_prosody_emb, h_prosody_lens,
        h_prosody_speakers, h_prosody_emotions,
        target_prosody=None, target_global_prosody=None,
        src_lens=None, is_inference=False
    ):
        encoder_output = self.convnorms(encoder_output)

        if self.global_prosody is True:
            # hidden_global: (B, global_d_gru)
            hidden_global = self.global_bi_gru(encoder_output, src_lens)[:, -1, :]

            g_pi = self.g_pi_linear(hidden_global)
            g_sigma = (self.g_sigma_linear(hidden_global)+1.0).view(-1, self.global_num_gaussians, self.d_out)
            g_mu = self.g_mu_linear(hidden_global).view(-1, self.global_num_gaussians, self.d_out)
            if target_global_prosody is None:
                target_global_prosody = self.sample(g_pi, g_sigma, g_mu)
            else:
                target_global_prosody = target_global_prosody.detach()

        if self.local_prosody is False:
            if self.global_prosody is False:
                raise RuntimeError("you have to set True at least local or global prosody")
            outs = target_global_prosody.unsqueeze(1).expand(-1, encoder_output.size(1), -1)
            return outs, None, None, None, g_pi, g_sigma, g_mu

        # GRU の状態をゼロで初期化
        h_list = []
        for _ in range(len(self.gru)):
            h_list.append(self._zero_state(encoder_output))

        # 最初の入力
        go_frame = encoder_output.new_zeros(encoder_output.size(0), self.d_out)
        prev_out = go_frame

        if self.prosody_attention is True:
            # 1 つ前の時刻のアテンション重み
            prev_att_w = None
            self.attention.reset()

        pi_outs = []
        sigma_outs = []
        mu_outs = []
        outs = []

        if self.prosody_attention is True:
            h_prosody_mask = make_pad_mask(h_prosody_lens)

        for t in range(encoder_output.size()[1]):
            # Pre-Net
            if target_global_prosody is not None:
                prev_out = prev_out + target_global_prosody
            prenet_out = self.prenet(prev_out)

            # LSTM
            xs = torch.cat([encoder_output[:, t, :], prenet_out], dim=1)

            if self.prosody_attention is True:
                att_c, att_w = self.attention(
                    h_prosody_emb, h_prosody_lens, h_list[0], prev_att_w, h_prosody_mask
                )
                if self.speaker_embedding is not None:
                    att_c = torch.cat([att_c, self.speaker_embedding(h_prosody_speakers)], dim=-1)
                if self.emotion_embedding is not None:
                    att_c = torch.cat([att_c, self.emotion_embedding(h_prosody_emotions)], dim=-1)

                xs = xs + self.attn_linear(att_c)
            h_list[0] = self.gru[0](xs, h_list[0])
            for i in range(1, len(self.gru)):
                h_list[i] = self.gru[i](
                    h_list[i - 1], h_list[i]
                )
            hcs = torch.cat([h_list[-1], encoder_output[:, t, :]], dim=1)
            pi_outs.append(self.pi_linear(hcs).unsqueeze(1))
            sigma_outs.append((self.sigma_linear(hcs)+1.0).view(-1, 1, self.num_gaussians, self.d_out))
            mu_outs.append(self.mu_linear(hcs).view(-1, 1, self.num_gaussians, self.d_out))

            # 次の時刻のデコーダの入力を更新
            if (is_inference is True) or (target_prosody is None):
                prev_out = self.sample(
                    pi_outs[-1].squeeze(1), sigma_outs[-1].squeeze(1), mu_outs[-1].squeeze(1)
                )  # (B, d_out)
            else:
                # Teacher forcing
                # prevent from backpropagation to prosody extractor
                prev_out = target_prosody[:, t, :].detach()
            outs.append(prev_out.unsqueeze(1))

            if self.prosody_attention is True:
                # 累積アテンション重み
                prev_att_w = att_w if prev_att_w is None else prev_att_w + att_w

        outs = torch.cat(outs, dim=1)
        pi_outs = torch.cat(pi_outs, dim=1)
        sigma_outs = torch.cat(sigma_outs, dim=1)
        mu_outs = torch.cat(mu_outs, dim=1)
        if self.global_prosody is True:
            return outs, pi_outs, sigma_outs, mu_outs, g_pi, g_sigma, g_mu
        return outs, pi_outs, sigma_outs, mu_outs


class PEProsodyEncoder(nn.Module):
    """
    pitch, energyを受け取って, globalなprosodyへとencodeするクラス
    """
    def __init__(
        self,
        peprosody_encoder_gru_dim: int,
        peprosody_encoder_gru_num_layer: int,
        pitch_embedding: Optional[Union[nn.Sequential, nn.Embedding]],
        energy_embedding: Optional[Union[nn.Sequential, nn.Embedding]],
        pitch_bins: Optional[nn.Parameter] = None,
        energy_bins: Optional[nn.Parameter] = None,
        shere_embedding: bool = True,
        n_mel_channel: int = 80,
        conv_in_channels=1,
        conv_out_channels=1,
        conv_kernel_size=1,
        conv_stride=1,
        conv_padding=None,
        conv_dilation=1,
        conv_bias=True,
        conv_n_layers=2,
    ):
        super().__init__()

        if pitch_embedding is not None:
            # pitch, energyを用いる
            if pitch_bins is None:
                # n_binsがNoneなら, 渡されるpitch_embeddingはnn.Sequential
                self.hidden_sise = pitch_embedding[0].out_channels  # type:ignore
                gru_input_size = self.hidden_sise * 2  # type:ignore
            else:
                gru_input_size = pitch_embedding.embedding_dim * 2  # type:ignore

            self.pitch_bins = pitch_bins
            self.energy_bins = energy_bins
            if shere_embedding is True:
                self.pitch_embedding = pitch_embedding
                self.energy_embedding = energy_embedding
            else:
                if pitch_bins is None:
                    self.pitch_embedding = nn.Sequential(  # type:ignore
                        nn.Conv1d(
                            in_channels=1,
                            out_channels=pitch_embedding[0].out_channels,  # type:ignore
                            kernel_size=pitch_embedding[0].kernel_size[0],  # type:ignore
                            padding=(pitch_embedding[0].kernel_size[0] - 1) // 2,  # type:ignore
                        ),
                        nn.Dropout(pitch_embedding[1].p),  # type:ignore
                    )
                    self.energy_embedding = nn.Sequential(  # type:ignore
                        nn.Conv1d(
                            in_channels=1,
                            out_channels=energy_embedding[0].out_channels,  # type:ignore
                            kernel_size=energy_embedding[0].kernel_size[0],  # type:ignore
                            padding=(energy_embedding[0].kernel_size[0] - 1) // 2,  # type:ignore
                        ),
                        nn.Dropout(energy_embedding[1].p),  # type:ignore
                    )
                else:
                    self.pitch_embedding = nn.Embedding(
                        pitch_embedding.num_embeddings, pitch_embedding.embedding_dim  # type:ignore
                    )
                    self.energy_embedding = nn.Embedding(
                        energy_embedding.num_embeddings, energy_embedding.embedding_dim  # type:ignore
                    )
            self.use_mel = False
        else:
            # melを用いる
            gru_input_size = n_mel_channel
            self.hidden_sise = n_mel_channel
            self.mel_embedding = ConvBNorms2d(
                conv_in_channels, conv_out_channels, conv_kernel_size,
                conv_stride, conv_padding, conv_dilation, conv_bias, conv_n_layers
            )
            self.mel_embedding.apply(encoder_init)

            self.use_mel = True

        # 最初の発話の場合, hist_len=0になるため, allow_zeroが必要
        self.global_bi_gru = GRUwSort(
            input_size=gru_input_size, hidden_size=peprosody_encoder_gru_dim // 2,
            num_layers=peprosody_encoder_gru_num_layer, batch_first=True, bidirectional=True,
            sort=True, allow_zero_length=True, need_last=True
        )

    def forward(self, h_prosody_embs, h_prosody_embs_lens):
        # h_prosody_embs: (B, hist, time, 2 or 80)
        # h_prosody_embs_lens: (B, hist)

        hist_len = h_prosody_embs.size(1)

        if self.use_mel is False:
            # pitch, energyのembedding化
            h_pitches = h_prosody_embs[:, :, :, 0]
            h_energies = h_prosody_embs[:, :, :, 1]

            if self.pitch_bins is not None:
                h_pitch_embs = self.pitch_embedding(
                    torch.bucketize(h_pitches, self.pitch_bins)
                )
                h_energy_embs = self.energy_embedding(
                    torch.bucketize(h_energies, self.energy_bins)
                )
            else:
                h_pitches = h_pitches.view(-1, h_pitches.size(-1)).unsqueeze(-1)
                h_energies = h_energies.view(-1, h_energies.size(-1)).unsqueeze(-1)

                h_pitch_embs = self.pitch_embedding(
                    h_pitches.transpose(1, 2)
                ).transpose(1, 2).view(-1, hist_len, h_pitches.size(1), self.hidden_sise)
                h_energy_embs = self.energy_embedding(
                    h_energies.transpose(1, 2)
                ).transpose(1, 2).view(-1, hist_len, h_energies.size(1), self.hidden_sise)
            h_prosody_embs = torch.concat([h_pitch_embs, h_energy_embs], dim=-1)

        else:
            h_prosody_embs = h_prosody_embs.view(-1, h_prosody_embs.size(-2), h_prosody_embs.size(-1))
            h_prosody_embs = self.mel_embedding(
                h_prosody_embs.unsqueeze(1)
            ).squeeze(1).view(-1, hist_len, h_prosody_embs.size(1), self.hidden_sise)

        # GRUによるglobal prosody化
        h_prosody_embs = h_prosody_embs.contiguous().view(
            -1, h_prosody_embs.size(2), h_prosody_embs.size(3),
        )
        if type(h_prosody_embs_lens) == torch.Tensor:
            h_prosody_embs_lens = h_prosody_embs_lens.contiguous().view(-1)
        elif type(h_prosody_embs_lens) == np.ndarray:
            h_prosody_embs_lens = h_prosody_embs_lens.reshape(-1)

        h_g_prosody_embs = self.global_bi_gru(h_prosody_embs, h_prosody_embs_lens)
        h_g_prosody_embs = h_g_prosody_embs.contiguous().view(-1, hist_len, h_g_prosody_embs.size(-1))
        return h_g_prosody_embs


class PEProsodyEncoder_old(nn.Module):
    """
    pitch, energyを受け取って, globalなprosodyへとencodeするクラス
    昔のもの.
    """
    def __init__(
        self,
        peprosody_encoder_gru_dim: int,
        peprosody_encoder_gru_num_layer: int,
        pitch_embedding: Optional[Union[nn.Sequential, nn.Embedding]],
        energy_embedding: Optional[Union[nn.Sequential, nn.Embedding]],
        pitch_bins: Optional[nn.Parameter] = None,
        energy_bins: Optional[nn.Parameter] = None,
        shere_embedding: bool = True,
        n_mel_channel: int = 80,
        mel_emb_dim: int = 256,
        mel_emb_kernel: int = 31,
        mel_emb_dropout: float = 0.5,
    ):
        super().__init__()

        if pitch_embedding is not None:
            # pitch, energyを用いる
            if pitch_bins is None:
                # n_binsがNoneなら, 渡されるpitch_embeddingはnn.Sequential
                self.hidden_sise = pitch_embedding[0].out_channels  # type:ignore
                gru_input_size = self.hidden_sise * 2  # type:ignore
            else:
                gru_input_size = pitch_embedding.embedding_dim * 2  # type:ignore

            self.pitch_bins = pitch_bins
            self.energy_bins = energy_bins
            if shere_embedding is True:
                self.pitch_embedding = pitch_embedding
                self.energy_embedding = energy_embedding
            else:
                if pitch_bins is None:
                    self.pitch_embedding = nn.Sequential(  # type:ignore
                        nn.Conv1d(
                            in_channels=1,
                            out_channels=pitch_embedding[0].out_channels,  # type:ignore
                            kernel_size=pitch_embedding[0].kernel_size[0],  # type:ignore
                            padding=(pitch_embedding[0].kernel_size[0] - 1) // 2,  # type:ignore
                        ),
                        nn.Dropout(pitch_embedding[1].p),  # type:ignore
                    )
                    self.energy_embedding = nn.Sequential(  # type:ignore
                        nn.Conv1d(
                            in_channels=1,
                            out_channels=energy_embedding[0].out_channels,  # type:ignore
                            kernel_size=energy_embedding[0].kernel_size[0],  # type:ignore
                            padding=(energy_embedding[0].kernel_size[0] - 1) // 2,  # type:ignore
                        ),
                        nn.Dropout(energy_embedding[1].p),  # type:ignore
                    )
                else:
                    self.pitch_embedding = nn.Embedding(
                        pitch_embedding.num_embeddings, pitch_embedding.embedding_dim  # type:ignore
                    )
                    self.energy_embedding = nn.Embedding(
                        energy_embedding.num_embeddings, energy_embedding.embedding_dim  # type:ignore
                    )
            self.use_mel = False
        else:
            # melを用いる
            self.hidden_sise = mel_emb_dim
            gru_input_size = mel_emb_dim
            self.mel_embedding = nn.Sequential(  # type:ignore
                nn.Conv1d(
                    in_channels=n_mel_channel,
                    out_channels=mel_emb_dim,  # type:ignore
                    kernel_size=mel_emb_kernel,  # type:ignore
                    padding=(mel_emb_kernel - 1) // 2,  # type:ignore
                ),
                nn.Dropout(mel_emb_dropout),  # type:ignore
            )
            self.use_mel = True

        # 最初の発話の場合, hist_len=0になるため, allow_zeroが必要
        self.global_bi_gru = GRUwSort(
            input_size=gru_input_size, hidden_size=peprosody_encoder_gru_dim // 2,
            num_layers=peprosody_encoder_gru_num_layer, batch_first=True, bidirectional=True,
            sort=True, allow_zero_length=True, need_last=True
        )

    def forward(self, h_prosody_embs, h_prosody_embs_lens):
        # h_prosody_embs: (B, hist, time, 2 or 80)
        # h_prosody_embs_lens: (B, hist)

        hist_len = h_prosody_embs.size(1)

        if self.use_mel is False:
            # pitch, energyのembedding化
            h_pitches = h_prosody_embs[:, :, :, 0]
            h_energies = h_prosody_embs[:, :, :, 1]

            if self.pitch_bins is not None:
                h_pitch_embs = self.pitch_embedding(
                    torch.bucketize(h_pitches, self.pitch_bins)
                )
                h_energy_embs = self.energy_embedding(
                    torch.bucketize(h_energies, self.energy_bins)
                )
            else:
                h_pitches = h_pitches.view(-1, h_pitches.size(-1)).unsqueeze(-1)
                h_energies = h_energies.view(-1, h_energies.size(-1)).unsqueeze(-1)

                h_pitch_embs = self.pitch_embedding(
                    h_pitches.transpose(1, 2)
                ).transpose(1, 2).view(-1, hist_len, h_pitches.size(1), self.hidden_sise)
                h_energy_embs = self.energy_embedding(
                    h_energies.transpose(1, 2)
                ).transpose(1, 2).view(-1, hist_len, h_energies.size(1), self.hidden_sise)
            h_prosody_embs = torch.concat([h_pitch_embs, h_energy_embs], dim=-1)

        else:
            h_prosody_embs = h_prosody_embs.view(-1, h_prosody_embs.size(-2), h_prosody_embs.size(-1))
            h_prosody_embs = self.mel_embedding(
                h_prosody_embs.transpose(1, 2)
            ).transpose(1, 2).view(-1, hist_len, h_prosody_embs.size(1), self.hidden_sise)

        # GRUによるglobal prosody化
        h_prosody_embs = h_prosody_embs.contiguous().view(
            -1, h_prosody_embs.size(2), h_prosody_embs.size(3),
        )
        if type(h_prosody_embs_lens) == torch.Tensor:
            h_prosody_embs_lens = h_prosody_embs_lens.contiguous().view(-1)
        elif type(h_prosody_embs_lens) == np.ndarray:
            h_prosody_embs_lens = h_prosody_embs_lens.reshape(-1)

        h_g_prosody_embs = self.global_bi_gru(h_prosody_embs, h_prosody_embs_lens)
        h_g_prosody_embs = h_g_prosody_embs.contiguous().view(-1, hist_len, h_g_prosody_embs.size(-1))
        return h_g_prosody_embs


class PEProsodyLocalEncoder(nn.Module):
    """
    pitch, energyを受け取って, globalなprosodyへとencodeするクラス
    """
    def __init__(
        self,
        pitch_embedding: Optional[nn.Embedding],
        energy_embedding: Optional[nn.Embedding],
        pitch_bins: Optional[nn.Parameter] = None,
        energy_bins: Optional[nn.Parameter] = None,
        shere_embedding: bool = True,
        n_mel_channel: int = 80,
        mel_emb_dim: int = 256,
        mel_emb_kernel: int = 31,
        mel_emb_dropout: float = 0.5,
    ):
        super().__init__()

        if pitch_embedding is not None:
            self.pitch_bins = pitch_bins
            self.energy_bins = energy_bins
            if shere_embedding is True:
                self.pitch_embedding = pitch_embedding
                self.energy_embedding = energy_embedding
            else:
                if pitch_bins is None:
                    self.pitch_embedding = nn.Sequential(  # type:ignore
                        nn.Conv1d(
                            in_channels=1,
                            out_channels=pitch_embedding[0].out_channels,  # type:ignore
                            kernel_size=pitch_embedding[0].kernel_size[0],  # type:ignore
                            padding=(pitch_embedding[0].kernel_size[0] - 1) // 2,  # type:ignore
                        ),
                        nn.Dropout(pitch_embedding[1].p),  # type:ignore
                    )
                    self.energy_embedding = nn.Sequential(  # type:ignore
                        nn.Conv1d(
                            in_channels=1,
                            out_channels=energy_embedding[0].out_channels,  # type:ignore
                            kernel_size=energy_embedding[0].kernel_size[0],  # type:ignore
                            padding=(energy_embedding[0].kernel_size[0] - 1) // 2,  # type:ignore
                        ),
                        nn.Dropout(energy_embedding[1].p),  # type:ignore
                    )
                else:
                    self.pitch_embedding = nn.Embedding(
                        pitch_embedding.num_embeddings, pitch_embedding.embedding_dim  # type:ignore
                    )
                    self.energy_embedding = nn.Embedding(
                        energy_embedding.num_embeddings, energy_embedding.embedding_dim  # type:ignore
                    )
            self.use_mel = False
        else:
            # melを用いる
            self.hidden_sise = mel_emb_dim
            self.mel_embedding = nn.Sequential(  # type:ignore
                nn.Conv1d(
                    in_channels=n_mel_channel,
                    out_channels=mel_emb_dim,  # type:ignore
                    kernel_size=mel_emb_kernel,  # type:ignore
                    padding=(mel_emb_kernel - 1) // 2,  # type:ignore
                ),
                nn.Dropout(mel_emb_dropout),  # type:ignore
            )
            self.use_mel = True

    def forward(self, h_local_prosody_emb):
        # h_local_prosody_emb: (B, time, 2 or 80)
        # h_local_prosody_emb_lens: (B)

        if self.use_mel is False:
            # pitch, energyのembedding化
            h_local_pitch = h_local_prosody_emb[:, :, 0]
            h_local_energy = h_local_prosody_emb[:, :, 1]

            if self.pitch_bins is not None:
                h_local_pitch_emb = self.pitch_embedding(
                    torch.bucketize(h_local_pitch, self.pitch_bins)
                )
                h_local_energy_emb = self.energy_embedding(
                    torch.bucketize(h_local_energy, self.energy_bins)
                )
            else:
                h_local_pitch_emb = self.pitch_embedding(
                    h_local_pitch.unsqueeze(-1).transpose(1, 2)
                ).transpose(1, 2)
                h_local_energy_emb = self.energy_embedding(
                    h_local_energy.unsqueeze(-1).transpose(1, 2)
                ).transpose(1, 2)

            h_local_prosody_emb = torch.concat([h_local_pitch_emb, h_local_energy_emb], dim=-1)
        else:
            # h_local_prosody_emb: (B, time, mel_emb_dim)
            h_local_prosody_emb = self.mel_embedding(
                h_local_prosody_emb.transpose(1, 2)
            ).transpose(1, 2)

        return h_local_prosody_emb

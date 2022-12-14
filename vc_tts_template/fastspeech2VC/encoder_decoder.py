import sys

import torch.nn as nn

sys.path.append('.')
from vc_tts_template.fastspeech2VC.layers import ConformerBlock


class Encoder(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int,
        attention_dim: int,
        encoder_num_layer: int,
        encoder_num_head: int,
        conv_kernel_size: int,
        ff_expansion_factor: int,
        conv_expansion_factor: int,
        ff_dropout: float,
        attention_dropout: float,
        conv_dropout: float,
    ):
        super().__init__()

        d_model = encoder_hidden_dim
        d_attention = attention_dim
        n_head = encoder_num_head
        ff_expansion = ff_expansion_factor  # paper: 4
        conv_expansion = conv_expansion_factor  # paper: 2
        ff_dropout = ff_dropout  # FFT: 0.2
        attention_dropout = attention_dropout  # FFT: 0.2
        conv_dropout = conv_dropout  # FFT: 0.2
        kernel_size = conv_kernel_size
        n_layers = encoder_num_layer  # paper: 4

        self.layer_stack = nn.ModuleList(
            [
                ConformerBlock(
                    d_model, d_attention, n_head, ff_expansion, conv_expansion, ff_dropout,
                    attention_dropout, conv_dropout, kernel_size
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_output, mask, return_attns=False):
        enc_slf_attn_lst = []
        _, max_len, _ = enc_output.size()
        # ここでもとめられているmaskは2次元にしたmask.
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=slf_attn_mask
            )
            if return_attns is True:
                enc_slf_attn_lst += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    def __init__(
        self,
        decoder_hidden: int,
        attention_dim: int,
        decoder_num_layer: int,
        decoder_num_head: int,
        conv_kernel_size: int,
        ff_expansion_factor: int,
        conv_expansion_factor: int,
        ff_dropout: float,
        attention_dropout: float,
        conv_dropout: float,
    ):
        super().__init__()

        d_model = decoder_hidden
        d_attention = attention_dim
        n_head = decoder_num_head
        ff_expansion = ff_expansion_factor  # paper: 4
        conv_expansion = conv_expansion_factor  # paper: 2
        ff_dropout = ff_dropout  # FFT: 0.2
        attention_dropout = attention_dropout  # FFT: 0.2
        conv_dropout = conv_dropout  # FFT: 0.2
        kernel_size = conv_kernel_size
        n_layers = decoder_num_layer  # paper: 4

        self.layer_stack = nn.ModuleList(
            [
                ConformerBlock(
                    d_model, d_attention, n_head, ff_expansion, conv_expansion, ff_dropout,
                    attention_dropout, conv_dropout, kernel_size
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, dec_output, mask, return_attns=False):
        dec_slf_attn_lst = []
        _, max_len, _ = dec_output.size()
        # ここでもとめられているmaskは2次元にしたmask.
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=slf_attn_mask
            )
            if return_attns is True:
                dec_slf_attn_lst += [dec_slf_attn]

        return dec_output

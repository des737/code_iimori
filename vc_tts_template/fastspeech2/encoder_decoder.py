from typing import List
import torch
import torch.nn as nn
import numpy as np

from vc_tts_template.fastspeech2.layers import FFTBlock, WordEncoder


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    """ Encoder """

    def __init__(
        self,
        max_seq_len: int,
        num_vocab: int,  # pad=0
        encoder_hidden_dim: int,
        encoder_num_layer: int,
        encoder_num_head: int,
        conv_filter_size: int,
        conv_kernel_size: List[int],
        encoder_dropout: float,
        accent_info: int = 0,
    ):
        super(Encoder, self).__init__()

        # 以下, モデルたちの定義に使用.
        n_position = max_seq_len + 1
        n_src_vocab = num_vocab
        d_word_vec = encoder_hidden_dim
        n_layers = encoder_num_layer
        n_head = encoder_num_head
        d_k = d_v = (
            encoder_hidden_dim
            // encoder_num_head
        )
        d_model = encoder_hidden_dim
        d_inner = conv_filter_size
        kernel_size = conv_kernel_size
        dropout = encoder_dropout

        # 以下, forwardで使用.
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.src_word_emb = WordEncoder(
            n_src_vocab, d_word_vec, padding_idx=0, accent_info=accent_info
        )

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,  # 勾配計算を行わない.
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):
        """
        Args:
          src_seq: textデータ.
          mask: textデータの, 非padding成分を取り出したもの.
        """

        enc_slf_attn_list = []
        batch_size, max_len = mask.shape[0], mask.shape[1]

        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        if not self.training and max_len > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                max_len, self.d_model
            )[: max_len, :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            # あとは流しに行く.
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    """ Decoder """

    def __init__(
        self,
        max_seq_len: int,
        decoder_hidden_dim: int,
        decoder_num_layer: int,
        decoder_num_head: int,
        conv_filter_size: int,
        conv_kernel_size: List[int],
        decoder_dropout: float
    ):
        super(Decoder, self).__init__()

        # encodingのコピペやんけ!
        n_position = max_seq_len + 1
        d_word_vec = decoder_hidden_dim
        n_layers = decoder_num_layer
        n_head = decoder_num_head
        d_k = d_v = (
            decoder_hidden_dim
            // decoder_num_head
        )
        d_model = decoder_hidden_dim
        d_inner = conv_filter_size
        kernel_size = conv_kernel_size
        dropout = decoder_dropout

        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # ここも同じ.
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        # ここも同じ.
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, return_attns=False):
        """
        Args:
          enc_seq: variance_adaptorのoutputが入ってくる.
                   shape: (batch, max_seq(durationしたので700とか), hidden)
          mask: mel_masksが入ってくる. mel_lensがNoneでないとき.
          mel_lensは普通にNoneではないみたい.
        """
        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
        # -- Forward
        # 条件分岐も同じですね.
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
            # inferenceでmax_seq_lenを超えた場合は, ↓みたいにちょん切りはしない.
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

            # 以下がencoderと違う.
            # max_lenでちぎることはencodingでは行っていなかった.
            # どちらも同じFFTBlockなのに, ここがどう影響するんだろうか.
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]
        # 以下も一緒.
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask

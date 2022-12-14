import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from vc_tts_template.fastspeech2.varianceadaptor import LengthRegulator
from vc_tts_template.utils import make_pad_mask


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """

    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class ConvBNorms2d(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        n_layers=2,
    ):
        super().__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.convolutions = []

        self.convolutions += [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]

        for _ in range(n_layers - 1):
            self.convolutions += [
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=bias
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
        self.convolutions = nn.Sequential(*self.convolutions)

    def forward(self, x):
        # expected (B, C, H, W)
        return self.convolutions(x)


class ConvLNorms1d(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        n_layers=2,
        drop_out=0.2
    ):
        super().__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.convolutions = []

        self.convolutions += [
            Transpose(shape=(1, 2)),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.ReLU(),
            Transpose(shape=(1, 2)),
            nn.LayerNorm(out_channels),
            nn.Dropout(drop_out)
        ]

        for _ in range(n_layers - 1):
            self.convolutions += [
                Transpose(shape=(1, 2)),
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=bias
                ),
                nn.ReLU(),
                Transpose(shape=(1, 2)),
                nn.LayerNorm(out_channels),
                nn.Dropout(drop_out)
            ]
        self.convolutions = nn.Sequential(*self.convolutions)

    def forward(self, x):
        # expected (B, T, d)
        return self.convolutions(x)


class GRUwSort(nn.Module):
    """
    pack_padded_sequenceなどを内部でやってくれるクラスです.
    Examples:
        input:
        >>> x = torch.randn(1, 10, 1)
        >>> lens = torch.tensor([0]).long()
        args: allow_zero_length=True, need_last=False, keep_dim_1=False,
        output: zeros: (1, 0, hidden_size*2)

        args: allow_zero_length=True, need_last=True, keep_dim_1=False,
        output: zeros: (1, hidden_size*2)

        args: allow_zero_length=True, need_last=False, keep_dim_1=True,
        output: zeros: (1, 10, hidden_size*2)

        args: allow_zero_length=True, need_last=True, keep_dim_1=True,
        output: zeros: (1, hidden_size*2)
    """
    def __init__(self, input_size, hidden_size, num_layers,
                 batch_first, bidirectional, sort, dropout=0.0,
                 allow_zero_length=False, need_last=False,
                 keep_dim_1=False,) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=batch_first, bidirectional=bidirectional, dropout=dropout
        )
        self.sort = sort
        self.batch_first = batch_first
        self.zero_num = 0
        self.allow_zero_length = allow_zero_length
        self.need_last = need_last
        self.keep_dim_1 = keep_dim_1

        self.output_dim = hidden_size * 2 if bidirectional is True else hidden_size

        self.length_regulator = LengthRegulator()

    def forward(self, x, lens, y=None, y_lens=None):
        # yがある場合: x, yの中身を一続きにする. そのあと，y_lens分のtensorを最終時刻から得る.
        if y is not None:
            x, lens = self.concat_sequences(x, lens, y, y_lens)

        if type(lens) == torch.Tensor:
            lens = lens.to("cpu").numpy()
            if (y_lens is not None) and (type(y_lens) == torch.Tensor):
                y_lens = y_lens.to("cpu").numpy()

        if self.sort is True:
            sort_idx = np.argsort(-lens)
            inv_sort_idx = np.argsort(sort_idx)
            x = x[sort_idx]
            lens = lens[sort_idx]

        if self.allow_zero_length is True:
            x, lens, allzero_flg = self.remove_zeros(x, lens)
            if allzero_flg is True:
                dim_1_len = x.size(1) if (self.keep_dim_1 or self.need_last) is True else 0
                out = torch.zeros(
                    x.size(0), dim_1_len, self.output_dim,
                    dtype=x.dtype
                ).to(x.device)
                if self.need_last is True:
                    out = self.get_last_timestep(out, lens, y_lens)
                return out

        x_len = x.size(1)
        x = pack_padded_sequence(x, lens, batch_first=self.batch_first)
        out, _ = self.gru(x)
        out, _ = pad_packed_sequence(
            out, batch_first=self.batch_first,
            total_length=x_len if self.keep_dim_1 is True else None
        )

        if self.allow_zero_length is True:
            out = self.restore_zeros(out)

        if self.need_last is True:
            out = self.get_last_timestep(out, lens, y_lens)

        if self.sort is True:
            out = out[inv_sort_idx]

        return out

    def remove_zeros(self, x, lens):
        # 最新実装ではもはやremoveしていないことに注意
        self.zero_num = np.sum(lens == 0)
        if self.zero_num == 0:
            return x, lens, False
        # 1にしておけば, 最後のget_last_timestepでも実装が容易になる
        lens[-self.zero_num:] = 1

        if x.size(1) == 0:
            raise ValueError("未対応です. xのtimeは少なくとも1以上にしてください.")

        if len(lens) == self.zero_num:
            # all zero!!
            return x, lens, True

        return x, lens, False

    def restore_zeros(self, x):
        if self.zero_num > 0:
            x[-self.zero_num:] = 0.0
        return x

    def get_last_timestep(self, out, lens, sec_lens=None):
        # sec_lensで，最終時刻からいくつのデータを持っていくかを決める
        time_dimension = 1 if self.batch_first else 0

        if sec_lens is not None:
            idx = torch.arange(
                0, np.max(lens)
            ).unsqueeze(0).expand(out.size(0), -1).to(out.device)
            mask = idx >= torch.LongTensor(lens).unsqueeze(1).expand(-1, np.max(lens)).to(out.device)
            idx = idx.masked_fill(mask, np.max(lens)-1)
            mask = idx < torch.LongTensor(lens-sec_lens).unsqueeze(1).expand(-1, np.max(lens)).to(out.device)
            idx = idx.masked_fill(mask, np.max(lens)-1)
            idx, _ = torch.sort(idx, dim=-1)

            mask = make_pad_mask(sec_lens, maxlen=np.max(lens))
            out = out.gather(
                time_dimension, idx.unsqueeze(-1).expand_as(out)
            ).masked_fill(mask.unsqueeze(-1), 0.0)[:, :np.max(sec_lens)]
        else:
            idx = (torch.LongTensor(lens) - 1).view(-1, 1).expand(
                len(lens), out.size(-1)
            )
            idx = idx.unsqueeze(time_dimension).to(out.device)
            out = out.gather(
                time_dimension, Variable(idx)
            ).squeeze(time_dimension)
        return out

    def concat_sequences(self, x, lens, y, y_lens):
        x_idxes = 1 - make_pad_mask(lens).float()
        y_idxes = 1 - make_pad_mask(y_lens).float()
        idxes = torch.cat([x_idxes, y_idxes], dim=-1)
        _, idxes = torch.sort(-idxes, dim=-1, stable=True)

        lens = lens + y_lens
        x = torch.cat([x, y], dim=1)
        time_dimension = 1 if self.batch_first else 0
        x = x.gather(
            time_dimension, idxes.unsqueeze(-1).expand(idxes.size(0), idxes.size(1), x.size(-1))
        )
        return x, lens

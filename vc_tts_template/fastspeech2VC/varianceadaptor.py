import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('.')
from vc_tts_template.utils import make_pad_mask, pad
from vc_tts_template.tacotron.decoder import ZoneOutCell
from vc_tts_template.fastspeech2wGMM.prosody_model import mel2phone
from vc_tts_template.fastspeech2.varianceadaptor import LengthRegulator
from vc_tts_template.train_utils import free_tensors_memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int,
        variance_predictor_filter_size: int,
        variance_predictor_kernel_size_d: int,
        variance_predictor_layer_num_d: int,
        variance_predictor_kernel_size_p: int,
        variance_predictor_layer_num_p: int,
        variance_predictor_kernel_size_e: int,
        variance_predictor_layer_num_e: int,
        variance_predictor_dropout: int,
        stop_gradient_flow_d: bool,
        stop_gradient_flow_p: bool,
        stop_gradient_flow_e: bool,
        reduction_factor: int,
        pitch_AR: bool = False,
        pitch_ARNAR: bool = False,
        lstm_layers: int = 2,
    ):
        super(VarianceAdaptor, self).__init__()
        self.reduction_factor = reduction_factor
        self.pitch_AR = pitch_AR
        self.pitch_ARNAR = pitch_ARNAR

        # duration, pitch, energyで共通なのね.
        self.duration_predictor = VariancePredictor(
            encoder_hidden_dim,
            variance_predictor_filter_size,
            variance_predictor_kernel_size_d,
            variance_predictor_layer_num_d,
            variance_predictor_dropout,
        )
        if (pitch_AR is False) and (pitch_ARNAR is False):
            self.pitch_predictor = VariancePredictor(
                encoder_hidden_dim,
                variance_predictor_filter_size,
                variance_predictor_kernel_size_p,
                variance_predictor_layer_num_p,
                variance_predictor_dropout,
                reduction_factor
            )
        elif pitch_AR is True:
            self.pitch_predictor = VarianceARPredictor(  # type: ignore
                encoder_hidden_dim,
                variance_predictor_filter_size,
                variance_predictor_kernel_size_p,
                variance_predictor_layer_num_p,
                variance_predictor_dropout,
                reduction_factor,
                lstm_layers
            )
        elif pitch_ARNAR is True:
            self.pitch_predictor = VarianceARNARPredictor(  # type: ignore
                encoder_hidden_dim,
                variance_predictor_filter_size,
                variance_predictor_kernel_size_p,
                variance_predictor_layer_num_p,
                variance_predictor_dropout,
                reduction_factor,
                lstm_layers
            )
        self.energy_predictor = VariancePredictor(
            encoder_hidden_dim,
            variance_predictor_filter_size,
            variance_predictor_kernel_size_e,
            variance_predictor_layer_num_e,
            variance_predictor_dropout,
            reduction_factor
        )
        self.length_regulator = LengthRegulator()
        self.pitch_conv1d_1 = Conv_emb(self.reduction_factor, encoder_hidden_dim)
        self.pitch_conv1d_2 = Conv_emb(self.reduction_factor, encoder_hidden_dim)
        self.energy_conv1d_1 = Conv_emb(self.reduction_factor, encoder_hidden_dim)
        self.energy_conv1d_2 = Conv_emb(self.reduction_factor, encoder_hidden_dim)

        self.pitch_stop_gradient_flow = stop_gradient_flow_p
        self.energy_stop_gradient_flow = stop_gradient_flow_e
        self.duration_stop_gradient_flow = stop_gradient_flow_d

    def forward(
        self,
        x,
        src_mask,
        src_max_len,
        src_pitch,
        src_energy,
        src_duration=None,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        s_snt_durations=None,
        t_snt_durations=None,
    ):
        # まずは, durationを計算する.
        if self.duration_stop_gradient_flow is True:
            log_duration_prediction = self.duration_predictor(x.detach(), src_mask)
        else:
            log_duration_prediction = self.duration_predictor(x, src_mask)

        # convして, 次元を合わせる
        src_pitch = self.reshape_with_reduction_factor(src_pitch, src_max_len)
        src_energy = self.reshape_with_reduction_factor(src_energy, src_max_len)

        pitch_conv = self.pitch_conv1d_1(src_pitch)
        energy_conv = self.energy_conv1d_1(src_energy)

        if src_duration is not None:
            duration_rounded = src_duration
        else:
            duration_rounded = torch.clamp(  # 最小値を0にする. マイナスは許さない.
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )

        x, mel_len = self.length_regulator(x, duration_rounded, max_len)
        pitch, _ = self.length_regulator(pitch_conv, duration_rounded, max_len)
        energy, _ = self.length_regulator(energy_conv, duration_rounded, max_len)

        if mel_mask is None:
            mel_mask = make_pad_mask(mel_len)
            max_len = max(mel_len.cpu().numpy())

        pitch = pitch + x.detach() if self.pitch_stop_gradient_flow is True else pitch + x
        energy = energy + x.detach() if self.energy_stop_gradient_flow is True else energy + x

        # pitch, energy: (B, T//r, d_enc)
        if (self.pitch_AR is False) and (self.pitch_ARNAR is False):
            pitch_prediction = self.pitch_predictor(pitch, mel_mask) * p_control
        elif self.pitch_AR is True:
            pitch_prediction = self.pitch_predictor(pitch, mel_mask, pitch_target) * p_control
        elif self.pitch_ARNAR is True:
            if t_snt_durations is None:
                t_snt_durations = self.make_t_snt_durations(s_snt_durations, duration_rounded)
            pitch_prediction = self.pitch_predictor(pitch, mel_mask, t_snt_durations, pitch_target) * p_control

        energy_prediction = self.energy_predictor(energy, mel_mask) * e_control
        # pitchを, また次元増やしてhiddenに足す.
        if pitch_target is not None:
            pitch = self.reshape_with_reduction_factor(pitch_target, max_len)
            energy = self.reshape_with_reduction_factor(energy_target, max_len)
        else:
            pitch = self.reshape_with_reduction_factor(pitch_prediction, max_len)
            energy = self.reshape_with_reduction_factor(energy_prediction, max_len)

        pitch = self.pitch_conv1d_2(pitch)
        energy = self.energy_conv1d_2(energy)

        x = x + pitch + energy
        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )

    def reshape_with_reduction_factor(self, x, max_len):
        assert len(x.size()) == 2
        assert max_len*self.reduction_factor <= x.size(1), f"{max_len*self.reduction_factor}, {x.size(1)}"
        x = x[:, :max_len*self.reduction_factor]
        x = x.unsqueeze(-1).contiguous().view(x.size(0), -1, self.reduction_factor)
        return x

    def make_t_snt_durations(self, s_snt_durations, duration):
        duration = duration.detach().cpu().numpy()
        t_snt_durations = []
        for i, snt_duration in enumerate(s_snt_durations):
            t_snt_duration = []
            idx = 0
            for s_d in snt_duration:
                if s_d == 0:
                    break
                t_snt_duration.append(np.sum(duration[i, idx:idx+s_d]))
                idx += s_d
            t_snt_durations.append(torch.from_numpy(np.array(t_snt_duration)).long().to(s_snt_durations.device))
        return pad(t_snt_durations)

    def forward_woDuration(
        self,
        x,
        src_mask,
        src_max_len,
        src_pitch,
        src_energy,
        p_control=1.0,
        e_control=1.0,
        pitch_energy_prediction=False,
    ):
        # convして, 次元を合わせる
        pitch = self.reshape_with_reduction_factor(src_pitch, src_max_len)
        energy = self.reshape_with_reduction_factor(src_energy, src_max_len)

        if pitch_energy_prediction is True:
            pitch = self.pitch_conv1d_1(pitch)
            energy = self.energy_conv1d_1(energy)

            pitch = pitch + x.detach() if self.pitch_stop_gradient_flow is True else pitch + x
            energy = energy + x.detach() if self.energy_stop_gradient_flow is True else energy + x

            # pitch, energy: (B, T//r, d_enc)
            if (self.pitch_AR is False) and (self.pitch_ARNAR is False):
                pitch_prediction = self.pitch_predictor(pitch, src_mask) * p_control
            elif self.pitch_AR is True:
                pitch_prediction = self.pitch_predictor(pitch, src_mask, None) * p_control
            elif self.pitch_ARNAR is True:
                raise RuntimeError("you cant use this mode without duration")

            energy_prediction = self.energy_predictor(energy, src_mask) * e_control
            pitch = self.reshape_with_reduction_factor(pitch_prediction, src_max_len)
            energy = self.reshape_with_reduction_factor(energy_prediction, src_max_len)
        else:
            pitch = pitch * p_control
            energy = energy * e_control

        pitch = self.pitch_conv1d_2(pitch)
        energy = self.energy_conv1d_2(energy)

        x = x + pitch + energy
        return x


class VariancePredictor(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int,
        variance_predictor_filter_size: int,
        variance_predictor_kernel_size: int,
        variance_predictor_layer_num: int,
        variance_predictor_dropout: int,
        reduction_factor: int = 1,
    ):
        super(VariancePredictor, self).__init__()

        self.input_size = encoder_hidden_dim
        self.filter_size = variance_predictor_filter_size
        self.kernel = variance_predictor_kernel_size
        self.layer_num = variance_predictor_layer_num
        self.conv_output_size = variance_predictor_filter_size
        self.dropout = variance_predictor_dropout
        self.reduction_factor = reduction_factor

        conv_layers = []

        for i in range(self.layer_num):
            if i == 0:
                conv = Conv(
                    self.input_size,
                    self.filter_size,
                    kernel_size=self.kernel,
                    padding=(self.kernel - 1) // 2,  # same sizeに.
                )
            else:
                conv = Conv(
                    self.filter_size,
                    self.filter_size,
                    kernel_size=self.kernel,
                    padding=(self.kernel - 1) // 2,  # same sizeに.
                )
            conv_layers += [conv, nn.ReLU(), nn.LayerNorm(self.filter_size),
                            nn.Dropout(self.dropout)]

        self.conv_layer = nn.Sequential(*conv_layers)

        self.linear_layer = nn.Linear(self.conv_output_size, self.reduction_factor)

    def forward(self, encoder_output, mask):
        # encoder_output: (B, T//r, d_enc)
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        # out: (B, T//r, r)

        if mask is not None:
            if self.reduction_factor > 1:
                out = out.masked_fill(mask.unsqueeze(-1).expand(mask.size(0), mask.size(1), self.reduction_factor), 0.0)
                out = out.contiguous().view(out.size(0), -1, 1)
                out = out.squeeze(-1)

            else:
                out = out.squeeze(-1)
                out = out.masked_fill(mask, 0.0)

        # out: (B, T)
        return out


class VarianceARPredictor(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int,
        variance_predictor_filter_size: int,
        variance_predictor_kernel_size: int,
        variance_predictor_layer_num: int,
        variance_predictor_dropout: int,
        reduction_factor: int = 1,
        lstm_layers: int = 2,
        zoneout: float = 0.1
    ):
        super(VarianceARPredictor, self).__init__()

        self.input_size = encoder_hidden_dim
        self.filter_size = variance_predictor_filter_size
        self.kernel = variance_predictor_kernel_size
        self.layer_num = variance_predictor_layer_num
        self.dropout = variance_predictor_dropout
        self.reduction_factor = reduction_factor

        conv_layers = []

        for i in range(self.layer_num):
            if i == 0:
                conv = Conv(
                    self.input_size,
                    self.filter_size,
                    kernel_size=self.kernel,
                    padding=(self.kernel - 1) // 2,  # same sizeに.
                )
            else:
                conv = Conv(
                    self.filter_size,
                    self.filter_size,
                    kernel_size=self.kernel,
                    padding=(self.kernel - 1) // 2,  # same sizeに.
                )
            conv_layers += [conv, nn.ReLU(), nn.LayerNorm(self.filter_size),
                            nn.Dropout(self.dropout)]

        self.conv_layer = nn.Sequential(*conv_layers)

        # 片方向 LSTM
        self.lstm = nn.ModuleList()
        for layer in range(lstm_layers):
            lstm = nn.LSTMCell(
                self.filter_size*2 if layer == 0 else self.filter_size,
                self.filter_size,
            )
            self.lstm += [ZoneOutCell(lstm, zoneout)]

        self.prenet = nn.Linear(1, self.filter_size)

        self.linear_layer = nn.Linear(self.filter_size*2, self.reduction_factor)

    def forward(self, encoder_output, mask, target=None):
        # encoder_output: (B, T//r, d_enc)
        # target: (B, T)

        if target is not None:
            target = target[
                :, self.reduction_factor - 1:: self.reduction_factor
            ].unsqueeze(-1)
        # target: (B, T//r)

        encoder_output = self.conv_layer(encoder_output)

        # LSTM の状態をゼロで初期化
        h_list, c_list = [], []
        for _ in range(len(self.lstm)):
            h_list.append(self._zero_state(encoder_output))
            c_list.append(self._zero_state(encoder_output))

        # デコーダの最初の入力
        go_frame = encoder_output.new_zeros(encoder_output.size(0), 1)
        prev_out = go_frame
        outs = []

        # main loop
        for t in range(encoder_output.size()[1]):
            # Pre-Net
            prenet_out = self.prenet(prev_out)

            # LSTM
            xs = torch.cat([encoder_output[:, t, :], prenet_out], dim=1)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](
                    h_list[i - 1], (h_list[i], c_list[i])
                )
            hcs = torch.cat([h_list[-1], encoder_output[:, t, :]], dim=1)
            outs.append(self.linear_layer(hcs).unsqueeze(1))

            # 次の時刻のデコーダの入力を更新
            if target is None:
                prev_out = outs[-1][:, :, -1]  # (B, 1)
            else:
                # Teacher forcing
                prev_out = target[:, t]

        outs = torch.cat(outs, dim=1)
        # outs: (B, T//r, r)

        if mask is not None:
            if self.reduction_factor > 1:
                outs = outs.masked_fill(
                    mask.unsqueeze(-1).expand(mask.size(0), mask.size(1), self.reduction_factor), 0.0
                )
                outs = outs.contiguous().view(outs.size(0), -1)

            else:
                outs = outs.squeeze(-1)
                outs = outs.masked_fill(mask, 0.0)

        # outs: (B, T)
        return outs

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs


class VarianceARNARPredictor(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int,
        variance_predictor_filter_size: int,
        variance_predictor_kernel_size: int,
        variance_predictor_layer_num: int,
        variance_predictor_dropout: int,
        reduction_factor: int = 1,
        lstm_layers: int = 2,
        zoneout: float = 0.1
    ):
        super(VarianceARNARPredictor, self).__init__()

        self.input_size = encoder_hidden_dim
        self.filter_size = variance_predictor_filter_size
        self.kernel = variance_predictor_kernel_size
        self.layer_num = variance_predictor_layer_num
        self.dropout = variance_predictor_dropout
        self.reduction_factor = reduction_factor

        conv_layers = []

        for i in range(self.layer_num):
            if i == 0:
                conv = Conv(
                    self.input_size,
                    self.filter_size,
                    kernel_size=self.kernel,
                    padding=(self.kernel - 1) // 2,  # same sizeに.
                )
            else:
                conv = Conv(
                    self.filter_size,
                    self.filter_size,
                    kernel_size=self.kernel,
                    padding=(self.kernel - 1) // 2,  # same sizeに.
                )
            conv_layers += [conv, nn.ReLU(), nn.LayerNorm(self.filter_size),
                            nn.Dropout(self.dropout)]

        self.conv_layer = nn.Sequential(*conv_layers)

        # 片方向 LSTM
        self.lstm = nn.ModuleList()
        for layer in range(lstm_layers):
            lstm = nn.LSTMCell(
                self.filter_size*2 if layer == 0 else self.filter_size,
                self.filter_size,
            )
            self.lstm += [ZoneOutCell(lstm, zoneout)]

        self.prenet = nn.Linear(1, self.filter_size)

        self.linear_layer = nn.Linear(self.filter_size*2, self.reduction_factor)

    def lstm_forward(self, encoder_output, target):
        # LSTM の状態をゼロで初期化
        h_list, c_list = [], []
        for _ in range(len(self.lstm)):
            h_list.append(self._zero_state(encoder_output))
            c_list.append(self._zero_state(encoder_output))

        # デコーダの最初の入力
        go_frame = encoder_output.new_zeros(encoder_output.size(0), 1)
        prev_out = go_frame
        outs = []

        # main loop
        for t in range(encoder_output.size()[1]):
            # Pre-Net
            prenet_out = self.prenet(prev_out)

            # LSTM
            xs = torch.cat([encoder_output[:, t, :], prenet_out], dim=1)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](
                    h_list[i - 1], (h_list[i], c_list[i])
                )
            hcs = torch.cat([h_list[-1], encoder_output[:, t, :]], dim=1)
            outs.append(self.linear_layer(hcs).unsqueeze(1))

            # 次の時刻のデコーダの入力を更新
            if target is None:
                prev_out = outs[-1][:, :, -1]  # (B, 1)
            else:
                # Teacher forcing
                prev_out = target[:, t]

        outs = torch.cat(outs, dim=1)
        # outs: (B, T//r, r)

        return outs

    def forward(self, encoder_output, mask, durations, target=None):
        # encoder_output: (B, T//r, d_enc)
        # target: (B, T)
        if (self.reduction_factor > 1) and (target is not None):
            target = target[
                :, self.reduction_factor - 1:: self.reduction_factor
            ].unsqueeze(-1)
        # target: (B, T//r)
        encoder_output = self.conv_layer(encoder_output)
        durations = durations.detach().cpu().numpy()  # detach
        output_sorted, src_lens_sorted, segment_nums, inv_sort_idx = mel2phone(encoder_output, durations)
        if target is not None:
            target_sorted, _, _, _ = mel2phone(target, durations)
        else:
            target_sorted = [None]*len(output_sorted)
        free_tensors_memory([durations])
        outs = list()
        for output, target in zip(output_sorted, target_sorted):
            # output: (B, T_n, d). T_n: length of sentence duration
            out = self.lstm_forward(output, target)
            # out: (B, T_n//r, r)
            outs.append(out)
        free_tensors_memory([output_sorted])
        outs = self.seg2utter(outs, src_lens_sorted, inv_sort_idx, segment_nums)
        free_tensors_memory([src_lens_sorted, inv_sort_idx, segment_nums])
        if mask is not None:
            if self.reduction_factor > 1:
                outs = outs.masked_fill(
                    mask.unsqueeze(-1).expand(mask.size(0), mask.size(1), self.reduction_factor), 0.0
                )
                outs = outs.contiguous().view(outs.size(0), -1)

            else:
                outs = outs.squeeze(-1)
                outs = outs.masked_fill(mask, 0.0)

        # outs: (B, T)
        return outs

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def seg2utter(self, outs, src_lens_sorted, inv_sort_idx, segment_nums):
        # outs: List[(B, T_n//r, r)]. T_n are differenct
        # padding
        max_len = max([x.size(1) for x in outs])
        for i, out in enumerate(outs):
            out = F.pad(
                out, (0, 0, 0, max_len - out.size(1)), "constant", 0.0
            )
            outs[i] = out
        outs = torch.cat(outs, dim=0)
        # sortで元に戻し, 順番も元に戻す.
        # outs: (B*n, T_n//r, r), n=segment数
        outs = outs[inv_sort_idx]
        src_lens = []
        for src_lens_seg in src_lens_sorted:
            src_lens += src_lens_seg
        src_lens_sorted = np.array(src_lens)[inv_sort_idx]
        # 元に戻していく.
        output = []
        idx = 0
        for seg_num in segment_nums:
            utter = []
            for _ in range(seg_num):
                utter.append(outs[idx][:src_lens_sorted[idx]])
                idx += 1
            output.append(torch.cat(utter, dim=0))
        # output: List[(T//r, r)*B]
        output = pad(output)
        return output


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


class Conv_emb(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        dropout=0.2
    ):
        super().__init__()
        self.conv = Conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x

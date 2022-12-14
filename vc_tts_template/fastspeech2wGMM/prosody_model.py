import sys

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical

sys.path.append("../..")
from vc_tts_template.fastspeech2wGMM.layers import ConvBNorms2d, ConvLNorms1d, GRUwSort
from vc_tts_template.tacotron.decoder import ZoneOutCell
from vc_tts_template.utils import pad
from vc_tts_template.train_utils import free_tensors_memory


def encoder_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain("relu"))
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain("relu"))


class ProsodyExtractor(nn.Module):
    def __init__(
        self,
        d_mel=80,
        d_out=128,
        local_prosody=True,
        conv_in_channels=1,
        conv_out_channels=1,
        conv_kernel_size=1,
        conv_stride=1,
        conv_padding=None,
        conv_dilation=1,
        conv_bias=True,
        conv_n_layers=2,
        gru_n_layers=1,
        global_prosody=False,
        global_gru_n_layers=1
    ):
        super().__init__()
        self.convnorms = ConvBNorms2d(
            conv_in_channels, conv_out_channels, conv_kernel_size,
            conv_stride, conv_padding, conv_dilation, conv_bias, conv_n_layers
        )
        self.convnorms.apply(encoder_init)
        if local_prosody is True:
            self.bi_gru = GRUwSort(
                input_size=d_mel, hidden_size=d_out // 2, num_layers=gru_n_layers,
                batch_first=True, bidirectional=True, sort=False, need_last=True
            )
        if global_prosody is True:
            if local_prosody is False:
                self.global_linear = nn.Linear(d_mel, d_out)
            self.global_bi_gru = GRUwSort(
                input_size=d_out,
                hidden_size=d_out // 2, num_layers=global_gru_n_layers,
                batch_first=True, bidirectional=True, sort=True, need_last=True
            )
        self.local_prosody = local_prosody
        self.global_prosody = global_prosody

        self.segment_nums = None

    def forward(self, mels, durations, emb_lens=None):
        # expected(B, T, d_mel)
        if mels is None:
            if self.global_prosody is True:
                return None, None
            return None
        durations = durations.detach().cpu().numpy()  # detach
        if self.local_prosody is True:
            output_sorted, src_lens_sorted, segment_nums, inv_sort_idx = mel2phone(mels, durations)
            free_tensors_memory([durations])
            outs = list()
            for output, src_lens in zip(output_sorted, src_lens_sorted):
                # output: (B, T_n, d_mel). T_n: length of phoneme
                # convolution
                mel_hidden = self.convnorms(output.unsqueeze(1))
                # Bi-GRU
                out = self.bi_gru(mel_hidden.squeeze(1), src_lens)
                # out: (B, T_n, d_out)
                outs.append(out)
            free_tensors_memory([output_sorted, src_lens_sorted])
            outs = torch.cat(outs, 0)
            out = phone2utter(outs[inv_sort_idx], segment_nums)
            self.segment_nums = torch.from_numpy(np.array(segment_nums)).long().to(out.device)
            free_tensors_memory([outs])

            if self.global_prosody is True:
                # global_emb: (B, d_out)
                if emb_lens is None:
                    emb_lens = self.segment_nums
                global_emb = self.global_bi_gru(out, emb_lens)
                out = out + global_emb.unsqueeze(1).expand_as(out)
                return out, global_emb
            return out

        else:
            if self.global_prosody is False:
                raise RuntimeError("you have to set True at least local or global prosody")
            out = self.convnorms(mels.unsqueeze(1)).squeeze(1)
            emb_lens = np.sum(durations, axis=-1).astype(np.int16)
            global_emb = self.global_bi_gru(self.global_linear(out), emb_lens)
            out = global_emb.unsqueeze(1).expand(-1, durations.shape[1], -1)
            return out, global_emb


def mel2phone(mels, durations):
    """
    melをphone単位のsegmentに分割
    長い順に降順ソートして, batch_sizeごとにまとめている
    降順sortはpack_padded_sequenceの要請. batch_sizeごとにまとめたのはpadの量を削減するため
    """
    output = list()
    src_lens = list()
    segment_nums = list()
    # devide mel into phone segments
    for mel, duration in zip(mels, durations):
        s_idx = 0
        cnt = 0
        for d in duration:
            if d == 0:
                # d = 0 is only allowed for pad
                break
            mel_seg = mel[s_idx: s_idx+d]
            s_idx += d
            cnt += 1
            output.append(mel_seg)
            src_lens.append(d)
        segment_nums.append(cnt)

    sort_idx = np.argsort(-np.array(src_lens))
    inv_sort_idx = np.argsort(sort_idx)
    # Regroup phoneme segments into batch sizes.
    batch_size = mels.size(0)
    output_sorted = list()
    src_lens_sorted = list()
    for i in range(len(output) // batch_size):
        sort_seg = sort_idx[i*batch_size:(i+1)*batch_size]
        if i == ((len(output) // batch_size) - 1):
            # This is a way to avoid setting batch_size=1 when doing BN.
            sort_seg = sort_idx[i*batch_size:]
        output_seg = [output[idx] for idx in sort_seg]
        src_lens_seg = [src_lens[idx] for idx in sort_seg]
        output_seg = pad(output_seg)
        output_sorted.append(output_seg)
        src_lens_sorted.append(src_lens_seg)
    free_tensors_memory([output])
    return output_sorted, src_lens_sorted, segment_nums, inv_sort_idx


def phone2utter(out, segment_nums):
    """
    音素ごとのmel segmentを, utteranceごとにまとめ直す
    """
    output = list()
    s_idx = 0
    for seg_num in segment_nums:
        output.append(out[s_idx:s_idx+seg_num])
        s_idx += seg_num
    return pad(output)


class Div(nn.Module):
    def __init__(self, a) -> None:
        super().__init__()
        self.a = a

        if a < 1e-8:
            raise ValueError("too small!!")

    def forward(self, x):
        return torch.div(x, self.a)


class ProsodyPredictor(nn.Module):
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
        softmax_temperature=1,
        global_prosody=False,
        global_gru_layers=1,
        global_d_gru=256,
        global_num_gaussians=10,
        global_softmax_temperature=1,
    ) -> None:
        super().__init__()
        self.convnorms = ConvLNorms1d(
            d_in, conv_out_channels, conv_kernel_size,
            conv_stride, conv_padding, conv_dilation, conv_bias,
            conv_n_layers, conv_dropout
        )
        self.convnorms.apply(encoder_init)
        if local_prosody is True:
            # 片方向 gru
            self.gru = nn.ModuleList()
            for layer in range(gru_layers):
                gru = nn.GRUCell(
                    conv_out_channels+d_out if layer == 0 else d_gru,
                    d_gru,
                )
                self.gru += [ZoneOutCell(gru, zoneout, gru=True)]

            self.prenet = nn.Linear(d_out, d_out)

            self.pi_linear = nn.Sequential(
                nn.Linear(conv_out_channels+d_gru, num_gaussians),
                Div(softmax_temperature),
                nn.Softmax(dim=1)
            )
            self.sigma_linear = nn.Sequential(
                nn.Linear(conv_out_channels+d_gru, d_out*num_gaussians),
                nn.ELU(inplace=True)
            )
            self.mu_linear = nn.Linear(conv_out_channels+d_gru, d_out*num_gaussians)

        if global_prosody is True:
            self.global_bi_gru = GRUwSort(
                input_size=conv_out_channels, hidden_size=global_d_gru // 2,
                num_layers=global_gru_layers, batch_first=True, bidirectional=True,
                sort=False, need_last=True,
            )
            self.g_pi_linear = nn.Sequential(
                nn.Linear(global_d_gru, global_num_gaussians),
                Div(global_softmax_temperature),
                nn.Softmax(dim=1)
            )
            self.g_sigma_linear = nn.Sequential(
                nn.Linear(global_d_gru, d_out*global_num_gaussians),
                nn.ELU(inplace=True)
            )
            self.g_mu_linear = nn.Linear(global_d_gru, d_out*global_num_gaussians)

        self.d_out = d_out
        self.num_gaussians = num_gaussians
        self.local_prosody = local_prosody
        self.global_prosody = global_prosody
        self.global_num_gaussians = global_num_gaussians

    def forward(self, encoder_output, target_prosody=None, target_global_prosody=None,
                src_lens=None, is_inference=False):
        encoder_output = self.convnorms(encoder_output)

        if self.global_prosody is True:
            # hidden_global: (B, global_d_gru)
            hidden_global = self.global_bi_gru(encoder_output, src_lens)

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
        pi_outs = []
        sigma_outs = []
        mu_outs = []
        outs = []

        for t in range(encoder_output.size()[1]):
            # Pre-Net
            if target_global_prosody is not None:
                prev_out = prev_out + target_global_prosody
            prenet_out = self.prenet(prev_out)

            # LSTM
            xs = torch.cat([encoder_output[:, t, :], prenet_out], dim=1)
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

        outs = torch.cat(outs, dim=1)
        pi_outs = torch.cat(pi_outs, dim=1)
        sigma_outs = torch.cat(sigma_outs, dim=1)
        mu_outs = torch.cat(mu_outs, dim=1)
        if self.global_prosody is True:
            return outs, pi_outs, sigma_outs, mu_outs, g_pi, g_sigma, g_mu
        return outs, pi_outs, sigma_outs, mu_outs

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.gru[0].hidden_size)
        return init_hs

    def sample(self, pi, sigma, mu):
        # pi: (B, num_gaussians)
        # sigma: (B, num_gaussians, d_out)
        # mu: (B, num_gaussians, d_out)
        pis = OneHotCategorical(probs=pi).sample().unsqueeze(-1)
        # pis: (B, num_gaussians), one-hot.
        normal = Normal(loc=mu, scale=sigma+1e-7).sample()
        samples = torch.sum(pis*normal, dim=1)
        return samples


if __name__ == "__main__":
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ProsodyExtractor(d_mel=1, d_out=4).to(device)
    mels = torch.Tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        ]
    ).unsqueeze(-1).to(device)  # (B, T, mel_bin)
    durations = torch.Tensor(
        [
            [2, 2, 2, 2, 2],
            [2, 3, 2, 0, 0],
            [10, 0, 0, 0, 0]
        ]
    ).long().to(device)
    out = extractor(mels, durations)
    print("Extractor output: ", out.size())

    predictor = ProsodyPredictor(
        d_in=15,
        d_gru=10,
        d_out=30,
        conv_out_channels=5,
    ).to(device)
    encoder_output = torch.randn((3, 11, 15)).to(device)
    outs, pi_outs, sigma_outs, mu_outs = predictor(encoder_output)
    print("Pridictor output: ", outs.size())
    print("Pridictor pi output: ", pi_outs.size())
    print("Pridictor sigma output: ", sigma_outs.size())
    print("Pridictor mu output: ", mu_outs.size())

    # test gmm
    """
    参考(一個めがメイン)
    https://github.com/tonyduan/mdn/blob/master/mdn/models.py
    https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn/mdn.py
    """
    sampler = ProsodyPredictor()
    # 3山. 2次元でどうなるかを見てみる.
    pi = torch.Tensor(
        [[0.0, 0.2, 0.8],
         [0.0, 0.9, 0.1],
         [0.1, 0.8, 0.1]]
    )
    mu = torch.Tensor(
        [[[1, 0], [-1, 0], [0, 1]],
         [[1, 0], [-1, 0], [0, 1]],
         [[1, 0], [-1, 0], [0, 1]]]
    )
    sigma = torch.Tensor(
        [[[0.3, 0.1], [0.3, 0.1], [0.1, 0.3]],
         [[0.3, 0.1], [0.3, 0.1], [0.1, 0.3]],
         [[0.3, 0.1], [0.3, 0.1], [0.1, 0.3]]]
    )
    x_sample = []
    y_sample = []
    for _ in range(100):
        samples = sampler.sample(pi, sigma, mu).cpu().numpy()
        x_sample += samples[:, 0].tolist()
        y_sample += samples[:, 1].tolist()

    import matplotlib.pyplot as plt
    plt.scatter(x=x_sample, y=y_sample)
    plt.savefig("test.png")

    """memo

    20211006: extractorの勾配伝播確認. あのような変な操作も、微分可能。
    """

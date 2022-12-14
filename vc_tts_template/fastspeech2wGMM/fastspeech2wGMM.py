from typing import Dict, Optional

import torch
import torch.nn as nn

from vc_tts_template.fastspeech2.fastspeech2 import FastSpeech2
from vc_tts_template.fastspeech2wGMM.prosody_model import ProsodyExtractor, ProsodyPredictor


class FastSpeech2wGMM(FastSpeech2):
    """ FastSpeech2wGMM """

    def __init__(
        self,
        max_seq_len: int,
        num_vocab: int,  # pad=0
        # encoder
        encoder_hidden_dim: int,
        encoder_num_layer: int,
        encoder_num_head: int,
        conv_filter_size: int,
        conv_kernel_size_1: int,
        conv_kernel_size_2: int,
        encoder_dropout: float,
        # prosody extractor
        prosody_emb_dim: int,
        extra_conv_kernel_size: int,
        extra_conv_n_layers: int,
        extra_gru_n_layers: int,
        extra_global_gru_n_layers: int,
        # prosody predictor
        gru_hidden_dim: int,
        gru_n_layers: int,
        pp_conv_out_channels: int,
        pp_conv_kernel_size: int,
        pp_conv_n_layers: int,
        pp_conv_dropout: float,
        pp_zoneout: float,
        num_gaussians: int,
        softmax_temperature: float,
        global_gru_n_layers: int,
        global_d_gru: int,
        global_num_gaussians: int,
        global_softmax_temperature: float,
        # variance predictor
        variance_predictor_filter_size: int,
        variance_predictor_kernel_size: int,
        variance_predictor_dropout: int,
        pitch_feature_level: int,  # 0 is frame 1 is phoneme
        energy_feature_level: int,  # 0 is frame 1 is phoneme
        pitch_quantization: str,
        energy_quantization: str,
        pitch_embed_kernel_size: int,
        pitch_embed_dropout: float,
        energy_embed_kernel_size: int,
        energy_embed_dropout: float,
        n_bins: int,
        # decoder
        decoder_hidden_dim: int,
        decoder_num_layer: int,
        decoder_num_head: int,
        decoder_dropout: float,
        n_mel_channel: int,
        # other
        encoder_fix: bool = False,
        prosody_spk_independence: bool = False,
        local_prosody: bool = True,
        global_prosody: bool = True,
        stats: Optional[Dict] = None,
        speakers: Optional[Dict] = None,
        emotions: Optional[Dict] = None,
        accent_info: int = 0,
    ):
        super().__init__(
            max_seq_len,
            num_vocab,
            encoder_hidden_dim,
            encoder_num_layer,
            encoder_num_head,
            conv_filter_size,
            conv_kernel_size_1,
            conv_kernel_size_2,
            encoder_dropout,
            variance_predictor_filter_size,
            variance_predictor_kernel_size,
            variance_predictor_dropout,
            pitch_feature_level,
            energy_feature_level,
            pitch_quantization,
            energy_quantization,
            pitch_embed_kernel_size,
            pitch_embed_dropout,
            energy_embed_kernel_size,
            energy_embed_dropout,
            n_bins,
            decoder_hidden_dim,
            decoder_num_layer,
            decoder_num_head,
            decoder_dropout,
            n_mel_channel,
            encoder_fix,
            stats,
            speakers,
            emotions,
            accent_info,
        )
        self.prosody_extractor = ProsodyExtractor(
            n_mel_channel,
            prosody_emb_dim,
            local_prosody=local_prosody,
            conv_kernel_size=extra_conv_kernel_size,
            conv_n_layers=extra_conv_n_layers,
            gru_n_layers=extra_gru_n_layers,
            global_prosody=global_prosody,
            global_gru_n_layers=extra_global_gru_n_layers,
        )
        self.prosody_predictor = ProsodyPredictor(
            encoder_hidden_dim,
            gru_hidden_dim,
            prosody_emb_dim,
            local_prosody=local_prosody,
            conv_out_channels=pp_conv_out_channels,
            conv_kernel_size=pp_conv_kernel_size,
            conv_n_layers=pp_conv_n_layers,
            conv_dropout=pp_conv_dropout,
            gru_layers=gru_n_layers,
            zoneout=pp_zoneout,
            num_gaussians=num_gaussians,
            softmax_temperature=softmax_temperature,
            global_prosody=global_prosody,
            global_gru_layers=global_gru_n_layers,
            global_d_gru=global_d_gru,
            global_num_gaussians=global_num_gaussians,
            global_softmax_temperature=global_softmax_temperature,
        )
        self.prosody_linear = nn.Linear(
            prosody_emb_dim,
            encoder_hidden_dim,
        )

        self.global_prosody = global_prosody
        self.prosody_spk_independence = prosody_spk_independence

    def prosody_forward(
        self,
        output,
        src_lens,
        mels,
        p_targets,
        d_targets,
        speakers,
    ):
        is_inference = True if p_targets is None else False

        if (self.prosody_spk_independence is True) and (self.speaker_emb is not None):
            output = output - self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, output.size(1), -1
            )

        if self.global_prosody is False:
            prosody_target = self.prosody_extractor(mels, d_targets)
            prosody_prediction, pi_outs, sigma_outs, mu_outs = self.prosody_predictor(
                output, target_prosody=prosody_target, is_inference=is_inference
            )
        else:
            prosody_target, g_prosody_target = self.prosody_extractor(mels, d_targets, src_lens)
            prosody_prediction, pi_outs, sigma_outs, mu_outs, g_pi, g_sigma, g_mu = self.prosody_predictor(
                output, target_prosody=prosody_target, target_global_prosody=g_prosody_target,
                src_lens=src_lens, is_inference=is_inference
            )
        if is_inference is True:
            output = output + self.prosody_linear(prosody_prediction)
        else:
            output = output + self.prosody_linear(prosody_target)

        if (self.prosody_spk_independence is True) and (self.speaker_emb is not None):
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, output.size(1), -1
            )

        if self.global_prosody is True:
            return (
                output,
                [prosody_target,
                 pi_outs,
                 sigma_outs,
                 mu_outs,
                 g_prosody_target,
                 g_pi,
                 g_sigma,
                 g_mu]
            )
        else:
            return (
                output,
                [prosody_target,
                 pi_outs,
                 sigma_outs,
                 mu_outs]
            )

    def forward(
        self,
        ids,
        speakers,
        emotions,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_lens, max_src_len, src_masks, mel_lens, max_mel_len, mel_masks = self.init_forward(
            src_lens, max_src_len, mel_lens, max_mel_len
        )
        output = self.encoder_forward(
            texts, src_masks, max_src_len, speakers, emotions
        )

        output, prosody_features = self.prosody_forward(
            output, src_lens, mels, p_targets, d_targets, speakers
        )
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, postnet_output, mel_masks = self.decoder_forward(
            output, mel_masks
        )

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            prosody_features,
        )

    def prosody_teacher_forcing(
        self,
        ids,
        speakers,
        emotions,
        texts,
        src_lens,
        max_src_len,
        mels,
        d_targets,
    ):
        # prosodyのみteacher forcingする場合.
        # init
        mel_lens = torch.tensor([mel.size(0) for mel in mels]).long().to(mels.device)
        max_mel_len = torch.max(mel_lens)

        src_lens, max_src_len, src_masks, mel_lens, max_mel_len, mel_masks = self.init_forward(
            src_lens, max_src_len, mel_lens, max_mel_len
        )
        output = self.encoder_forward(
            texts, src_masks, max_src_len, speakers, emotions
        )

        output, prosody_features = self.prosody_forward(
            output, src_lens, mels, True, d_targets, speakers,
        )
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            None,
            None,
            d_targets,
            1.0,
            1.0,
            1.0,
        )

        output, postnet_output, mel_masks = self.decoder_forward(
            output, mel_masks
        )

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            prosody_features,
        )

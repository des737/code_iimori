from typing import Dict, Optional

import torch.nn as nn
import torch

from vc_tts_template.fastspeech2.varianceadaptor import LengthRegulator
from vc_tts_template.fastspeech2VC.fastspeech2VC import fastspeech2VC
from vc_tts_template.fastspeech2wGMM.prosody_model import ProsodyExtractor
from vc_tts_template.fastspeech2VCwGMM.prosody_model import ProsodyPredictor
from vc_tts_template.train_utils import free_tensors_memory


class fastspeech2VCwGMM(fastspeech2VC):
    """ FastSpeech2 """

    def __init__(
        self,
        n_mel_channel,
        # enoder_decoder
        attention_dim: int,
        encoder_hidden_dim: int,
        encoder_num_layer: int,
        encoder_num_head: int,
        decoder_hidden_dim: int,
        decoder_num_layer: int,
        decoder_num_head: int,
        conv_kernel_size: int,
        ff_expansion_factor: int,
        conv_expansion_factor: int,
        ff_dropout: float,
        attention_dropout: float,
        conv_dropout: float,
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
        global_gru_n_layers: int,
        global_d_gru: int,
        global_num_gaussians: int,
        # varianceadaptor
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

        encoder_fix: bool = False,
        decoder_fix: bool = False,
        global_prosody: bool = True,
        pitch_AR: bool = False,
        pitch_ARNAR: bool = False,
        lstm_layers: int = 2,
        speakers: Optional[Dict] = None,
        emotions: Optional[Dict] = None
    ):
        super().__init__(
            n_mel_channel,
            attention_dim,
            encoder_hidden_dim,
            encoder_num_layer,
            encoder_num_head,
            decoder_hidden_dim,
            decoder_num_layer,
            decoder_num_head,
            conv_kernel_size,
            ff_expansion_factor,
            conv_expansion_factor,
            ff_dropout,
            attention_dropout,
            conv_dropout,
            variance_predictor_filter_size,
            variance_predictor_kernel_size_d,
            variance_predictor_layer_num_d,
            variance_predictor_kernel_size_p,
            variance_predictor_layer_num_p,
            variance_predictor_kernel_size_e,
            variance_predictor_layer_num_e,
            variance_predictor_dropout,
            stop_gradient_flow_d,
            stop_gradient_flow_p,
            stop_gradient_flow_e,
            reduction_factor,
            encoder_fix,
            decoder_fix,
            pitch_AR,
            pitch_ARNAR,
            lstm_layers,
            speakers,
            emotions,
        )
        self.prosody_extractor = ProsodyExtractor(
            n_mel_channel,
            prosody_emb_dim,
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
            pp_conv_out_channels,
            conv_kernel_size=pp_conv_kernel_size,
            conv_n_layers=pp_conv_n_layers,
            conv_dropout=pp_conv_dropout,
            gru_layers=gru_n_layers,
            zoneout=pp_zoneout,
            num_gaussians=num_gaussians,
            global_prosody=global_prosody,
            global_gru_layers=global_gru_n_layers,
            global_d_gru=global_d_gru,
            global_num_gaussians=global_num_gaussians,
        )

        self.prosody_linear = nn.Linear(
            prosody_emb_dim,
            encoder_hidden_dim,
        )
        self.prosody_length_regulator = LengthRegulator()

        self.global_prosody = global_prosody

        if self.global_prosody is True:
            self.prosody_extractor.global_bi_gru.sort = True

    def init_forward_wGMM(
        self,
        s_mels,
        s_mel_lens,
        max_s_mel_len,
        t_mels,
        t_mel_lens,
        max_t_mel_len,
        s_snt_durations
    ):
        if s_snt_durations is None:
            s_snt_durations = torch.Tensor([s_mel.size(0) for s_mel in s_mels]).long()

        (
            s_mels,
            s_mel_lens,
            max_s_mel_len,
            s_mel_masks,
            t_mel_lens,
            max_t_mel_len,
            t_mel_masks,
        ) = self.init_forward(
            s_mels, s_mel_lens, max_s_mel_len, t_mel_lens, max_t_mel_len
        )
        if t_mels is not None:
            t_mels = t_mels[:, ::self.reduction_factor, :]

        return (
            s_mels,
            s_mel_lens,
            max_s_mel_len,
            s_mel_masks,
            t_mels,
            t_mel_lens,
            max_t_mel_len,
            t_mel_masks,
            s_snt_durations
        )

    def prosody_forward(
        self,
        output,
        t_mels,
        t_pitches,
        s_snt_durations,
        t_snt_durations,
    ):
        is_inference = True if t_pitches is None else False

        if self.global_prosody is False:
            prosody_target = self.prosody_extractor(t_mels, t_snt_durations)
            prosody_prediction, pi_outs, sigma_outs, mu_outs, snt_mask = self.prosody_predictor(
                output, s_snt_durations, target_prosody=prosody_target, is_inference=is_inference
            )
        else:
            prosody_target, g_prosody_target = self.prosody_extractor(t_mels, t_snt_durations)
            prosody_prediction, pi_outs, sigma_outs, mu_outs, snt_mask, g_pi, g_sigma, g_mu = self.prosody_predictor(
                output, s_snt_durations, target_prosody=prosody_target, target_global_prosody=g_prosody_target,
                is_inference=is_inference
            )
        if is_inference is True:
            prosody_prediction_expanded, _ = self.prosody_length_regulator(prosody_prediction, s_snt_durations)
            output = output + self.prosody_linear(prosody_prediction_expanded)
            free_tensors_memory([prosody_prediction, prosody_prediction_expanded, s_snt_durations])
        else:
            prosody_target_expanded, _ = self.prosody_length_regulator(prosody_target, s_snt_durations)
            output = output + self.prosody_linear(prosody_target_expanded)
            free_tensors_memory([prosody_prediction, prosody_target_expanded, s_snt_durations])

        if self.global_prosody is True:
            return (
                output,
                [prosody_target,
                 pi_outs,
                 mu_outs,
                 sigma_outs,
                 snt_mask,
                 g_prosody_target,
                 g_pi,
                 g_mu,
                 g_sigma]
            )
        return (
            output,
            [prosody_target,
             pi_outs,
             mu_outs,
             sigma_outs,
             snt_mask]
        )

    def forward(
        self,
        ids,
        s_sp_ids,
        t_sp_ids,
        s_em_ids,
        t_em_ids,
        s_mels,
        s_mel_lens,
        max_s_mel_len,
        s_pitches,
        s_energies,
        t_mels=None,
        t_mel_lens=None,
        max_t_mel_len=None,
        t_pitches=None,
        t_energies=None,
        t_durations=None,
        s_snt_durations=None,
        t_snt_durations=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        (
            s_mels,
            s_mel_lens,
            max_s_mel_len,
            s_mel_masks,
            t_mels,
            t_mel_lens,
            max_t_mel_len,
            t_mel_masks,
            s_snt_durations,
        ) = self.init_forward_wGMM(
            s_mels, s_mel_lens, max_s_mel_len, t_mels, t_mel_lens, max_t_mel_len, s_snt_durations
        )

        output = self.encoder_forward(
            s_sp_ids, s_em_ids, s_mels, s_mel_masks,
        )

        output, prosody_features = self.prosody_forward(
            output,
            t_mels,
            t_pitches,
            s_snt_durations,
            t_snt_durations,
        )
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            t_mel_lens,
            t_mel_masks,
        ) = self.variance_adaptor(
            output,
            s_mel_masks,
            max_s_mel_len,
            s_pitches,
            s_energies,
            t_durations,
            t_mel_masks,
            max_t_mel_len,
            t_pitches,
            t_energies,
            p_control,
            e_control,
            d_control,
            s_snt_durations,
            t_snt_durations,
        )

        (
            output,
            postnet_output,
            t_mel_lens,
            t_mel_masks,
        ) = self.decoder_forward(
            output, t_sp_ids, t_em_ids, t_mel_lens, t_mel_masks,
        )

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            s_mel_masks,
            t_mel_masks,
            s_mel_lens,
            t_mel_lens,
            prosody_features
        )

    def forward_woDuration(
        self,
        ids,
        s_sp_ids,
        t_sp_ids,
        s_em_ids,
        t_em_ids,
        s_mels,
        s_mel_lens,
        max_s_mel_len,
        s_pitches,
        s_energies,
        s_snt_durations,
        p_control=1.0,
        e_control=1.0,
        pitch_energy_prediction=False,
    ):
        # use source duration, pitch.
        (
            s_mels,
            s_mel_lens,
            max_s_mel_len,
            s_mel_masks,
            _,
            _,
            _,
            _,
            s_snt_durations,
        ) = self.init_forward_wGMM(
            s_mels, s_mel_lens, max_s_mel_len, None, None, None, s_snt_durations
        )

        output = self.encoder_forward(
            s_sp_ids, s_em_ids, s_mels, s_mel_masks,
        )
        if self.global_prosody is False:
            prosody_prediction, _, _, _, _ = self.prosody_predictor(
                output, s_snt_durations, target_prosody=None, is_inference=True
            )
        else:
            prosody_prediction, _, _, _, _, _, _, _ = self.prosody_predictor(
                output, s_snt_durations, target_prosody=None, target_global_prosody=None,
                is_inference=True
            )
        prosody_prediction_expanded, _ = self.prosody_length_regulator(prosody_prediction, s_snt_durations)
        output = output + self.prosody_linear(prosody_prediction_expanded)
        free_tensors_memory([prosody_prediction, prosody_prediction_expanded, s_snt_durations])

        output = self.variance_adaptor.forward_woDuration(
            output,
            s_mel_masks,
            max_s_mel_len,
            s_pitches,
            s_energies,
            p_control,
            e_control,
            pitch_energy_prediction,
        )
        (
            _,
            postnet_output,
            _,
            _,
        ) = self.decoder_forward(
            output, t_sp_ids, t_em_ids, s_mel_lens, s_mel_masks,
        )

        return postnet_output

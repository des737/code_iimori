from typing import Dict, Optional

import torch.nn as nn

from vc_tts_template.fastspeech2wGMM.fastspeech2wGMM import FastSpeech2wGMM
from vc_tts_template.fastspeech2wContexts.context_encoder import ConversationalProsodyContextEncoder
from vc_tts_template.fastspeech2wContexts.prosody_model import (
    ProsodyPredictorwAttention, PEProsodyEncoder, PEProsodyLocalEncoder
)


class Fastspeech2wGMMwContextswPEProsody(FastSpeech2wGMM):
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
        # context encoder
        context_encoder_hidden_dim: int,
        context_num_layer: int,
        context_encoder_dropout: float,
        text_emb_dim: int,
        peprosody_encoder_gru_dim: int,
        peprosody_encoder_gru_num_layer: int,
        shere_embedding: bool,
        current_attention: bool,
        mel_embedding_mode: int,
        mel_emb_dim: int,
        mel_emb_kernel: int,
        mel_emb_dropout: float,
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
        prosody_emb_size: int,
        attention_hidden_dim: int,
        attention_conv_channels: int,
        attention_conv_kernel_size: int,
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
        encoder_fix: bool,
        prosody_spk_independence: bool,
        local_prosody: bool,
        global_prosody: bool,
        prosody_attention: bool,
        stats: Optional[Dict],
        speakers: Dict,
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
            prosody_emb_dim,
            extra_conv_kernel_size,
            extra_conv_n_layers,
            extra_gru_n_layers,
            extra_global_gru_n_layers,
            gru_hidden_dim,
            gru_n_layers,
            pp_conv_out_channels,
            pp_conv_kernel_size,
            pp_conv_n_layers,
            pp_conv_dropout,
            pp_zoneout,
            num_gaussians,
            softmax_temperature,
            global_gru_n_layers,
            global_d_gru,
            global_num_gaussians,
            global_softmax_temperature,
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
            prosody_spk_independence,
            local_prosody,
            global_prosody,
            stats,
            speakers,
            emotions,
            accent_info,
        )
        # override to add padding_idx
        n_speaker = len(speakers)
        self.speaker_emb = nn.Embedding(
            n_speaker,
            encoder_hidden_dim,
            padding_idx=0,
        )
        self.emotion_emb = None
        if emotions is not None:
            n_emotion = len(emotions)
            self.emotion_emb = nn.Embedding(
                n_emotion,
                encoder_hidden_dim,
                padding_idx=0,
            )

        self.context_encoder = ConversationalProsodyContextEncoder(
            d_encoder_hidden=encoder_hidden_dim,
            d_context_hidden=context_encoder_hidden_dim,
            context_layer_num=context_num_layer,
            context_dropout=context_encoder_dropout,
            text_emb_size=text_emb_dim,
            g_prosody_emb_size=peprosody_encoder_gru_dim,
            speaker_embedding=self.speaker_emb,
            emotion_embedding=self.emotion_emb,
            current_attention=current_attention,
        )
        self.prosody_predictor = ProsodyPredictorwAttention(
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
            h_prosody_emb_size=encoder_hidden_dim*2 if mel_embedding_mode == 0 else mel_emb_dim,
            prosody_attention=prosody_attention,
            attention_hidden_dim=attention_hidden_dim,
            attention_conv_channels=attention_conv_channels,
            attention_conv_kernel_size=attention_conv_kernel_size,
            speaker_embedding=self.speaker_emb,
            emotion_embedding=self.emotion_emb,
        )
        if stats is not None:
            self.peprosody_encoder = PEProsodyEncoder(
                peprosody_encoder_gru_dim,
                peprosody_encoder_gru_num_layer,
                pitch_embedding=self.variance_adaptor.pitch_embedding,
                energy_embedding=self.variance_adaptor.energy_embedding,
                pitch_bins=self.variance_adaptor.pitch_bins,
                energy_bins=self.variance_adaptor.energy_bins,
                shere_embedding=shere_embedding
            )
            if prosody_attention is True:
                self.peprosody_local_encoder = PEProsodyLocalEncoder(
                    pitch_embedding=self.variance_adaptor.pitch_embedding,
                    energy_embedding=self.variance_adaptor.energy_embedding,
                    pitch_bins=self.variance_adaptor.pitch_bins,
                    energy_bins=self.variance_adaptor.energy_bins,
                    shere_embedding=shere_embedding
                )
        else:
            if mel_embedding_mode == 0:
                self.peprosody_encoder = PEProsodyEncoder(
                    peprosody_encoder_gru_dim,
                    peprosody_encoder_gru_num_layer,
                    pitch_embedding=self.variance_adaptor.pitch_embedding,
                    energy_embedding=self.variance_adaptor.energy_embedding,
                    shere_embedding=shere_embedding
                )
                if prosody_attention is True:
                    self.peprosody_local_encoder = PEProsodyLocalEncoder(
                        pitch_embedding=self.variance_adaptor.pitch_embedding,
                        energy_embedding=self.variance_adaptor.energy_embedding,
                        shere_embedding=shere_embedding
                    )
            else:
                self.peprosody_encoder = PEProsodyEncoder(
                    peprosody_encoder_gru_dim,
                    peprosody_encoder_gru_num_layer,
                    pitch_embedding=None,
                    energy_embedding=None,
                    shere_embedding=shere_embedding,
                    n_mel_channel=n_mel_channel,
                    mel_emb_dim=mel_emb_dim,
                    mel_emb_kernel=mel_emb_kernel,
                    mel_emb_dropout=mel_emb_dropout,
                )
                if prosody_attention is True:
                    self.peprosody_local_encoder = PEProsodyLocalEncoder(
                        pitch_embedding=None,
                        energy_embedding=None,
                        shere_embedding=shere_embedding,
                        n_mel_channel=n_mel_channel,
                        mel_emb_dim=mel_emb_dim,
                        mel_emb_kernel=mel_emb_kernel,
                        mel_emb_dropout=mel_emb_dropout,
                    )

    def contexts_forward(
        self,
        output,
        max_src_len,
        c_txt_embs,
        speakers,
        emotions,
        h_txt_embs,
        h_txt_emb_lens,
        h_speakers,
        h_emotions,
        h_prosody_embs,
        h_prosody_embs_lens,
        h_prosody_embs_len,
    ):
        h_prosody_emb = self.peprosody_encoder(
            h_prosody_embs,
            h_prosody_embs_lens,
        )

        context_enc = self.context_encoder(
            c_txt_embs,
            speakers,
            emotions,
            h_txt_embs,
            h_speakers,
            h_emotions,
            h_txt_emb_lens,
            h_prosody_emb,
            h_prosody_embs_len,
        )
        output = output + context_enc.unsqueeze(1).expand(
            -1, max_src_len, -1
        )
        return output

    def prosody_forward(
        self,
        output,
        src_lens,
        mels,
        p_targets,
        d_targets,
        h_local_prosody_emb,
        h_local_prosody_emb_lens,
        h_local_prosody_speakers,
        h_local_prosody_emotions,
        speakers,
    ):
        # h_local_prosody_emb: (B, time, 2)
        # h_local_prosody_emb_lens: (B)
        if h_local_prosody_emb is not None:
            h_local_prosody_emb = self.peprosody_local_encoder(h_local_prosody_emb)

        is_inference = True if p_targets is None else False

        if (self.prosody_spk_independence is True) and (self.speaker_emb is not None):
            output = output - self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, output.size(1), -1
            )

        if self.global_prosody is False:
            prosody_target = self.prosody_extractor(mels, d_targets)
            prosody_prediction, pi_outs, sigma_outs, mu_outs = self.prosody_predictor(
                output, h_local_prosody_emb, h_local_prosody_emb_lens,
                h_local_prosody_speakers, h_local_prosody_emotions,
                target_prosody=prosody_target, is_inference=is_inference
            )
        else:
            prosody_target, g_prosody_target = self.prosody_extractor(mels, d_targets, src_lens)
            prosody_prediction, pi_outs, sigma_outs, mu_outs, g_pi, g_sigma, g_mu = self.prosody_predictor(
                output, h_local_prosody_emb, h_local_prosody_emb_lens,
                h_local_prosody_speakers, h_local_prosody_emotions,
                target_prosody=prosody_target, target_global_prosody=g_prosody_target,
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
        c_txt_embs,
        h_txt_embs,
        h_txt_emb_lens,
        h_speakers,
        h_emotions,
        c_prosody_embs,
        c_prosody_embs_lens,
        h_prosody_embs,
        h_prosody_embs_lens,
        h_prosody_embs_len,
        h_local_prosody_emb,
        h_local_prosody_emb_lens,
        h_local_prosody_speakers,
        h_local_prosody_emotions,
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
        output = self.contexts_forward(
            output, max_src_len, c_txt_embs, speakers, emotions,
            h_txt_embs, h_txt_emb_lens, h_speakers, h_emotions,
            h_prosody_embs, h_prosody_embs_lens, h_prosody_embs_len,
        )

        output, prosody_features = self.prosody_forward(
            output, src_lens,
            mels, p_targets, d_targets,
            h_local_prosody_emb, h_local_prosody_emb_lens,
            h_local_prosody_speakers, h_local_prosody_emotions,
            speakers
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

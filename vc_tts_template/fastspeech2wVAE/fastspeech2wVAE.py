from typing import Dict, Optional, List

from vc_tts_template.fastspeech2 import FastSpeech2
from vc_tts_template.fastspeech2wVAE.vae import VAE_GST


class FastSpeech2wVAE(FastSpeech2):
    """ FastSpeech2 """

    def __init__(
        self,
        # init
        max_seq_len: int,
        num_vocab: int,
        # encoder
        encoder_hidden_dim: int,
        encoder_num_layer: int,
        encoder_num_head: int,
        conv_filter_size: int,
        conv_kernel_size_1: int,
        conv_kernel_size_2: int,
        encoder_dropout: float,
        # ref encoder
        ref_enc_dim: int,
        ref_enc_filters: List[int],
        ref_enc_kernel_size: int,
        ref_enc_stride: int,
        ref_enc_pad: int,
        ref_enc_gru_size: int,
        z_latent_dim: int,
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
        # other
        n_mel_channel: int,
        encoder_fix: bool,
        stats: Dict,
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
        self.vae = VAE_GST(
            encoder_hidden_dim,
            ref_enc_dim,
            ref_enc_filters,
            ref_enc_kernel_size,
            ref_enc_stride,
            ref_enc_pad,
            n_mel_channel,
            ref_enc_gru_size,
            z_latent_dim,
        )

        self.encoder_fix = encoder_fix
        self.speakers = speakers
        self.emotions = emotions
        self.accent_info = accent_info

    def vae_forward(
        self,
        encoder_output,
        target_mel,
    ):
        prosody_outputs, mu, logvar = self.vae(
            target_mel, batch_size=encoder_output.size(0), device=encoder_output.device
        )
        prosody_outputs = prosody_outputs.unsqueeze(1).expand_as(encoder_output)
        encoder_output = encoder_output + prosody_outputs

        return encoder_output, mu, logvar

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

        output, mu, logvar = self.vae_forward(output, mels)

        output, p_predictions, e_predictions, log_d_predictions, d_rounded, mel_lens, mel_masks = self.variance_adaptor(
            output, src_masks, mel_masks, max_mel_len, p_targets, e_targets, d_targets, p_control, e_control, d_control,
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
            mu,
            logvar,
        )

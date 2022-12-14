from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np

from vc_tts_template.fastspeech2.fastspeech2 import FastSpeech2
from vc_tts_template.fastspeech2wContexts.prosody_model import PEProsodyEncoder
from vc_tts_template.fastspeech2wGMM.prosody_model import mel2phone, phone2utter
from vc_tts_template.fastspeech2.varianceadaptor import LengthRegulator


class FastSpeech2wPEProsodywoPEPCE(FastSpeech2):
    """ FastSpeech2wContexts """

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
        peprosody_encoder_gru_dim: int,
        peprosody_encoder_gru_num_layer: int,
        shere_embedding: bool,
        mel_embedding_mode: int,
        pau_split_mode: int,
        # mel_emb_dim: int,
        # mel_emb_kernel: int,
        # mel_emb_dropout: float,
        peprosody_encoder_conv_kernel_size: int,
        peprosody_encoder_conv_n_layers: int,
        sslprosody_emb_dim: Optional[int],
        sslprosody_layer_num: Optional[int],
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

        self.context_encoder = nn.Linear(  # type: ignore
            peprosody_encoder_gru_dim if sslprosody_emb_dim is None else sslprosody_emb_dim,
            encoder_hidden_dim
        )
        if sslprosody_emb_dim is None:
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
            else:
                if mel_embedding_mode == 0:
                    self.peprosody_encoder = PEProsodyEncoder(
                        peprosody_encoder_gru_dim,
                        peprosody_encoder_gru_num_layer,
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
                        conv_kernel_size=peprosody_encoder_conv_kernel_size,
                        conv_n_layers=peprosody_encoder_conv_n_layers,
                    )
            self.use_ssl = False
        else:
            if sslprosody_layer_num > 1:  # type:ignore
                self.peprosody_encoder = nn.Conv1d(  # type: ignore
                    in_channels=sslprosody_layer_num,  # type: ignore
                    out_channels=1,
                    kernel_size=1,
                    bias=False,
                )
            else:
                self.peprosody_encoder = None  # type:ignore
            self.use_ssl = True
        self.length_regulator = LengthRegulator()
        self.pau_split_mode = pau_split_mode > 0
        self.sslprosody_layer_num = sslprosody_layer_num

    def contexts_forward(
        self,
        output,
        max_src_len,
        c_prosody_embs,
        c_prosody_embs_lens,
        c_prosody_embs_duration,
        c_prosody_embs_phonemes,
    ):
        if self.pau_split_mode is False:
            # c_prosody_embs: (B, layer_num, dim)
            if self.use_ssl is False:
                h_prosody_emb = self.peprosody_encoder(
                    c_prosody_embs.unsqueeze(1),
                    c_prosody_embs_lens.unsqueeze(1),
                )
            else:
                if self.peprosody_encoder is not None:
                    h_prosody_emb = self.peprosody_encoder(c_prosody_embs)
                else:
                    h_prosody_emb = c_prosody_embs.squeeze(1)

            context_enc = self.context_encoder(h_prosody_emb).squeeze(1)

            output = output + context_enc.unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        else:
            if self.use_ssl is False:
                (
                    output_sorted, src_lens_sorted, segment_nums, inv_sort_idx
                ) = mel2phone(c_prosody_embs, c_prosody_embs_duration.cpu().numpy())
                outs = list()
                for _output, src_lens in zip(output_sorted, src_lens_sorted):
                    out = self.peprosody_encoder(
                        _output.unsqueeze(1),
                        np.array(src_lens),
                    )
                    outs.append(out.squeeze(1))
                outs = torch.cat(outs, 0)
                h_prosody_emb = phone2utter(outs[inv_sort_idx], segment_nums)
            else:
                # c_prosody_embs: (B, max_seg_len*self.sslprosody_layer_num, dim)
                if self.peprosody_encoder is not None:
                    max_seg_len = int(torch.max(c_prosody_embs_lens) / self.sslprosody_layer_num)
                    c_prosody_embs = c_prosody_embs.view(
                        c_prosody_embs.size(0), max_seg_len, self.sslprosody_layer_num, -1
                    )
                    c_prosody_embs = c_prosody_embs.transpose(1, 2).contiguous().view(
                        c_prosody_embs.size(0), self.sslprosody_layer_num, -1
                    )
                    h_prosody_emb = self.peprosody_encoder(
                        c_prosody_embs
                    ).squeeze(1).view(c_prosody_embs.size(0), max_seg_len, -1)
                else:
                    h_prosody_emb = c_prosody_embs
            context_enc = self.context_encoder(h_prosody_emb)
            context_enc, _ = self.length_regulator(
                context_enc, c_prosody_embs_phonemes, torch.max(c_prosody_embs_phonemes)
            )
            output = output + context_enc

        return output

    def forward(
        self,
        ids,
        speakers,
        emotions,
        texts,
        src_lens,
        max_src_len,
        c_txt_embs,
        c_txt_embs_lens,
        h_txt_embs,
        h_txt_emb_lens,
        h_speakers,
        h_emotions,
        c_prosody_embs,
        c_prosody_embs_lens,
        c_prosody_embs_duration,
        c_prosody_embs_phonemes,
        h_prosody_embs,
        h_prosody_embs_lens,
        h_prosody_embs_len,
        h_local_prosody_emb=None,
        h_local_prosody_emb_lens=None,
        h_local_prosody_speakers=None,
        h_local_prosody_emotions=None,
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
            output, max_src_len, c_prosody_embs, c_prosody_embs_lens,
            c_prosody_embs_duration, c_prosody_embs_phonemes
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
        )

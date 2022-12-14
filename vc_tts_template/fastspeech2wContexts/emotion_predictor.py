from typing import Dict

import torch
import torch.nn as nn

from vc_tts_template.fastspeech2wContexts.context_encoder import (
    ConversationalProsodyContextEncoder, ConversationalContextEncoder, ConversationalProsodyEncoder
)
from vc_tts_template.fastspeech2wContexts.prosody_model import PEProsodyEncoder


class EmotionPredictor(nn.Module):
    """ Emotion Predictor

    context encoderやprosody context encoderの正当性を確かめるために,
    prosodyからemotion labelを予測するモデルです.
    簡単のために, `FastSpeech2wContextswPEProsody`と同じ引数と入力を取ることとします.
    """

    def __init__(
        self,
        encoder_hidden_dim: int,
        # context encoder
        context_encoder_hidden_dim: int,
        context_num_layer: int,
        context_encoder_dropout: float,
        text_emb_dim: int,
        peprosody_encoder_gru_dim: int,
        peprosody_encoder_gru_num_layer: int,
        gru_bidirectional: bool,
        # variance predictor
        pitch_embed_kernel_size: int,
        pitch_embed_dropout: float,
        energy_embed_kernel_size: int,
        energy_embed_dropout: float,
        # other
        speakers: Dict,
        emotions: Dict,
        linear_hidden_dim: int,
        use_context_encoder: bool,
        use_prosody_encoder: bool,
        use_peprosody_encoder: bool,
        use_melprosody_encoder: bool,
        mel_emb_dim: int,
        mel_emb_kernel: int,
        mel_emb_dropout: float,
        current_attention: bool,
        # garbege
        pitch_feature_level: int,
        energy_feature_level: int,
        n_mel_channel: int,
        accent_info: int,
        mel_embedding_mode: int,
    ):
        super().__init__()
        n_speaker = len(speakers)
        self.speaker_emb = nn.Embedding(
            n_speaker,
            encoder_hidden_dim,
            padding_idx=0,
        )
        self.emotion_emb = None  # emotion label予測タスクなので当然使わない

        if use_prosody_encoder is True:
            # 外部で用意したglobal prosody embeddingを使う方式
            raise RuntimeError("未対応です")
        if (use_context_encoder is True) and ((use_peprosody_encoder or use_melprosody_encoder) is True):
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
                gru_bidirectional=gru_bidirectional,
            )
        elif (use_context_encoder is True) and ((use_peprosody_encoder or use_melprosody_encoder) is False):
            self.context_encoder = ConversationalContextEncoder(  # type:ignore
                d_encoder_hidden=encoder_hidden_dim,
                d_context_hidden=context_encoder_hidden_dim,
                context_layer_num=context_num_layer,
                context_dropout=context_encoder_dropout,
                text_emb_size=text_emb_dim,
                speaker_embedding=self.speaker_emb,
                emotion_embedding=self.emotion_emb,
                current_attention=current_attention,
                gru_bidirectional=gru_bidirectional,
            )
        elif (use_context_encoder is False) and ((use_peprosody_encoder or use_melprosody_encoder) is True):
            self.context_encoder = ConversationalProsodyEncoder(  # type:ignore
                d_encoder_hidden=encoder_hidden_dim,
                d_context_hidden=context_encoder_hidden_dim,
                context_layer_num=context_num_layer,
                context_dropout=context_encoder_dropout,
                g_prosody_emb_size=peprosody_encoder_gru_dim,
                speaker_embedding=self.speaker_emb,
                emotion_embedding=self.emotion_emb,
            )
        else:
            raise RuntimeError("未対応です. CEかPEProsodyのどちらかは利用しましょう.")

        if use_peprosody_encoder is True:
            _pitch_embedding = nn.Sequential(  # type:ignore
                nn.Conv1d(
                    in_channels=1,
                    out_channels=encoder_hidden_dim,
                    kernel_size=pitch_embed_kernel_size,
                    padding=(pitch_embed_kernel_size - 1) // 2,
                ),
                nn.Dropout(pitch_embed_dropout),
            )
            _energy_embedding = nn.Sequential(  # type:ignore
                nn.Conv1d(
                    in_channels=1,
                    out_channels=encoder_hidden_dim,
                    kernel_size=energy_embed_kernel_size,
                    padding=(energy_embed_kernel_size - 1) // 2,
                ),
                nn.Dropout(energy_embed_dropout),
            )
            self.peprosody_encoder = PEProsodyEncoder(
                peprosody_encoder_gru_dim,
                peprosody_encoder_gru_num_layer,
                pitch_embedding=_pitch_embedding,
                energy_embedding=_energy_embedding,
                shere_embedding=False,  # 当然Trueは使えない
            )
        elif use_melprosody_encoder is True:
            self.peprosody_encoder = PEProsodyEncoder(
                peprosody_encoder_gru_dim,
                peprosody_encoder_gru_num_layer,
                pitch_embedding=None,
                energy_embedding=None,
                shere_embedding=False,  # 当然Trueは使えない
                n_mel_channel=n_mel_channel,
                mel_emb_dim=mel_emb_dim,
                mel_emb_kernel=mel_emb_kernel,
                mel_emb_dropout=mel_emb_dropout,
            )
        else:
            self.peprosody_encoder = None  # type:ignore

        self.output_linear = nn.Sequential(
            nn.Linear(encoder_hidden_dim, linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, len(emotions)),
        )

        self.use_context_encoder = use_context_encoder
        self.use_peprosody_encoder = use_peprosody_encoder
        self.use_melprosody_encoder = use_melprosody_encoder

    def contexts_forward(
        self,
        c_txt_embs,
        speakers,
        emotions,
        h_txt_embs,
        h_txt_emb_lens,
        h_speakers,
        h_emotions,
        h_prosody_embs,
        h_prosody_embs_lens,
    ):
        if (self.use_peprosody_encoder or self.use_melprosody_encoder) is True:
            h_prosody_emb = self.peprosody_encoder(
                h_prosody_embs,
                h_prosody_embs_lens,
            )

        if (self.use_context_encoder is True) and \
                ((self.use_peprosody_encoder or self.use_melprosody_encoder) is True):
            context_enc = self.context_encoder(
                c_txt_embs,
                speakers,
                emotions,
                h_txt_embs,
                h_speakers,
                h_emotions,
                h_txt_emb_lens,
                h_prosody_emb,
            )
        elif (self.use_context_encoder is True) and \
                ((self.use_peprosody_encoder or self.use_melprosody_encoder) is False):
            context_enc = self.context_encoder(
                c_txt_embs,
                speakers,
                emotions,
                h_txt_embs,
                h_speakers,
                h_emotions,
                h_txt_emb_lens,
            )
        elif (self.use_context_encoder is False) and \
                ((self.use_peprosody_encoder or self.use_melprosody_encoder) is True):
            context_enc = self.context_encoder(
                h_speakers,
                h_emotions,
                h_txt_emb_lens,
                h_prosody_emb,
            )

        return context_enc

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
        h_prosody_embs,
        h_prosody_embs_lens,
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
        output = self.contexts_forward(
            c_txt_embs, speakers, emotions,
            h_txt_embs, h_txt_emb_lens, h_speakers, h_emotions,
            h_prosody_embs, h_prosody_embs_lens,
        )
        logits = self.output_linear(output)

        return logits


class EmotionPredictorLoss(nn.Module):
    """ EmotionPredictor Loss """

    def __init__(
        self, pitch_feature_level, energy_feature_level,
        emotion_label_dict, sum_emotions
    ):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss()
        self.emotion2id = emotion_label_dict
        self.id2emotion = {v: k for k, v in self.emotion2id.items()}
        self.sum_emotions = sum_emotions

    def forward(self, inputs, predictions):
        emotions = inputs[2]
        loss = self.loss_fn(predictions, emotions)
        correct_num = torch.sum(torch.argmax(predictions, dim=-1) == emotions)
        acc = correct_num.item() / len(emotions)

        loss_values = {}

        # emotionの登場回数を見る
        preds = torch.argmax(predictions, dim=-1).cpu().numpy()
        tgts = emotions.cpu().numpy()

        # init
        for emo in self.emotion2id.keys():
            loss_values[
                "total-num_" + emo
            ] = 0
            loss_values[
                "total-collect-num_" + emo
            ] = 0
        if len(self.sum_emotions) > 0:
            loss_values[
                "total-num_" + "+".join(self.sum_emotions)
            ] = 0
            loss_values[
                "total-collect-num_" + "+".join(self.sum_emotions)
            ] = 0

        for pre, tgt in zip(preds, tgts):
            loss_values["total-num_" + self.id2emotion[tgt]] += 1
            if self.id2emotion[tgt] in self.sum_emotions:
                loss_values["total-num_" + "+".join(self.sum_emotions)] += 1

            if pre == tgt:
                loss_values["total-collect-num_" + self.id2emotion[tgt]] += 1
                if self.id2emotion[tgt] in self.sum_emotions:
                    loss_values["total-collect-num_" + "+".join(self.sum_emotions)] += 1

        loss_values.update({
            "accuracy": acc,
            "class_loss": loss.item(),
            "total_loss": loss.item(),
        })

        return loss, loss_values

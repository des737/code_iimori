# https://github.dev/keonlee9420/Expressive-FastSpeech2/tree/conversational
import sys

import torch
import torch.nn as nn
import numpy as np

sys.path.append("../..")
from vc_tts_template.fastspeech2wGMM.layers import GRUwSort
from vc_tts_template.utils import make_pad_mask


class ConversationalContextEncoder(nn.Module):
    """ Conversational Context Encoder """

    def __init__(
        self,
        d_encoder_hidden,
        d_context_hidden,
        context_layer_num,
        context_dropout,
        text_emb_size,
        prosody_emb_size,
        speaker_embedding,
        emotion_embedding,
        use_text_modal=True,
        use_speech_modal=True,
        current_attention=True,  # attentionを用いるか
        past_global_gru=False,  # 従来法のgruを用いるか
        gru_bidirectional=True,
        pau_split_mode=False,
        last_concat=False,
    ):
        super(ConversationalContextEncoder, self).__init__()
        d_model = d_encoder_hidden
        d_cont_enc = d_context_hidden
        num_layers = context_layer_num
        dropout = context_dropout
        text_emb_size = text_emb_size
        prosody_emb_size = prosody_emb_size

        if (use_text_modal is False) and (use_speech_modal is False):
            raise RuntimeError("textかspeechのどちらかのmodalはTrueにしてください.")

        self.text_emb_linear = nn.Linear(text_emb_size, d_cont_enc)
        self.text_enc_linear = nn.Sequential(
            nn.Linear(
                2*d_cont_enc if emotion_embedding is None else 3*d_cont_enc,
                d_cont_enc
            ),
            nn.ReLU()
        )
        if use_text_modal is True:
            self.text_gru = GRUwSort(
                input_size=d_cont_enc,
                hidden_size=d_cont_enc if gru_bidirectional is True else d_cont_enc * 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=gru_bidirectional,
                sort=True,
                dropout=dropout,
                allow_zero_length=True,
                need_last=True if past_global_gru is True else False,
            )
            self.text_gru_linear = nn.Sequential(
                nn.Linear(2*d_cont_enc, d_cont_enc),
                nn.ReLU()
            )
            if (current_attention is True) and (past_global_gru is False):
                # 基本Trueのほうが性能がいいです.
                self.context_text_value_linear = nn.Linear(d_cont_enc, d_model)
                self.context_text_query_linear = nn.Linear(d_cont_enc, d_model)
                self.context_text_attention = SLA_wQuery(pau_split_mode)
                self.context_text_linear = nn.Linear(d_cont_enc + d_model, d_model)
            elif (current_attention is False) and (past_global_gru is True):
                self.context_text_linear = nn.Linear(d_cont_enc * 2, d_model)
                self.context_text_attention = None
            elif (current_attention is False) and (past_global_gru is False):
                self.context_text_linear = nn.Linear(d_cont_enc, d_model)
                self.context_text_attention = SLA(d_model)
            else:
                raise RuntimeError("未対応です")
        if use_speech_modal is True:
            self.prosody_emb_linear = nn.Linear(prosody_emb_size, d_cont_enc)
            self.prosody_enc_linear = nn.Sequential(
                nn.Linear(
                    2*d_cont_enc if emotion_embedding is None else 3*d_cont_enc,
                    d_cont_enc
                ),
                nn.ReLU()
            )
            self.prosody_gru = GRUwSort(
                input_size=d_cont_enc,
                hidden_size=d_cont_enc,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                sort=True,
                dropout=dropout,
                allow_zero_length=True,
                need_last=True if past_global_gru is True else False,
            )
            self.prosody_gru_linear = nn.Sequential(
                nn.Linear(2*d_cont_enc, d_cont_enc),
                nn.ReLU()
            )
            if (current_attention is True) and (past_global_gru is False):
                # 基本Trueのほうが性能がいいです.
                self.context_prosody_value_linear = nn.Linear(d_cont_enc, d_model)
                self.context_prosody_query_linear = nn.Linear(d_cont_enc, d_model)
                self.context_prosody_attention = SLA_wQuery(pau_split_mode)
                self.context_prosody_linear = nn.Linear(d_cont_enc + d_model, d_model)
            elif (current_attention is False) and (past_global_gru is True):
                self.context_prosody_linear = nn.Linear(d_cont_enc * 2, d_model)
                self.context_prosody_attention = None
            elif (current_attention is False) and (past_global_gru is False):
                self.context_prosody_linear = nn.Linear(d_cont_enc, d_model)
                self.context_prosody_attention = SLA(d_model)
            else:
                raise RuntimeError("未対応です")

        self.speaker_linear = nn.Linear(d_model, d_cont_enc)
        if emotion_embedding is not None:
            self.emotion_linear = nn.Linear(d_model, d_cont_enc)

        if (last_concat is True) and (use_text_modal is True) and (use_speech_modal is True):
            self.last_concat_linear = nn.Linear(d_model*2, d_model)

        self.speaker_embedding = speaker_embedding
        self.emotion_embedding = emotion_embedding
        self.current_attention = current_attention
        self.past_global_gru = past_global_gru
        self.pau_split_mode = pau_split_mode

        self.use_text_modal = use_text_modal
        self.use_speech_modal = use_speech_modal
        self.last_concat = last_concat

    def forward(
        self, c_txt_embs, c_txt_embs_lens, speakers, emotions,
        h_txt_embs, h_txt_emb_lens, h_speakers, h_emotions,
        h_prosody_emb, h_prosody_embs_len,
    ):
        if self.pau_split_mode is False:
            if self.use_text_modal is True:
                max_t_history_len = torch.max(h_txt_emb_lens)
                t_history_masks = make_pad_mask(h_txt_emb_lens, max_t_history_len)

            if self.use_speech_modal is True:
                max_p_history_len = torch.max(h_prosody_embs_len)
                p_history_masks = make_pad_mask(h_prosody_embs_len, max_p_history_len)

            current_speaker = speakers.unsqueeze(1)
            current_text_emb = self.text_emb_linear(c_txt_embs.unsqueeze(1))
        else:
            max_segment_len = c_txt_embs.size(1)

            if self.use_text_modal is True:
                max_t_history_len = torch.max(h_txt_emb_lens+c_txt_embs_lens)
                t_history_masks = make_pad_mask(h_txt_emb_lens+c_txt_embs_lens, max_t_history_len)

            if self.use_speech_modal is True:
                max_p_history_len = torch.max(h_prosody_embs_len+c_txt_embs_lens)
                p_history_masks = make_pad_mask(h_prosody_embs_len+c_txt_embs_lens, max_p_history_len)

            current_speaker = speakers.unsqueeze(1).expand(-1, max_segment_len)
            current_text_emb = self.text_emb_linear(c_txt_embs)

        history_speaker = self.speaker_linear(self.speaker_embedding(h_speakers))

        if self.use_text_modal is True:
            history_text_emb = self.text_emb_linear(h_txt_embs)
            history_text_enc = torch.cat([history_text_emb, history_speaker[:, :history_text_emb.size(1)]], dim=-1)

        if self.use_speech_modal is True:
            history_prosody_emb = self.prosody_emb_linear(h_prosody_emb)
            history_prosody_enc = torch.cat([
                history_prosody_emb, history_speaker[:, :history_prosody_emb.size(1)]
            ], dim=-1)

        current_speaker = self.speaker_linear(self.speaker_embedding(current_speaker))
        current_text_enc = torch.cat([current_text_emb, current_speaker], dim=-1)

        if self.emotion_embedding is not None:
            if self.pau_split_mode is False:
                current_emotion = emotions.unsqueeze(1)
            else:
                current_emotion = emotions.unsqueeze(1).expand(-1, max_segment_len)
            history_emotion = self.emotion_linear(self.emotion_embedding(h_emotions))
            current_emotion = self.emotion_linear(self.emotion_embedding(current_emotion))

            if self.use_text_modal is True:
                history_text_enc = torch.cat([history_text_enc, history_emotion[:, :history_text_enc.size(1)]], dim=-1)
            if self.use_speech_modal is True:
                history_prosody_enc = torch.cat([
                    history_prosody_enc, history_emotion[:, :history_prosody_enc.size(1)]
                ], dim=-1)
            current_text_enc = torch.cat([current_text_enc, current_emotion], dim=-1)

        if self.use_text_modal is True:
            history_text_enc = self.text_enc_linear(history_text_enc)
        if self.use_speech_modal is True:
            history_prosody_enc = self.prosody_enc_linear(history_prosody_enc)

        current_text_enc = self.text_enc_linear(current_text_enc)

        if self.pau_split_mode is False:
            if self.use_text_modal is True:
                # GRU
                history_text_enc = self.text_gru_linear(self.text_gru(history_text_enc, h_txt_emb_lens))
            if self.use_speech_modal is True:
                # GRU
                history_prosody_enc = self.prosody_gru_linear(self.prosody_gru(history_prosody_enc, h_prosody_embs_len))

            # Encoding
            if self.current_attention is True:
                # Attention
                if self.use_text_modal is True:
                    context_text_enc, text_score = self.context_text_attention(
                        self.context_text_query_linear(current_text_enc),
                        self.context_text_value_linear(history_text_enc),
                        mask=t_history_masks,
                    )
                    context_text_enc = self.context_text_linear(
                        torch.cat([current_text_enc.squeeze(1), context_text_enc], dim=-1)
                    )
                if self.use_speech_modal is True:
                    context_prosody_enc, prosody_score = self.context_prosody_attention(
                        self.context_prosody_query_linear(current_text_enc),
                        self.context_prosody_value_linear(history_prosody_enc),
                        mask=p_history_masks,
                    )
                    context_prosody_enc = self.context_prosody_linear(
                        torch.cat([current_text_enc.squeeze(1), context_prosody_enc], dim=-1)
                    )
            else:
                if self.past_global_gru is True:
                    # 従来法
                    if self.use_text_modal is True:
                        context_text_enc = torch.cat([current_text_enc.squeeze(1), history_text_enc], dim=-1)
                        context_text_enc = self.context_text_linear(context_text_enc)
                    if self.use_speech_modal is True:
                        context_prosody_enc = torch.cat([current_text_enc.squeeze(1), history_prosody_enc], dim=-1)
                        context_prosody_enc = self.context_prosody_linear(context_prosody_enc)
                else:
                    # SLA実装
                    if self.use_text_modal is True:
                        history_text_enc = history_text_enc.masked_fill(t_history_masks.unsqueeze(-1), 0)
                        context_text_enc = torch.cat([current_text_enc, history_text_enc], dim=1)
                        context_text_enc, text_score = self.context_text_attention(
                            self.context_text_linear(context_text_enc)
                        )
                    if self.use_speech_modal is True:
                        history_prosody_enc = history_prosody_enc.masked_fill(p_history_masks.unsqueeze(-1), 0)
                        context_prosody_enc = torch.cat([current_text_enc, history_prosody_enc], dim=1)
                        context_prosody_enc, prosody_score = self.context_prosody_attention(
                            self.context_prosody_linear(context_prosody_enc)
                        )
        else:
            if self.use_text_modal is True:
                context_text_enc = self.text_gru_linear(
                    self.text_gru(history_text_enc, h_txt_emb_lens, current_text_enc, c_txt_embs_lens)
                )
            if self.use_speech_modal is True:
                context_prosody_enc = self.prosody_gru_linear(
                    self.prosody_gru(history_prosody_enc, h_prosody_embs_len, current_text_enc, c_txt_embs_lens)
                )

            if self.current_attention is True:
                # Attention
                if self.use_text_modal is True:
                    context_text_enc, text_score = self.context_text_attention(
                        self.context_text_query_linear(current_text_enc),
                        self.context_text_value_linear(context_text_enc),
                        mask=t_history_masks,
                    )
                    context_text_enc = self.context_text_linear(
                        torch.cat([current_text_enc, context_text_enc], dim=-1)
                    )
                if self.use_speech_modal is True:
                    context_prosody_enc, prosody_score = self.context_prosody_attention(
                        self.context_prosody_query_linear(current_text_enc),
                        self.context_prosody_value_linear(context_prosody_enc),
                        mask=p_history_masks,
                    )
                    context_prosody_enc = self.context_prosody_linear(
                        torch.cat([current_text_enc, context_prosody_enc], dim=-1)
                    )
            else:
                if self.past_global_gru is True:
                    # 従来法
                    if self.use_text_modal is True:
                        context_text_enc = torch.cat([
                            current_text_enc, context_text_enc
                        ], dim=-1)
                        context_text_enc = self.context_text_linear(context_text_enc)
                    if self.use_speech_modal is True:
                        context_prosody_enc = torch.cat([
                            current_text_enc, context_prosody_enc
                        ], dim=-1)
                        context_prosody_enc = self.context_prosody_linear(context_prosody_enc)
                else:
                    # SLAのやつ
                    assert RuntimeError("未対応")

        if self.current_attention is False:
            if (self.use_text_modal is True) and (self.use_speech_modal is True):
                if self.last_concat is True:
                    return self.last_concat_linear(
                        torch.cat([
                            context_text_enc, context_prosody_enc
                        ], dim=-1)
                    )
                else:
                    return context_text_enc + context_prosody_enc
            elif (self.use_text_modal is True) and (self.use_speech_modal is False):
                return context_text_enc
            elif (self.use_text_modal is False) and (self.use_speech_modal is True):
                return context_prosody_enc
        else:
            if (self.use_text_modal is True) and (self.use_speech_modal is True):
                if self.last_concat is True:
                    return self.last_concat_linear(
                        torch.cat([
                            context_text_enc, context_prosody_enc
                        ], dim=-1)
                    ), text_score, prosody_score
                else:
                    return context_text_enc + context_prosody_enc, text_score, prosody_score
            elif (self.use_text_modal is True) and (self.use_speech_modal is False):
                return context_text_enc, text_score, None
            elif (self.use_text_modal is False) and (self.use_speech_modal is True):
                return context_prosody_enc, None, prosody_score


class SLA(nn.Module):
    """ Sequence Level Attention """

    def __init__(self, d_enc):
        super(SLA, self).__init__()
        self.linear = nn.Linear(d_enc, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoding, mask=None):
        # encoding: (B, time, d_enc)
        # mask: (B, time)
        with torch.cuda.amp.autocast(enabled=False):
            encoding = encoding.to(torch.float32)
            attn = self.linear(encoding)

        # attn: (B, time, 1)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1), -np.inf)
            aux_mask = (attn == -np.inf).all(self.softmax.dim).unsqueeze(self.softmax.dim)
            attn = attn.masked_fill(aux_mask, 0)  # Remove all -inf along softmax.dim
        score = self.softmax(attn).transpose(-2, -1)  # [B, 1, T]
        fused_rep = torch.matmul(score, encoding).squeeze(1)  # [B, d]
        return fused_rep, score


class SLA_wQuery(nn.Module):
    """ Sequence Level Attention
    Queryを取れるようにして, そのqueryでattentionをとる.
    """

    def __init__(self, leave_dim_1=False):
        super(SLA_wQuery, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.leave_dim_1 = leave_dim_1

    def forward(self, query, encoding, mask=None):
        # query: (B, x, d_enc)
        # encoding: (B, time, d_enc)
        # mask: (B, time)
        with torch.cuda.amp.autocast(enabled=False):
            encoding = encoding.to(torch.float32)
            query = query.to(torch.float32)
            attn = torch.bmm(encoding, query.transpose(1, 2))

        # attn: (B, time, x)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1), -np.inf)
            aux_mask = (attn == -np.inf).all(self.softmax.dim).unsqueeze(self.softmax.dim)
            attn = attn.masked_fill(aux_mask, 0)  # Remove all -inf along softmax.dim
        # score: (B, x, time)
        score = self.softmax(attn).transpose(-2, -1)
        fused_rep = torch.matmul(score, encoding)  # [B, x, d]

        if self.leave_dim_1 is True:
            return fused_rep, score
        else:
            return fused_rep.squeeze(1), score

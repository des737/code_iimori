import shutil
import sys
from pathlib import Path
import os
from typing import Dict, List

import hydra
import joblib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from scipy.io import wavfile
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, Wav2Vec2Model
import librosa

sys.path.append("../..")
from recipes.common.fit_scaler import MultiSpeakerStandardScaler  # noqa: F401
from vc_tts_template.fastspeech2wContexts.collate_fn import (
    get_embs, make_dialogue_dict)
from vc_tts_template.fastspeech2wContexts.collate_fn_PEProsody import \
    get_peprosody_embs
from vc_tts_template.fastspeech2wContexts.gen import synthesis
from vc_tts_template.fastspeech2wContexts.gen_PEProsody import \
    synthesis_PEProsody
from vc_tts_template.utils import load_utt_list, optional_tqdm


@hydra.main(config_path="conf/synthesis", config_name="config")
def my_app(config: DictConfig) -> None:
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    # acoustic model
    acoustic_config = OmegaConf.load(to_absolute_path(config.acoustic.model_yaml))
    acoustic_model = hydra.utils.instantiate(acoustic_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.acoustic.checkpoint),
        map_location=device,
    )
    acoustic_model.load_state_dict(checkpoint["state_dict"])
    acoustic_out_scaler = joblib.load(to_absolute_path(config.acoustic.out_scaler_path))
    acoustic_model.eval()

    # vocoder
    vocoder_config = OmegaConf.load(to_absolute_path(config.vocoder.model_yaml))
    vocoder_model = hydra.utils.instantiate(vocoder_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.vocoder.checkpoint),
        map_location=device,
    )
    vocoder_model.load_state_dict(checkpoint["state_dict"]["netG"])
    vocoder_model.eval()
    vocoder_model.remove_weight_norm()

    in_dir = Path(to_absolute_path(config.in_dir))
    out_dir = Path(to_absolute_path(config.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # data prepare
    utt_ids = load_utt_list(to_absolute_path(config.utt_list))
    if config.reverse:
        utt_ids = utt_ids[::-1]

    lab_files = [in_dir / f"{utt_id.strip()}-feats.npy" for utt_id in utt_ids]

    # read text embeddings
    dialogue_info = to_absolute_path(config.dialogue_info)
    utt2id, id2utt = make_dialogue_dict(dialogue_info)
    text_emb_paths = list(Path(to_absolute_path(config.emb_dir)).glob("*.npy"))
    emb_seg_dir = Path(to_absolute_path(config.emb_dir)) / "segmented_text_emb"
    text_seg_emb_paths = list(emb_seg_dir.glob("*.npy")) if emb_seg_dir.exists() else None
    use_hist_num = config.use_hist_num
    use_prosody_hist_num = config.use_prosody_hist_num if type(config.use_prosody_hist_num) is int else use_hist_num
    # read prosody embeddings
    prosody_emb_paths = list(Path(to_absolute_path(config.prosody_emb_dir)).glob("*.npy"))
    g_prosody_emb_paths = list(Path(to_absolute_path(config.g_prosody_emb_dir)).glob("*.npy"))
    prosody_seg_emb_dir = Path(to_absolute_path(config.prosody_emb_dir)) / "segmented_prosody_emb"
    prosody_emb_seg_d_dir = Path(to_absolute_path(config.prosody_emb_dir)) / "segment_duration"
    prosody_emb_seg_p_dir = Path(to_absolute_path(config.prosody_emb_dir)) / "segment_phonemes"
    prosody_seg_emb_paths = list(prosody_seg_emb_dir.glob("*.npy")) if prosody_seg_emb_dir.exists() else None
    prosody_seg_d_emb_paths = list(prosody_emb_seg_d_dir.glob("*.npy")) if prosody_emb_seg_d_dir.exists() else None
    prosody_seg_p_emb_paths = list(prosody_emb_seg_p_dir.glob("*.npy")) if prosody_emb_seg_p_dir.exists() else None

    # prepare text embedding
    context_embeddings = []
    for utt_id in utt_ids:
        current_txt_emb, history_txt_embs, hist_emb_len, history_speakers, history_emotions = get_embs(
            utt_id, text_emb_paths,
            utt2id, id2utt, use_hist_num,
            seg_emb_paths=text_seg_emb_paths,
            start_index=1 if config.use_situation_text == 0 else 0,
        )
        context_embeddings.append([current_txt_emb, history_txt_embs, hist_emb_len, history_speakers, history_emotions])

    # prepare prosody embedding
    acoustic_model_name = Path(config.acoustic.model_yaml).parent.stem
    prosody_embeddings = []
    for utt_id in utt_ids:
        if "wProsody" in acoustic_model_name:
            if len(prosody_emb_paths) > 0:
                _, h_prosody_emb, _, h_prosody_speakers, h_prosody_emotions = get_embs(
                    utt_id, prosody_emb_paths,
                    utt2id, id2utt, use_hist_num,
                    start_index=1, only_latest=True,
                    use_local_prosody_hist_idx=config.use_local_prosody_hist_idx,
                )
            else:
                h_prosody_emb = h_prosody_speakers = h_prosody_emotions = None
            if len(g_prosody_emb_paths) > 0:
                _, h_g_prosody_embs, _, _, _ = get_embs(
                    utt_id, g_prosody_emb_paths,
                    utt2id, id2utt, use_hist_num,
                    start_index=1
                )
            else:
                h_g_prosody_embs = None
            prosody_embeddings.append([h_prosody_emb, h_g_prosody_embs, h_prosody_speakers, h_prosody_emotions])
        elif "wPEProsody" in acoustic_model_name:
            (
                current_prosody_emb, current_prosody_emb_duration, current_prosody_emb_phonemes,
                hist_prosody_embs, hist_prosody_embs_lens, hist_prosody_embs_len, _, _,
                hist_local_prosody_emb, hist_local_prosody_speaker, hist_local_prosody_emotion
            ) = get_peprosody_embs(
                utt_id, prosody_emb_paths,
                utt2id, id2utt, use_prosody_hist_num, start_index=1,
                use_local_prosody_hist_idx=config.use_local_prosody_hist_idx,
                seg_emb_paths=prosody_seg_emb_paths,
                seg_d_emb_paths=prosody_seg_d_emb_paths, seg_p_emb_paths=prosody_seg_p_emb_paths,
            )
            prosody_embeddings.append([
                current_prosody_emb, current_prosody_emb_duration, current_prosody_emb_phonemes,
                hist_prosody_embs, hist_prosody_embs_lens, hist_prosody_embs_len,
                hist_local_prosody_emb, hist_local_prosody_speaker, hist_local_prosody_emotion
            ])
        else:
            prosody_embeddings.append([None, None, None, None])

    if config.num_eval_utts is not None and config.num_eval_utts > 0:
        lab_files = lab_files[: config.num_eval_utts]

    # Run synthesis for each utt.
    _synthesis = synthesis_PEProsody if "wPEProsody" in acoustic_model_name else synthesis

    attention_data: Dict[str, Dict[int, List[np.ndarray]]] = {}
    attention_len_num: Dict[int, int] = {}
    for lab_file, context_embedding, prosody_embedding in optional_tqdm(config.tqdm, desc="Utterance")(
        zip(lab_files, context_embeddings, prosody_embeddings)
    ):
        wav, attentions = _synthesis(  # type: ignore
            device, lab_file, context_embedding, prosody_embedding,
            acoustic_config.netG.speakers, acoustic_config.netG.emotions,
            acoustic_model, acoustic_out_scaler, vocoder_model
        )
        # TODO: prosody emb lenがcontext embと異なる場合，うまく計算できない．
        # dictを別に用意してあげる必要があり．現状は片方が = 1 の時のみの利用を推奨
        hist_prosody_emb_len = max(prosody_embedding[5], context_embedding[2])

        if attentions is not None:
            text_attn, speech_attn = attentions
            if hist_prosody_emb_len not in attention_len_num.keys():
                attention_len_num[hist_prosody_emb_len] = 1
                if text_attn is not None:
                    if "text" not in attention_data.keys():
                        attention_data["text"] = {}
                    attention_data["text"][hist_prosody_emb_len] = [text_attn.contiguous().view(-1).cpu().numpy()]
                if speech_attn is not None:
                    if "speech" not in attention_data.keys():
                        attention_data["speech"] = {}
                    attention_data["speech"][hist_prosody_emb_len] = [speech_attn.contiguous().view(-1).cpu().numpy()]
            else:
                attention_len_num[hist_prosody_emb_len] += 1
                if text_attn is not None:
                    attention_data["text"][hist_prosody_emb_len].append(text_attn.contiguous().view(-1).cpu().numpy())
                if speech_attn is not None:
                    attention_data["speech"][hist_prosody_emb_len].append(speech_attn.contiguous().view(-1).cpu().numpy())

        wav = np.clip(wav, -1.0, 1.0)

        utt_id = Path(lab_file).name.replace("-feats.npy", "")
        out_wav_path = out_dir / f"{utt_id}.wav"
        wavfile.write(
            out_wav_path,
            rate=config.sample_rate,
            data=(wav * 32767.0).astype(np.int16),
        )

    if len(attention_len_num) > 0:
        output_data = []
        for hist_len in sorted(list(attention_len_num.keys())):
            output_data.append(f"hist_len: {hist_len}:\n")
            if "text" in attention_data.keys():
                output_data.append("\ttext:\n")
                attention_mean = np.mean(np.array(attention_data["text"][hist_len]), axis=0)
                attention_std = np.std(np.array(attention_data["text"][hist_len]), axis=0)
                output_data.append(f"\t\tattention mean: {attention_mean}\n")
                output_data.append(f"\t\tattention std: {attention_std}\n")
            if "speech" in attention_data.keys():
                output_data.append("\tspeech:\n")
                attention_mean = np.mean(np.array(attention_data["speech"][hist_len]), axis=0)
                attention_std = np.std(np.array(attention_data["speech"][hist_len]), axis=0)
                output_data.append(f"\t\tattention mean: {attention_mean}\n")
                output_data.append(f"\t\tattention std: {attention_std}\n")
        with open(out_dir / "attention_info.txt", "w") as f:
            f.writelines(output_data)

    # Run synthesis by context of synthesized speech
    if "wPEProsodywoPEPCE" not in acoustic_model_name:
        out_wav_base = out_dir / "real_dialogue_synthesis"
        out_wav_base.mkdir(parents=True, exist_ok=True)

        # -1. SSL modelを利用するならここで準備しておく
        if config.SSL_name != "":
            if config.SSL_name == "WavLM":
                assert config.SSL_sample_rate == 16000, "sampling rateは16000のみ有効です"
                processor = Wav2Vec2FeatureExtractor.from_pretrained(config.SSL_weight)
                model = WavLMModel.from_pretrained(config.SSL_weight).to(device)
                histry_dummy_dim = 1024
            elif config.SSL_name == "wav2vec2":
                assert config.SSL_sample_rate == 16000, "sampling rateは16000のみ有効です"
                processor = Wav2Vec2FeatureExtractor.from_pretrained(config.SSL_weight)
                model = Wav2Vec2Model.from_pretrained(config.SSL_weight).to(device)
                histry_dummy_dim = 1024
            else:
                raise RuntimeError(f"model名: {config.SSL_name} は未対応です.")
        else:
            histry_dummy_dim = 80  # for mel

        # 0. prosody embをためる場所を新しく用意
        synthesis_prosdoy_emb_base = out_dir / "real_dialogue_synthesis_embs"
        synthesis_prosdoy_emb_base.mkdir(parents=True, exist_ok=True)
        # copyしておく
        preprocessed_prosody_emb_dir = Path(to_absolute_path(config.prosody_emb_dir))
        for prosody_emb_path in preprocessed_prosody_emb_dir.glob("*.npy"):
            shutil.copy(prosody_emb_path, synthesis_prosdoy_emb_base)
        if (preprocessed_prosody_emb_dir / "segment_duration").exists():
            synthesis_prosdoy_emb_seg_d_base = synthesis_prosdoy_emb_base / "segment_duration"
            synthesis_prosdoy_emb_seg_d_base.mkdir(parents=True, exist_ok=True)
            for prosody_emb_seg_d_path in (preprocessed_prosody_emb_dir / "segment_duration").glob("*.npy"):
                shutil.copy(prosody_emb_seg_d_path, synthesis_prosdoy_emb_seg_d_base)
        if (preprocessed_prosody_emb_dir / "segment_phonemes").exists():
            synthesis_prosdoy_emb_seg_p_base = synthesis_prosdoy_emb_base / "segment_phonemes"
            synthesis_prosdoy_emb_seg_p_base.mkdir(parents=True, exist_ok=True)
            for prosody_emb_seg_p_path in (preprocessed_prosody_emb_dir / "segment_phonemes").glob("*.npy"):
                shutil.copy(prosody_emb_seg_p_path, synthesis_prosdoy_emb_seg_p_base)
        # 1. 対話順に読み込む
        dialogue_keys = list(id2utt.keys())
        dialogue_keys = sorted(dialogue_keys, key=lambda x: (int(x[0]), int(x[1])), reverse=False)
        # 2. 対話順にデータを読み込み，処理していく
        synthesis_prosdoy_emb_list = list(synthesis_prosdoy_emb_base.glob("*.npy"))
        # ここでリークを避けるため，合成対象のファイルをすべて削除しておく
        for emb_path in synthesis_prosdoy_emb_list:
            utt_id = emb_path.stem.replace("-feats", "")
            if utt_id in utt_ids:
                os.remove(emb_path)

        attention_data_real: Dict[str, Dict[int, List[np.ndarray]]] = {}
        attention_len_num_real: Dict[int, int] = {}
        for dialogue_key in tqdm(dialogue_keys):
            if int(dialogue_key[1]) == 0:
                # turn=0は状況説明
                continue
            utt_id = id2utt[dialogue_key]

            if utt_id not in utt_ids:
                # 生成すべき発話でない場合(つまり，生徒とかの場合)
                continue
            context_embedding = get_embs(
                utt_id, text_emb_paths,
                utt2id, id2utt, use_hist_num,
                seg_emb_paths=text_seg_emb_paths,
                start_index=1 if config.use_situation_text == 0 else 0,
            )
            if "wProsody" in acoustic_model_name:
                # NOTE: 対応したかったら，PEProsody同様にcopyしてあげる必要あり
                assert RuntimeError("未対応")
            elif "wPEProsody" in acoustic_model_name:
                (
                    current_prosody_emb, current_prosody_emb_duration, current_prosody_emb_phonemes,
                    hist_prosody_embs, hist_prosody_embs_lens, hist_prosody_embs_len, _, _,
                    hist_local_prosody_emb, hist_local_prosody_speaker, hist_local_prosody_emotion
                ) = get_peprosody_embs(
                    utt_id, synthesis_prosdoy_emb_list,
                    utt2id, id2utt, use_prosody_hist_num, start_index=1,
                    use_local_prosody_hist_idx=config.use_local_prosody_hist_idx,
                    seg_d_emb_paths=prosody_seg_d_emb_paths, seg_p_emb_paths=prosody_seg_p_emb_paths,
                    emb_dim=histry_dummy_dim,
                )
                prosody_embedding = [
                    current_prosody_emb, current_prosody_emb_duration, current_prosody_emb_phonemes,
                    hist_prosody_embs, hist_prosody_embs_lens, hist_prosody_embs_len,
                    hist_local_prosody_emb, hist_local_prosody_speaker, hist_local_prosody_emotion
                ]
            else:
                prosody_embedding = [None, None, None, None]

            lab_file = in_dir / f"{utt_id.strip()}-feats.npy"

            if (config.mel_mode == 0) and (config.SSL_name == ""):
                # NOTE: 対応させたいなら，下の_synthsisでpitchとかを返せるようにすればいい.
                assert RuntimeError("未対応")
            wav, synthesised_mel, attentions = _synthesis(  # type: ignore
                device, lab_file, context_embedding, prosody_embedding,
                acoustic_config.netG.speakers, acoustic_config.netG.emotions,
                acoustic_model, acoustic_out_scaler, vocoder_model,
                need_mel=True,
            )
            wav = np.clip(wav, -1.0, 1.0)
            wavfile.write(
                out_wav_base / f"{utt_id}.wav",
                rate=config.sample_rate,
                data=(wav * 32767.0).astype(np.int16),
            )
            if attentions is not None:
                hist_prosody_emb_len = max(prosody_embedding[5], context_embedding[2])
                text_attn, speech_attn = attentions
                if hist_prosody_emb_len not in attention_len_num_real.keys():
                    attention_len_num_real[hist_prosody_emb_len] = 1
                    if text_attn is not None:
                        if "text" not in attention_data_real.keys():
                            attention_data_real["text"] = {}
                        attention_data_real["text"][hist_prosody_emb_len] = [text_attn.contiguous().view(-1).cpu().numpy()]
                    if speech_attn is not None:
                        if "speech" not in attention_data_real.keys():
                            attention_data_real["speech"] = {}
                        attention_data_real["speech"][hist_prosody_emb_len] = [speech_attn.contiguous().view(-1).cpu().numpy()]
                else:
                    attention_len_num_real[hist_prosody_emb_len] += 1
                    if text_attn is not None:
                        attention_data_real["text"][hist_prosody_emb_len].append(text_attn.contiguous().view(-1).cpu().numpy())
                    if speech_attn is not None:
                        attention_data_real["speech"][hist_prosody_emb_len].append(speech_attn.contiguous().view(-1).cpu().numpy())

            # 3. 新しいembeddingの用意
            # 上書き保存という形でコピーしたprosody embがたまったdirに保存
            # seg_dに関しては推論時には使わないので正解のままにしておいてok
            # seg_pはテキストから決まるsegmentごとの音素数なのでそのままでok
            # つまり，今後対話履歴として用いるmelだけ合成したもので上書きすればok
            if config.mel_mode == 1:
                np.save(
                    synthesis_prosdoy_emb_base / f"{utt_id}-feats.npy",
                    synthesised_mel.astype(np.float32),
                    allow_pickle=False,
                )
            elif config.SSL_name != "":
                _sr, x = wavfile.read(out_wav_base / f"{utt_id}.wav")
                if x.dtype in [np.int16, np.int32]:
                    x = (x / np.iinfo(x.dtype).max).astype(np.float64)
                wav = librosa.resample(x, _sr, config.SSL_sample_rate)
                inputs = processor(
                    wav, sampling_rate=config.SSL_sample_rate, return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                if config.SSL_name == "WavLM":
                    outputs = torch.tensor(np.array([tn.cpu().numpy() for tn in outputs.hidden_states]))[1:]
                    outputs = torch.mean(outputs.squeeze(1), dim=1).numpy()
                elif config.SSL_name == "wav2vec2":
                    outputs = outputs.last_hidden_state.cpu().squeeze(0)
                    outputs = torch.mean(outputs, dim=0, keepdim=True).numpy()
                np.save(
                    synthesis_prosdoy_emb_base / f"{utt_id}-feats.npy",
                    outputs.astype(np.float32),
                    allow_pickle=False,
                )

        if len(attention_len_num_real) > 0:
            output_data = []
            for hist_len in sorted(list(attention_len_num_real.keys())):
                output_data.append(f"hist_len: {hist_len}:\n")
                if "text" in attention_data_real.keys():
                    output_data.append("\ttext:\n")
                    attention_mean = np.mean(np.array(attention_data_real["text"][hist_len]), axis=0)
                    attention_std = np.std(np.array(attention_data_real["text"][hist_len]), axis=0)
                    output_data.append(f"\t\tattention mean: {attention_mean}\n")
                    output_data.append(f"\t\tattention std: {attention_std}\n")
                if "speech" in attention_data_real.keys():
                    output_data.append("\tspeech:\n")
                    attention_mean = np.mean(np.array(attention_data_real["speech"][hist_len]), axis=0)
                    attention_std = np.std(np.array(attention_data_real["speech"][hist_len]), axis=0)
                    output_data.append(f"\t\tattention mean: {attention_mean}\n")
                    output_data.append(f"\t\tattention std: {attention_std}\n")
            with open(out_wav_base / "attention_info_realdialogue.txt", "w") as f:
                f.writelines(output_data)

    # add reconstruct wav output
    out_dir = out_dir / "reconstruct"
    out_dir.mkdir(parents=True, exist_ok=True)
    if config.in_mel_dir is not None:
        in_mel_dir = Path(to_absolute_path(config.in_mel_dir))
        mel_files = [in_mel_dir / f"{utt_id.strip()}-feats.npy" for utt_id in utt_ids]

        if config.num_eval_utts is not None and config.num_eval_utts > 0:
            mel_files = mel_files[: config.num_eval_utts]

        for mel_file in optional_tqdm(config.tqdm, desc="Utterance")(mel_files):
            mel_org = np.load(mel_file)
            mel_org = acoustic_out_scaler.inverse_transform(mel_org)  # type: ignore
            mel_org = torch.Tensor(mel_org).unsqueeze(0).to(device)
            wav = vocoder_model(mel_org.transpose(1, 2)).squeeze(1).cpu().data.numpy()[0]
            utt_id = Path(mel_file).name.replace("-feats.npy", "")
            out_wav_path = out_dir / f"{utt_id}.wav"
            wavfile.write(
                out_wav_path,
                rate=config.sample_rate,
                data=(wav * 32767.0).astype(np.int16),
            )


def entry():
    my_app()


if __name__ == "__main__":
    my_app()

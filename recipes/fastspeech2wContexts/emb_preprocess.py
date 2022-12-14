import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, Wav2Vec2Model
import librosa
from scipy.io import wavfile

sys.path.append("../..")
from recipes.fastspeech2wContexts.preprocess import process_utterance
from vc_tts_template.utils import adaptive_load_state_dict, pad_1d, pad_2d
from recipes.fastspeech2.preprocess import process_lab_file


def init_for_list(input_):
    return input_.split(',')


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess for wContexts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dialogue_info", type=str, help="dialogue_info")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--BERT_weight", type=str)
    # for GMM
    parser.add_argument(
        "--input_duration_paths",
        type=init_for_list,
        help='durationのbasepathを「，」区切りで与えてください.',
        nargs='?'
    )
    parser.add_argument("--output_mel_file_paths", type=init_for_list, nargs='?')
    parser.add_argument("--model_config_paths", type=init_for_list, nargs='?')
    parser.add_argument("--pretrained_checkpoints", type=init_for_list, nargs='?')
    # for wav
    parser.add_argument("--input_wav_paths", type=init_for_list, nargs='?')
    parser.add_argument("--input_lab_paths", type=init_for_list, nargs='?')
    parser.add_argument("--input_textgrid_paths", type=init_for_list, nargs='?')
    parser.add_argument("--sample_rate", type=int)
    parser.add_argument("--filter_length", type=int)
    parser.add_argument("--hop_length", type=int)
    parser.add_argument("--win_length", type=int)
    parser.add_argument("--n_mel_channels", type=int)
    parser.add_argument("--mel_fmin", type=int)
    parser.add_argument("--mel_fmax", type=int)
    parser.add_argument("--clip", type=float)
    parser.add_argument("--log_base", type=str)
    parser.add_argument("--pitch_phoneme_averaging", type=int)
    parser.add_argument("--energy_phoneme_averaging", type=int)
    parser.add_argument("--SSL_name", type=str, nargs='?')
    parser.add_argument("--SSL_weight", type=str, nargs='?')
    parser.add_argument("--SSL_sample_rate", type=int, nargs='?')
    parser.add_argument("--mel_mode", type=int)
    parser.add_argument("--pau_split", type=int)
    parser.add_argument("--n_jobs", type=int)
    return parser


def split_text_by_pau(text: str):
    pau_cher = ["、", "。", "！", "？", "「", "」", "・", "…", "，", "!"]
    output = []

    flg = 0
    tmp = ""
    for i, c_ in enumerate(text):
        if c_ in pau_cher:
            tmp += c_
            if i == 0:
                # 先頭のpauは無視される
                continue
            flg = 1
        else:
            if flg == 1:
                # 切り替わり
                output.append(tmp)
                flg = 0
                tmp = ""
            tmp += c_
    output.append(tmp)
    return output


def get_text_embeddings(
    input_txt_file_path, output_dir, BERT_weight, pau_split, batch_size=16
):
    model = SentenceTransformer(BERT_weight)

    with open(input_txt_file_path, 'r') as f:
        texts = f.readlines()
    file_names = [text.strip().split(':')[0] for text in texts]
    texts = [text.strip().split(':')[-1] for text in texts]

    batch_id = []
    batch_text = []
    for i, (id_, text) in tqdm(enumerate(zip(file_names, texts))):
        batch_id.append(id_)
        batch_text.append(text)

        if ((i+1) % batch_size == 0) or (i == len(file_names)-1):
            output = model.encode(batch_text)
            for filename, text_emb in zip(batch_id, output):
                np.save(
                    output_dir / f"{filename}-feats.npy",
                    text_emb.astype(np.float32),
                    allow_pickle=False,
                )

            batch_id = []
            batch_text = []
        else:
            continue

    if pau_split is True:
        output_dir = output_dir / "segmented_text_emb"
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_id = []
        batch_text = []
        for i, (id_, text) in tqdm(enumerate(zip(file_names, texts))):
            output = model.encode(split_text_by_pau(text))
            np.save(
                output_dir / f"{id_}-feats.npy",
                np.array(output).astype(np.float32),
                allow_pickle=False,
            )

            batch_id = []
            batch_text = []


def get_prosody_embeddings_wGMM(
    input_duration_paths, output_mel_file_paths,
    output_dir,
    model_config_paths, pretrained_checkpoints,
    batch_size=16
):
    """対応するデータフォルダと, modelのpretrainを複数指定する.
    想定:
      dump/spk_sr22050/train/in_*をいくつか指定する.
      また, 既に正規化されている前提.
    """
    output_dir = Path(output_dir)
    (output_dir / "prosody_emb").mkdir(parents=True, exist_ok=True)
    (output_dir / "g_prosody_emb").mkdir(parents=True, exist_ok=True)
    # model load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for (
        input_duration_path_base, output_mel_file_path_base,
        model_config_path, pretrained_checkpoint
    ) in zip(
        input_duration_paths, output_mel_file_paths,
        model_config_paths, pretrained_checkpoints
    ):
        print(f"process {output_mel_file_path_base}...")
        # model init
        checkpoint = torch.load(to_absolute_path(pretrained_checkpoint), map_location=device)
        model_config = OmegaConf.load(to_absolute_path(model_config_path))
        model = hydra.utils.instantiate(model_config.netG).to(device)
        adaptive_load_state_dict(model, checkpoint["state_dict"])
        model.eval()
        assert model.global_prosody, "this is allowed only for the model which use global prosody"
        # data setting
        for s in ["train", "dev", "eval"]:
            input_duration_path = Path(input_duration_path_base.format(s))
            output_mel_file_path = Path(output_mel_file_path_base.format(s))

            batch_id = []
            batch_src_len = []
            batch_duration = []
            batch_mel = []
            input_duration_paths = list(input_duration_path.glob("*.npy"))
            for i, input_duration_path_ in tqdm(enumerate(input_duration_paths)):
                filename = input_duration_path_.name
                utt_id = filename.replace("-feats.npy", "")
                input_duration = np.load(input_duration_path_)
                output_mel = np.load(output_mel_file_path / filename)
                batch_id.append(utt_id)
                batch_src_len.append(len(input_duration))
                batch_duration.append(input_duration)
                batch_mel.append(output_mel)
                if ((i+1) % batch_size == 0) or (i == len(input_duration_paths)-1):
                    # prepare start
                    sort_idx = np.argsort(-np.array(batch_src_len))
                    batch_id = [batch_id[idx] for idx in sort_idx]
                    batch_src_len = [batch_src_len[idx] for idx in sort_idx]
                    batch_duration = [batch_duration[idx] for idx in sort_idx]
                    batch_mel = [batch_mel[idx] for idx in sort_idx]

                    batch_duration = pad_1d(batch_duration)
                    batch_mel = pad_2d(batch_mel)

                    batch_src_len = torch.tensor(batch_src_len, dtype=torch.long).to(device)
                    batch_duration = torch.tensor(batch_duration, dtype=torch.long).to(device)
                    batch_mel = torch.tensor(batch_mel, dtype=torch.float32).to(device)

                    with torch.no_grad():
                        prosody_target, g_prosody_target = model.prosody_extractor(
                            batch_mel, batch_duration, batch_src_len
                        )
                    for filename, prosody_emb, g_prosody_emb in zip(batch_id, prosody_target, g_prosody_target):
                        np.save(
                            output_dir / "prosody_emb" / f"{filename}-feats.npy",
                            prosody_emb.cpu().numpy().astype(np.float32),
                            allow_pickle=False,
                        )
                        np.save(
                            output_dir / "g_prosody_emb" / f"{filename}-feats.npy",
                            g_prosody_emb.cpu().numpy().astype(np.float32),
                            allow_pickle=False,
                        )

                    batch_id = []
                    batch_src_len = []
                    batch_duration = []
                    batch_mel = []
                else:
                    continue


def duration_split_by_pau(text: List[str], duration: np.ndarray):
    duration_output = []
    phone_output = []

    if len(text) == (len(duration) * 2):
        text = text[::2]

    assert len(text) == len(duration)

    cum_d = 0
    cnt = 0
    for p, d in zip(text, duration):
        cum_d += d
        cnt += 1
        if p in ["sp", "pau"]:
            duration_output.append(cum_d)
            phone_output.append(cnt)
            cum_d = 0
            cnt = 0
    duration_output.append(cum_d)
    phone_output.append(cnt)
    return np.array(duration_output), np.array(phone_output)


def _process_wav(
    wav_file, lab_file, sample_rate, filter_length, hop_length, win_length,
    n_mel_channels, mel_fmin, mel_fmax, clip, log_base,
    pitch_phoneme_averaging, energy_phoneme_averaging,
    input_context_path_postfix,
    mel_mode, pau_split, output_dir_prosody,
    output_dir_prosody_seg_d, output_dir_prosody_seg_p
):
    text, mel, pitch, energy, duration, utt_id = process_utterance(
        wav_file, lab_file, sample_rate, filter_length, hop_length, win_length,
        n_mel_channels, mel_fmin, mel_fmax, clip, log_base,
        pitch_phoneme_averaging, energy_phoneme_averaging,
        input_context_path_postfix == ".lab", return_utt_id=True
    )
    if mel_mode is True:
        np.save(
            output_dir_prosody / f"{utt_id}-feats.npy",
            mel.astype(np.float32),
            allow_pickle=False,
        )
    else:
        prosody = np.stack([pitch, energy], -1)
        np.save(
            output_dir_prosody / f"{utt_id}-feats.npy",
            prosody.astype(np.float32),
            allow_pickle=False,
        )
    if pau_split is True:
        seg_duration, seg_phoneme = duration_split_by_pau(text, duration)
        np.save(
            output_dir_prosody_seg_d / f"{utt_id}-feats.npy",
            seg_duration.astype(np.int32),
            allow_pickle=False,
        )
        np.save(
            output_dir_prosody_seg_p / f"{utt_id}-feats.npy",
            seg_phoneme.astype(np.int32),
            allow_pickle=False,
        )
    return utt_id


def _process_wav_bySSL(
    wav_files, lab_files,
    accent_info, hop_length,
    SSL_name, SSL_weight, SSL_sample_rate,
    output_dir, pau_split,
    output_dir_prosody_seg_p
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if SSL_name == "WavLM":
        assert SSL_sample_rate == 16000, "sampling rateは16000のみ有効です"
        processor = Wav2Vec2FeatureExtractor.from_pretrained(SSL_weight)
        model = WavLMModel.from_pretrained(SSL_weight).to(device)
    elif SSL_name == "wav2vec2":
        assert SSL_sample_rate == 16000, "sampling rateは16000のみ有効です"
        processor = Wav2Vec2FeatureExtractor.from_pretrained(SSL_weight)
        model = Wav2Vec2Model.from_pretrained(SSL_weight).to(device)
    else:
        raise RuntimeError(f"model名: {SSL_name} は未対応です.")

    if pau_split is True:
        output_seg_dir = output_dir / "segmented_prosody_emb"
        output_seg_dir.mkdir(parents=True, exist_ok=True)

    utt_ids = []
    for wav_path, lab_path in tqdm(zip(wav_files, lab_files)):
        utt_id = wav_path.stem

        text, duration, start, end = process_lab_file(
            lab_path, accent_info, SSL_sample_rate, hop_length
        )

        _sr, x = wavfile.read(wav_path)
        if x.dtype in [np.int16, np.int32]:
            x = (x / np.iinfo(x.dtype).max).astype(np.float64)
        wav = librosa.resample(x, _sr, SSL_sample_rate)
        wav = wav[
            int(SSL_sample_rate * start): int(SSL_sample_rate * end)
        ].astype(np.float32)

        inputs = processor(
            wav, sampling_rate=SSL_sample_rate, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        if SSL_name == "WavLM":
            outputs = torch.tensor(np.array([tn.cpu().numpy() for tn in outputs.hidden_states]))[1:]
            outputs = torch.mean(outputs.squeeze(1), dim=1).numpy()
        elif SSL_name == "wav2vec2":
            outputs = outputs.last_hidden_state.cpu().squeeze(0)
            outputs = torch.mean(outputs, dim=0, keepdim=True).numpy()

        np.save(
            output_dir / f"{utt_id}-feats.npy",
            outputs.astype(np.float32),
            allow_pickle=False,
        )
        utt_ids.append(utt_id)

        if pau_split is True:
            seg_duration, seg_phoneme = duration_split_by_pau(text, duration)

            before_d = 0
            outputs = []
            for d in seg_duration:
                seg_wav = wav[before_d*hop_length:(before_d+d)*hop_length]
                seg_input = processor(
                    seg_wav, sampling_rate=SSL_sample_rate, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    seg_output = model(**seg_input, output_hidden_states=True)
                # [1:]で embedding を除去. 先頭に入っている根拠は，実装.
                if SSL_name == "WavLM":
                    seg_output = torch.tensor(np.array([tn.cpu().numpy() for tn in seg_output.hidden_states]))[1:]
                    seg_output = torch.mean(seg_output.squeeze(1), dim=1).numpy()
                elif SSL_name == "wav2vec2":
                    seg_output = seg_output.last_hidden_state.cpu().squeeze(0)
                    seg_output = torch.mean(seg_output, dim=0, keepdim=True).numpy()
                outputs.append(seg_output)

                before_d += d

            # NOTE: 時間方向にmeanを取っているから以下はできること．
            outputs = np.concatenate(outputs)
            np.save(
                output_seg_dir / f"{utt_id}-feats.npy",
                np.array(outputs).astype(np.float32),
                allow_pickle=False,
            )
            np.save(
                output_dir_prosody_seg_p / f"{utt_id}-feats.npy",
                seg_phoneme.astype(np.int32),
                allow_pickle=False,
            )

    return utt_ids


def get_prosody_embeddings_wWav(
    input_wav_paths, input_lab_paths, input_textgrid_paths, output_dir,
    sample_rate, filter_length, hop_length, win_length,
    n_mel_channels, mel_fmin, mel_fmax, clip, log_base,
    pitch_phoneme_averaging, energy_phoneme_averaging,
    SSL_name, SSL_weight, SSL_sample_rate,
    mel_mode, pau_split, n_jobs
):
    if (input_lab_paths is not None) and (input_textgrid_paths is not None):
        raise ValueError("labを利用したいのか, textgridを利用したいのか, どちらか一方のみを埋めてください")
    if (SSL_name is not None) and (mel_mode == 1):
        raise ValueError("melを利用したいのか, SSLmodelを利用したいのか一方を選択してください.")

    input_context_path = input_lab_paths if input_lab_paths is not None else input_textgrid_paths
    input_context_path_postfix = ".lab" if input_lab_paths is not None else ".TextGrid"

    output_dir = Path(output_dir)
    output_dir_prosody = output_dir / "prosody_emb"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_prosody.mkdir(parents=True, exist_ok=True)

    if pau_split is True:
        output_dir_prosody_seg_p = output_dir_prosody / "segment_phonemes"
        output_dir_prosody_seg_p.mkdir(parents=True, exist_ok=True)

        if SSL_name is None:
            # mel modeの時に使いたい
            output_dir_prosody_seg_d = output_dir_prosody / "segment_duration"
            output_dir_prosody_seg_d.mkdir(parents=True, exist_ok=True)
    else:
        output_dir_prosody_seg_p = output_dir_prosody_seg_d = None

    utt_ids = []

    for input_wav_path_base, input_context_path_base in zip(input_wav_paths, input_context_path):
        wav_files = list(Path(input_wav_path_base).glob("*.wav"))
        lab_files = [
            Path(input_context_path_base) / (input_wav_path.stem + input_context_path_postfix)
            for input_wav_path in wav_files
        ]
        if SSL_name is not None:
            utt_ids = _process_wav_bySSL(
                wav_files, lab_files,
                input_context_path_postfix == ".lab", hop_length,
                SSL_name, SSL_weight, SSL_sample_rate,
                output_dir_prosody, pau_split,
                output_dir_prosody_seg_p
            )
        else:
            with ProcessPoolExecutor(n_jobs) as executor:
                futures = [
                    executor.submit(
                        _process_wav,
                        wav_file,
                        lab_file,
                        sample_rate,
                        filter_length,
                        hop_length,
                        win_length,
                        n_mel_channels,
                        mel_fmin,
                        mel_fmax,
                        clip,
                        log_base,
                        pitch_phoneme_averaging > 0,
                        energy_phoneme_averaging > 0,
                        input_context_path_postfix,
                        mel_mode,
                        pau_split,
                        output_dir_prosody,
                        output_dir_prosody_seg_d,
                        output_dir_prosody_seg_p
                    )
                    for wav_file, lab_file in zip(wav_files, lab_files)
                ]
                for future in tqdm(futures):
                    utt_id = future.result()
                    utt_ids.append(utt_id+'\n')
    with open(output_dir / "prosody_emb.list", "w") as f:
        f.writelines(utt_ids)


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    in_dir = Path(args.out_dir)
    in_text_emb_dir = in_dir / "text_emb"

    in_dir.mkdir(parents=True, exist_ok=True)
    in_text_emb_dir.mkdir(parents=True, exist_ok=True)
    print("get text embeddings...")
    get_text_embeddings(
        args.dialogue_info, in_text_emb_dir, args.BERT_weight, args.pau_split > 0
    )

    if (args.input_duration_paths is not None) and (args.input_wav_paths is not None):
        raise ValueError("GMMを利用したいのか, wavを利用したいのか, どちらか一方のみを埋めてください")

    if args.input_duration_paths is not None:
        print("get prosody embeddings from GMM...")
        get_prosody_embeddings_wGMM(
            args.input_duration_paths, args.output_mel_file_paths,
            args.out_dir, args.model_config_paths, args.pretrained_checkpoints
        )
    elif args.input_wav_paths is not None:
        print("get prosody embeddings from wavs...")
        get_prosody_embeddings_wWav(
            args.input_wav_paths, args.input_lab_paths, args.input_textgrid_paths, args.out_dir,
            args.sample_rate, args.filter_length, args.hop_length, args.win_length,
            args.n_mel_channels, args.mel_fmin, args.mel_fmax, args.clip, args.log_base,
            args.pitch_phoneme_averaging, args.energy_phoneme_averaging,
            args.SSL_name, args.SSL_weight, args.SSL_sample_rate,
            args.mel_mode > 0, args.pau_split > 0, args.n_jobs,
        )

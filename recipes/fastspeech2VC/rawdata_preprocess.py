import argparse
import sys
import shutil

from pathlib import Path
import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess for Fastspeech2VC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("src_name", type=str, help="source name")
    parser.add_argument("tgt_name", type=str, help="target name")
    parser.add_argument("src_wav_root", type=str, help="wav root")
    parser.add_argument("tgt_wav_root", type=str, help="wav root")
    parser.add_argument("--output_root", type=str, help="wav root")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    src_name = args.src_name
    tgt_name = args.tgt_name
    src_wav_root = Path(args.src_wav_root)
    tgt_wav_root = Path(args.tgt_wav_root)

    src_wav_set = set([wav_path.name for wav_path in src_wav_root.glob("**/*.wav")])
    tgt_wav_set = set([wav_path.name for wav_path in tgt_wav_root.glob("**/*.wav")])
    wav_list = src_wav_set & tgt_wav_set

    src_wav_out = Path(args.output_root) / (src_name + '_' + tgt_name) / "source"
    tgt_wav_out = Path(args.output_root) / (src_name + '_' + tgt_name) / "target"

    src_wav_out.mkdir(parents=True, exist_ok=True)
    tgt_wav_out.mkdir(parents=True, exist_ok=True)

    for wav_name in tqdm.tqdm(wav_list):
        wav_src_path = list(src_wav_root.glob(f"**/{wav_name}"))[0]
        wav_tgt_path = list(tgt_wav_root.glob(f"**/{wav_name}"))[0]
        wav_src_path_new = src_wav_out / (src_name + '_' + tgt_name + '_' + wav_src_path.name)
        wav_tgt_path_new = tgt_wav_out / (src_name + '_' + tgt_name + '_' + wav_tgt_path.name)
        shutil.copy(wav_src_path, wav_src_path_new)
        shutil.copy(wav_tgt_path, wav_tgt_path_new)

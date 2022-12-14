import argparse
import sys
from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class MultiSpeakerStandardScaler():
    def __init__(self, speakers: Optional[List[str]] = None):
        if speakers is None:
            speakers = [""]
            self.multi_speaker = False
        else:
            self.multi_speaker = True
        self.speakers = {'JSUT': 0, 'NICT': 1, 'Teacher': 2, 'FStudent': 3, 'MStudent': 4}
        # speakers
        # 一時的措置
        print(speakers)
        self.scalers_dict = {
            spk: StandardScaler() for spk in speakers
        }

    def partial_fit(self, X, speaker=""):
        if speaker not in self.speakers:
            raise KeyError("存在する話者かどうかを確認して下さい")
        self.scalers_dict[speaker].partial_fit(X)

    def transform(self, X, speaker=""):
        if speaker not in self.speakers:
            raise KeyError("存在する話者かどうかを確認して下さい")
        y = self.scalers_dict[speaker].transform(X)
        return y

    def inverse_transform(self, X, speaker=""):
        print(self.speakers)
        print(speaker)
        self.speakers = {'JSUT': 0, 'NICT': 1, 'Teacher': 2, 'FStudent': 3, 'MStudent': 4}
        # speakers
        # 一時的措置
        if speaker not in self.speakers:
            raise KeyError("存在する話者かどうかを確認して下さい")
        y = self.scalers_dict[speaker].inverse_transform(X)
        return y

    @property
    def mean_(self):
        output = [scaler.mean_ for scaler in self.scalers_dict.values()]
        feat_lens = [len(scaler.mean_) for scaler in self.scalers_dict.values()]

        if len(set(feat_lens)) > 1:
            raise RuntimeError("all speakers feats must have same number of features!")

        return np.array(output)

    @property
    def scale_(self):
        output = [scaler.scale_ for scaler in self.scalers_dict.values()]
        feat_lens = [len(scaler.scale_) for scaler in self.scalers_dict.values()]

        if len(set(feat_lens)) > 1:
            raise RuntimeError("all speakers feats must have same number of features!")

        return np.array(output)

    @property
    def var_(self):
        output = [scaler.var_ for scaler in self.scalers_dict.values()]
        feat_lens = [len(scaler.var_) for scaler in self.scalers_dict.values()]

        if len(set(feat_lens)) > 1:
            raise RuntimeError("all speakers feats must have same number of features!")

        return np.array(output)


def init_for_list(input_):
    return input_.split(',')


def get_parser():
    parser = argparse.ArgumentParser(description="Fit scalers")
    parser.add_argument("utt_list", type=str, help="utternace list")
    parser.add_argument("in_dir", type=str, help="in directory")
    parser.add_argument("out_path", type=str, help="Output path")
    parser.add_argument("--external_scaler", type=str, help="External scaler")
    parser.add_argument(
        "--speakers_list",
        type=init_for_list,
        help='speaker nameを「，」区切りで与えてください.',
        nargs='?'
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    in_dir = Path(args.in_dir)
    if args.external_scaler is not None:
        scaler = joblib.load(args.external_scaler)
    else:
        scaler = MultiSpeakerStandardScaler(
            args.speakers_list
        )
    with open(args.utt_list) as f:
        for utt_id in tqdm(f):
            c = np.load(in_dir / f"{utt_id.strip()}-feats.npy")
            if len(c.shape) == 1:
                c = c.reshape(-1, 1)
            # print(args.speakers_list)
            speaker = utt_id.split("_")[0] if args.speakers_list is not None else ""
            #speaker = {"JSUT": 0, "NICT": 1, "Teacher": 2, "FStudent": 3, "MStudent": 4}
            print(speaker)
            scaler.partial_fit(
                c, speaker
            )
        joblib.dump(scaler, args.out_path)

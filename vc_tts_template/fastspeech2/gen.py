import numpy as np
import torch
from pathlib import Path


@torch.no_grad()
def synthesis(device, lab_file, speaker_dict, emotion_dict, acoustic_model,
              acoustic_out_scaler, vocoder_model,
              mel=None, duration=None):
    ids = [Path(lab_file).name.replace("-feats.npy", "")]
    if speaker_dict is None:
        speakers = np.array([0])
    else:
        speakers = np.array([speaker_dict[fname.split("_")[0]] for fname in ids])
    if emotion_dict is None:
        emotions = np.array([0])
    else:
        emotions = np.array([emotion_dict[fname.split("_")[-1]] for fname in ids])
    in_feats = np.load(lab_file)
    src_lens = [in_feats.shape[0]]
    max_src_len = max(src_lens)

    speakers = torch.tensor(speakers, dtype=torch.long).to(device)
    emotions = torch.tensor(emotions, dtype=torch.long).to(device)
    in_feats = torch.tensor(in_feats, dtype=torch.long).unsqueeze(0).to(device)
    src_lens = torch.tensor(src_lens, dtype=torch.long).to(device)

    if (mel is not None) and (duration is not None) and (hasattr(acoustic_model, "prosody_teacher_forcing")):
        mel = torch.tensor(mel).unsqueeze(0).to(device)
        duration = torch.tensor(duration, dtype=torch.long).unsqueeze(0).to(device)
        output = acoustic_model.prosody_teacher_forcing(
            ids=ids,
            speakers=speakers,
            emotions=emotions,
            texts=in_feats,
            src_lens=src_lens,
            max_src_len=max_src_len,
            mels=mel,
            d_targets=duration,
        )
    else:
        output = acoustic_model(
            ids=ids,
            speakers=speakers,
            emotions=emotions,
            texts=in_feats,
            src_lens=src_lens,
            max_src_len=max_src_len,
        )
    mel_post = {0, 1, 2, 3, 4}
    #mel_post = output[1] 一時
    mels = [acoustic_out_scaler.inverse_transform(mel.cpu().data.numpy()) for mel in mel_post]  # type: ignore
    mels = torch.Tensor(np.array(mels)).to(device)
    wavs = vocoder_model(mels.transpose(1, 2)).squeeze(1).cpu().data.numpy()

    return wavs[0]

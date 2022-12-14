import numpy as np
import torch
from pathlib import Path


@torch.no_grad()
def synthesis(device, data, speaker_dict, emotion_dict, acoustic_model,
              acoustic_out_scaler, vocoder_model):

    ids = [Path(data[0]).name.replace("-feats.npy", "")]
    if speaker_dict is None:
        s_speakers = np.array([0])
        t_speakers = np.array([0])
    else:
        s_speakers = np.array([speaker_dict[fname.split("_")[0]] for fname in ids])
        t_speakers = np.array([speaker_dict[fname.split("_")[1]] for fname in ids])
    if emotion_dict is None:
        s_emotions = np.array([0])
        t_emotions = np.array([0])
    else:
        s_emotions = np.array([emotion_dict[fname.split("_")[-2]] for fname in ids])
        t_emotions = np.array([emotion_dict[fname.split("_")[-1]] for fname in ids])
    s_mels = np.array([np.load(data[0])])
    s_pitches = np.array([np.load(data[1])])
    s_energies = np.array([np.load(data[2])])

    s_mel_lens = np.array([s_mels[0].shape[0]])
    max_s_mel_len = max(s_mel_lens)

    s_speakers = torch.tensor(s_speakers).long().to(device)
    t_speakers = torch.tensor(t_speakers).long().to(device)
    s_emotions = torch.tensor(s_emotions).long().to(device)
    t_emotions = torch.tensor(t_emotions).long().to(device)
    s_mels = torch.tensor(s_mels).float().to(device)
    s_mel_lens = torch.tensor(s_mel_lens).long().to(device)
    s_pitches = torch.tensor(s_pitches).float().to(device)
    s_energies = torch.tensor(s_energies).float().to(device)

    if data[3].exists():
        s_snt_durations = np.array([np.load(data[3])])
        s_snt_durations = torch.tensor(s_snt_durations).long().to(device)
        output = acoustic_model(
            ids, s_speakers, t_speakers, s_emotions, t_emotions,
            s_mels, s_mel_lens, max_s_mel_len, s_pitches, s_energies,
            s_snt_durations=s_snt_durations
        )
    else:
        output = acoustic_model(
            ids, s_speakers, t_speakers, s_emotions, t_emotions,
            s_mels, s_mel_lens, max_s_mel_len, s_pitches, s_energies
        )

    mel_post = output[1]
    mels = [acoustic_out_scaler.inverse_transform(mel.cpu().data.numpy()) for mel in mel_post]  # type: ignore
    mels = torch.Tensor(np.array(mels)).to(device)
    wavs = vocoder_model(mels.transpose(1, 2)).squeeze(1).cpu().data.numpy()

    return wavs[0]

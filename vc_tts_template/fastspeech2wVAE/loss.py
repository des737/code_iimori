import torch
import torch.nn as nn
import numpy as np


class FastSpeech2wVAELoss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(
        self, pitch_feature_level, energy_feature_level,
        anneal_function, anneal_lag, anneal_k,
        anneal_x0, anneal_upper
    ):
        super(FastSpeech2wVAELoss, self).__init__()
        self.pitch_feature_level = "phoneme_level" if pitch_feature_level > 0 else "frame_level"
        self.energy_feature_level = "phoneme_level" if energy_feature_level > 0 else "frame_level"

        self.anneal_function = anneal_function
        self.lag = anneal_lag
        self.k = anneal_k
        self.x0 = anneal_x0
        self.upper = anneal_upper
        self.step = 0

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def kl_anneal_function(self):
        self.step += 1
        if self.anneal_function == 'logistic':
            return float(self.upper/(self.upper+np.exp(-self.k*(self.step-self.x0))))
        elif self.anneal_function == 'linear':
            if self.step > self.lag:
                return min(self.upper, self.step/self.x0)
            else:
                return 0
        elif self.anneal_function == 'constant':
            return 0.001

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[-6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            mu,
            logvar,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_weight = self.kl_anneal_function()

        total_loss = mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + kl_weight*kl_loss

        loss_values = {
            "mel_loss": mel_loss.item(),
            "postnet_mel_loss": postnet_mel_loss.item(),
            "pitch_loss": pitch_loss.item(),
            "energy_loss": energy_loss.item(),
            "duration_loss": duration_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item()
        }

        return total_loss, loss_values

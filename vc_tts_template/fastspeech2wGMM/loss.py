import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, pitch_feature_level, energy_feature_level, beta, g_beta=None):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = "phoneme_level" if pitch_feature_level > 0 else "frame_level"
        self.energy_feature_level = "phoneme_level" if energy_feature_level > 0 else "frame_level"
        self.beta = beta
        self.g_beta = g_beta

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

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
            prosody_features
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

        # mdn loss
        prosody_target, pi_outs, sigma_outs, mu_outs = prosody_features[:4]

        if pi_outs is not None:
            normal_dist = Normal(loc=mu_outs, scale=(sigma_outs + 1e-3))
            loglik = normal_dist.log_prob(prosody_target.detach().unsqueeze(2).expand_as(normal_dist.loc))
            # 共分散行列は対角行列という仮定なので, 確率は各次元で計算後logとって和をとればよい.
            loglik = torch.sum(loglik, dim=-1)
            # logsumexpを使わないとunderflowする.
            prosody_loss = -torch.logsumexp(torch.log(pi_outs+1e-7) + loglik, dim=-1)
            prosody_loss = torch.mean(prosody_loss.masked_select(src_masks))
        else:
            # for local_prosody: False
            prosody_loss = torch.tensor([0], dtype=torch.float16).to(pitch_loss.device)

        if len(prosody_features) > 4:
            # global embedding True
            g_prosody_target, g_pi, g_sigma, g_mu = prosody_features[4:]
            normal_dist = Normal(loc=g_mu, scale=(g_sigma+1e-3))
            loglik = normal_dist.log_prob(g_prosody_target.detach().unsqueeze(1).expand_as(normal_dist.loc))
            # 共分散行列は対角行列という仮定なので, 確率は各次元で計算後logとって和をとればよい.
            loglik = torch.sum(loglik, dim=-1)
            # logsumexpを使わないとunderflowする.
            g_prosody_loss = -torch.logsumexp(torch.log(g_pi+1e-7) + loglik, dim=-1)
            g_prosody_loss = torch.mean(g_prosody_loss)

            total_loss = mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + \
                self.beta*prosody_loss + self.g_beta*g_prosody_loss

            loss_values = {
                "mel_loss": mel_loss.item(),
                "postnet_mel_loss": postnet_mel_loss.item(),
                "pitch_loss": pitch_loss.item(),
                "energy_loss": energy_loss.item(),
                "duration_loss": duration_loss.item(),
                "prosody_loss": prosody_loss.item(),
                "global_prosody_loss": g_prosody_loss.item(),
                "total_loss": total_loss.item()
            }

        else:
            total_loss = mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + self.beta*prosody_loss

            loss_values = {
                "mel_loss": mel_loss.item(),
                "postnet_mel_loss": postnet_mel_loss.item(),
                "pitch_loss": pitch_loss.item(),
                "energy_loss": energy_loss.item(),
                "duration_loss": duration_loss.item(),
                "prosody_loss": prosody_loss.item(),
                "total_loss": total_loss.item()
            }

        return total_loss, loss_values

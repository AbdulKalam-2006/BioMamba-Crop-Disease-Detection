import torch
import torch.nn as nn

class BioDiffusionLoss(nn.Module):
    def __init__(self, alpha=0.002, class_weights=None, label_smoothing=0.02):
        super(BioDiffusionLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(self, logits, targets, ssm_hidden_states):
        ce = self.ce_loss(logits, targets)

        # Smoothness penalty along vein-scanned tokens
        diff = torch.abs(ssm_hidden_states[:, 1:, :] - ssm_hidden_states[:, :-1, :])
        diffusion_penalty = torch.mean(diff)

        total_loss = ce + self.alpha * diffusion_penalty
        return total_loss, ce, diffusion_penalty
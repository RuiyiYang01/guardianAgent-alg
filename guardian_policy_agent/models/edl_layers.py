# guardian_policy_agent/models/edl_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialGuardianNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2,
                 use_interaction: bool = False):
        super(EvidentialGuardianNet, self).__init__()

        self.use_interaction = use_interaction

        if use_interaction:
            # Interaction features: [b, p, b*p, |b-p|] → 4x input_dim
            self.fc1 = nn.Linear(input_dim * 4, hidden_dim)
        else:
            # Legacy: simple concatenation [b, p] → 2x input_dim
            self.fc1 = nn.Linear(input_dim * 2, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.2)

    def forward(self, behavior_vec: torch.Tensor, policy_vec: torch.Tensor):
        """
        Forward Pass
        :param behavior_vec: [Batch, input_dim]
        :param policy_vec:   [Batch, input_dim]
        :return: alpha parameters for Dirichlet distribution (strictly positive)
        """
        if self.use_interaction:
            # Interaction features for learning similarity patterns
            # (standard approach from InferSent / SBERT cross-encoder)
            x = torch.cat([
                behavior_vec,
                policy_vec,
                behavior_vec * policy_vec,         # element-wise product (similarity signal)
                torch.abs(behavior_vec - policy_vec),  # absolute difference (contrast signal)
            ], dim=1)
        else:
            x = torch.cat([behavior_vec, policy_vec], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        logits = self.fc3(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0

        return alpha

    def predict_uncertainty(self, behavior_vec: torch.Tensor, policy_vec: torch.Tensor):
        """
        推理专用函数，返回 (Risk_Score, Uncertainty)
        """
        self.eval()
        with torch.no_grad():
            alpha = self.forward(behavior_vec, policy_vec)  # [Batch, 2]

            S = torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha / S  # [Batch, 2] -> [Prob_Safe, Prob_Risky]
            uncertainty = 2.0 / torch.squeeze(S)
            risk_score = probs[:, 1]

            return risk_score, uncertainty

import torch
import os
from guardian_policy_agent.models.vectorizer import SimpleFeatureEncoder
from guardian_policy_agent.models.edl_layers import EvidentialGuardianNet
from guardian_policy_agent.models.loss import edl_mse_loss
import torch.optim as optim

def train_dummy():
    print(">>> Phase 1: Pre-training System 1 (Dummy Data)")
    encoder = SimpleFeatureEncoder()
    model = EvidentialGuardianNet(input_dim=encoder.input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy data creation (Behavior, Policy) -> Label
    # Label: [1, 0] = Safe, [0, 1] = Risky
    dummy_b = {"data_categories": ["Contact"], "actions": ["Collect"]}
    dummy_p_match = {"data_categories": ["Contact"], "actions": ["Collect"]} # Match
    dummy_p_mismatch = {"data_categories": ["Financial"], "actions": ["Share"]} # Mismatch

    b_vec = encoder.vectorize(dummy_b).unsqueeze(0)
    p_vec_safe = encoder.vectorize(dummy_p_match).unsqueeze(0)
    p_vec_risk = encoder.vectorize(dummy_p_mismatch).unsqueeze(0)

    target_safe = torch.tensor([[1.0, 0.0]])
    target_risk = torch.tensor([[0.0, 1.0]])

    print("Training for 10 epochs to initialize weights...")
    model.train()
    for i in range(10):
        optimizer.zero_grad()
        # Train Safe Case
        out_safe = model(b_vec, p_vec_safe)
        loss1 = edl_mse_loss(out_safe, target_safe, i, 2, 10)
        # Train Risky Case
        out_risk = model(b_vec, p_vec_risk)
        loss2 = edl_mse_loss(out_risk, target_risk, i, 2, 10)

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

    save_path = "checkpoints/sys1_opp_pretrained.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_dummy()
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import os
import glob
import numpy as np
from tqdm import tqdm  # 引入进度条库

from guardian_policy_agent.models.vectorizer import SimpleFeatureEncoder
from guardian_policy_agent.models.edl_layers import EvidentialGuardianNet
from guardian_policy_agent.models.loss import edl_mse_loss

# === 配置 ===
ANNOTATIONS_DIR = "data/raw/OPP-115/annotations"
SAVE_PATH = "checkpoints/sys1_opp_pretrained.pth"
BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001

# === 映射规则 (保持不变) ===
KEYWORD_MAPPING = {
    "location": "Location", "gps": "Location",
    "contact": "Contact", "email": "Contact", "phone": "Contact", "address": "Contact",
    "demographic": "Demographic", "gender": "Demographic", "age": "Demographic",
    "health": "Health", "medical": "Health",
    "financial": "Financial", "credit card": "Financial", "bank": "Financial",
    "device id": "DeviceID", "imei": "DeviceID", "mac address": "DeviceID",
    "ip address": "IPAddress", "cookies": "cookies",
    "browser history": "BrowsingHistory",
    "collection": "Collect", "collect": "Collect",
    "sharing": "Share", "share": "Share", "disclosure": "Share",
    "use": "Use",
    "marketing": "Marketing", "advertising": "Advertising",
    "analytics": "Analytics", "legal": "Legal", "security": "Security"
}

class OPP115Dataset(Dataset):
    def __init__(self, annotation_dir, encoder):
        self.encoder = encoder
        self.samples = []
        self.vocab_data = [k for k in encoder.data_map.keys()]

        file_list = glob.glob(os.path.join(annotation_dir, "*.csv"))
        if not file_list:
            raise FileNotFoundError(f"No CSV files found in {annotation_dir}")

        print(f"Found {len(file_list)} annotation files. Parsing...")

        # 使用 tqdm 显示文件解析进度
        for csv_file in tqdm(file_list, desc="Parsing OPP-115 Files"):
            self._parse_file(csv_file)

        print(f"Processed complete. Total synthetic samples generated: {len(self.samples)}")

    def _parse_file(self, filepath):
        try:
            df = pd.read_csv(filepath, header=None, on_bad_lines='skip')
        except Exception as e:
            # print(f"Skipping {filepath}: {e}") # 减少噪音
            return

        current_data = set()
        current_actions = set()
        current_purposes = set()

        for idx, row in df.iterrows():
            row_text = " ".join([str(x).lower() for x in row.values])
            for kw, target in KEYWORD_MAPPING.items():
                if kw in row_text:
                    if target in self.encoder.data_map:
                        current_data.add(target)
                    elif target in self.encoder.action_map:
                        current_actions.add(target)
                    elif target in self.encoder.purpose_map:
                        current_purposes.add(target)

            if len(current_data) > 0 and len(current_actions) > 0:
                self._generate_contrastive_samples(list(current_data), list(current_actions), list(current_purposes))
                current_data = set()
                current_actions = set()
                current_purposes = set()

    def _generate_contrastive_samples(self, data_cats, actions, purposes):
        policy_dict = {"data_categories": data_cats, "actions": actions, "purposes": purposes}

        # Positive Sample
        pos_behavior = {"data_categories": data_cats, "actions": actions, "purposes": purposes}
        self.samples.append((pos_behavior, policy_dict, torch.tensor([1.0, 0.0])))

        # Negative Sample
        risk_candidates = list(set(self.vocab_data) - set(data_cats))
        if risk_candidates:
            fake_data = random.choice(risk_candidates)
            neg_behavior = {"data_categories": [fake_data], "actions": actions, "purposes": purposes}
            self.samples.append((neg_behavior, policy_dict, torch.tensor([0.0, 1.0])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b_dict, p_dict, label = self.samples[idx]
        b_vec = self.encoder.vectorize(b_dict)
        p_vec = self.encoder.vectorize(p_dict)
        return b_vec, p_vec, label

def train():
    # 1. Device Setup (自动检测 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Training on device: {device}")
    if device.type == 'cuda':
        print(f">>> GPU Name: {torch.cuda.get_device_name(0)}")

    encoder = SimpleFeatureEncoder()
    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"Error: Path {ANNOTATIONS_DIR} does not exist.")
        return

    dataset = OPP115Dataset(ANNOTATIONS_DIR, encoder)
    if len(dataset) == 0:
        print("Error: No samples generated.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 将模型移动到 GPU
    model = EvidentialGuardianNet(input_dim=encoder.input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f">>> Starting Training on {len(dataset)} samples for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        # === 进度条：训练 ===
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

        for b, p, target in train_pbar:
            # 将数据移动到 GPU
            b, p, target = b.to(device), p.to(device), target.to(device)

            optimizer.zero_grad()
            out = model(b, p)
            loss = edl_mse_loss(out, target, epoch, 2, 10)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # 实时显示 Loss
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # === 进度条：验证 ===
        model.eval()
        correct = 0
        total = 0
        # 验证过程通常很快，可以不显示详细进度条，或者简单显示
        with torch.no_grad():
            for b, p, target in val_loader:
                # 将数据移动到 GPU
                b, p, target = b.to(device), p.to(device), target.to(device)

                risk, _ = model.predict_uncertainty(b, p)
                pred = (risk > 0.5).long()
                truth = torch.argmax(target, dim=1)
                correct += (pred == truth).sum().item()
                total += truth.size(0)

        acc = correct / total if total > 0 else 0
        avg_loss = total_loss / len(train_loader)

        # 打印本轮总结
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"System 1 Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()
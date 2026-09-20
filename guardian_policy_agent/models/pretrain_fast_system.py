# pretrain_fast_system.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# 引入之前的模块
from guardian_policy_agent.models.vectorizer import SimpleFeatureEncoder
from guardian_policy_agent.models.edl_layers import EvidentialGuardianNet
from guardian_policy_agent.models.loss import edl_mse_loss
from tools.opp_loader import OPP115Dataset

# 配置
CSV_PATH = "data/raw/opp115/annotations_per_segment.csv" # 请确保路径正确
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001
SAVE_PATH = "checkpoints/sys1_opp_pretrained.pth"

def main():
    # 0. 检查数据文件
    if not os.path.exists(CSV_PATH):
        print(f"Error: OPP-115 dataset not found at {CSV_PATH}")
        print("Please download 'annotations_per_segment.csv' from usableprivacy.org")
        return

    # 1. 初始化组件
    encoder = SimpleFeatureEncoder()
    dataset = OPP115Dataset(CSV_PATH, encoder)

    # 划分训练集/验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型 (System 1)
    model = EvidentialGuardianNet(input_dim=encoder.input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Start training on {len(train_dataset)} samples from OPP-115...")

    # 3. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (b_vec, p_vec, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward
            logits = model(b_vec, p_vec)

            # EDL Loss
            # epoch 用于 KL 散度退火 (Annealing)
            loss = edl_mse_loss(logits, target, epoch, num_classes=2, annealing_step=10)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 4. 验证循环 (计算 Accuracy)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for b_vec, p_vec, target in val_loader:
                # 获取预测 (Risk Probability)
                risk, _ = model.predict_uncertainty(b_vec, p_vec) # [Batch]

                # Target: [1, 0] is Safe, [0, 1] is Risky
                # target_cls = 1 (Risky) if target[1] == 1 else 0
                target_cls = torch.argmax(target, dim=1)

                # Prediction: Risk > 0.5 -> 1
                pred_cls = (risk > 0.5).long()

                correct += (pred_cls == target_cls).sum().item()
                total += target_cls.size(0)

        val_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 5. 保存
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Pre-trained model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
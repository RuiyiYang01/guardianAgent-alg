# guardian_policy_agent/models/loss.py
import torch
import torch.nn.functional as F

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def kl_divergence(alpha, num_classes, device=None):
    """
    计算预测的 Dirichlet 分布与均匀 Dirichlet 分布之间的 KL 散度。
    用于作为正则项，防止模型在不知道答案时还盲目自信。
    """
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    return first_term + second_term

def loglikelihood_loss(y, alpha, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    return loglikelihood_err + loglikelihood_var

def edl_mse_loss(output, target, epoch_num, num_classes=2, annealing_step=10, device=None):
    """
    Evidential Deep Learning MSE Loss.
    :param output: 模型的输出 alpha
    :param target: One-hot 标签 [Batch, 2]
    :param epoch_num: 当前训练轮数 (用于 KL 退火)
    """
    if device is None:
        device = output.device

    alpha = output
    S = torch.sum(alpha, dim=1, keepdim=True)

    # 1. 准确性损失 (Expected Mean Square Error)
    # 也就是：Sum ( (y - alpha/S)^2 + Variance )
    belief = alpha / S
    A = torch.sum((target - belief) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    mse_loss = A + B

    # 2. KL 散度正则项 (KL Divergence Penalty)
    # 让那些“被预测错误”的样本的 Evidence 尽量归零，即回归均匀分布
    kl_alpha = (alpha - 1) * (1 - target) + 1
    kl = kl_divergence(kl_alpha, num_classes, device=device)

    # 3. 退火系数 (Annealing Coefficient)
    # 在训练初期不加 KL 惩罚，让模型先学特征；后期逐渐增加惩罚，防止过拟合
    annealing_coef = min(1.0, epoch_num / annealing_step)

    return torch.mean(mse_loss + annealing_coef * kl)
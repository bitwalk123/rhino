import torch
import torch.nn as nn
from torch.distributions import Categorical


def compute_ppo_loss(policy_net, value_net, obs, actions, old_log_probs, returns, advantages, action_masks):
    logits = policy_net(obs, action_masks)
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)

    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    values = value_net(obs).squeeze()
    value_loss = nn.functional.mse_loss(values, returns)

    return policy_loss + 0.5 * value_loss


def select_action(policy_net, obs, action_mask):
    logits = policy_net(obs, action_mask)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, obs, action_mask=None):
        logits = self.net(obs)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))
        return logits


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 出力はスカラー（状態価値）
        )

    def forward(self, obs):
        return self.net(obs)  # shape: [batch_size, 1]

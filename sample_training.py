import os

import torch
from torch import optim

from funcs.ios import get_excel_sheet
from funcs.models import get_trained_ppo_model_path
from modules.agent_mask import PolicyNetwork, ValueNetwork, compute_ppo_loss, select_action
from modules.env_mask import TradingEnv
from structs.res import AppRes

if __name__ == "__main__":
    res = AppRes()
    file_excel = "ticks_20250819.xlsx"
    code = "7011"
    path_excel = os.path.join(res.dir_collection, file_excel)
    df = get_excel_sheet(path_excel, code)
    env = TradingEnv(df)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy_net = PolicyNetwork(obs_dim, act_dim)
    value_net = ValueNetwork(obs_dim)
    optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=3e-4)

    num_epochs = 3
    gamma = 0.99

    for epoch in range(num_epochs):
        obs_list, action_list, logprob_list, reward_list, mask_list = [], [], [], [], []

        obs, info = env.reset()
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            mask_tensor = torch.tensor(info["action_mask"], dtype=torch.float32)
            action, log_prob = select_action(policy_net, obs_tensor, mask_tensor)

            obs_list.append(obs_tensor)
            action_list.append(torch.tensor(action))
            logprob_list.append(log_prob)
            mask_list.append(mask_tensor)

            obs, reward, done, _, info = env.step(action)
            reward_list.append(torch.tensor(reward, dtype=torch.float32))

        # Return と Advantage の計算
        returns = []
        G = 0
        for r in reversed(reward_list):
            G = r + gamma * G
            returns.insert(0, G)

        obs_batch = torch.stack(obs_list)
        action_batch = torch.stack(action_list)
        logprob_batch = torch.stack(logprob_list)
        return_batch = torch.stack(returns)
        value_batch = value_net(obs_batch).squeeze()
        adv_batch = return_batch - value_batch.detach()
        mask_batch = torch.stack(mask_list)

        # 学習
        loss = compute_ppo_loss(
            policy_net,
            value_net,
            obs_batch,
            action_batch,
            logprob_batch,
            return_batch,
            adv_batch,
            mask_batch
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    # 学習モデルの保存
    # https://docs.pytorch.org/docs/stable/generated/torch.save.html
    model_path = get_trained_ppo_model_path(res, code)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    obj = {
        "policy_state_dict": policy_net.state_dict(),
        "value_state_dict": value_net.state_dict()
    }
    torch.save(obj, model_path)
    print(f"✅ モデルを保存しました: {model_path}")

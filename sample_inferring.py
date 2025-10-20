import os

import torch

from funcs.ios import get_excel_sheet
from modules.agent_mask import PolicyNetwork, select_action
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
    obs = torch.tensor(env._get_observation(), dtype=torch.float32)
    mask = torch.tensor(env._get_action_mask(), dtype=torch.float32)
    action, log_prob = select_action(policy_net, obs, mask)
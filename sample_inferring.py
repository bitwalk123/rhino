import os

import pandas as pd
import torch
from funcs.ios import get_excel_sheet
from funcs.models import get_trained_ppo_model_path
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

    # モデルの読み込み
    model_path = get_trained_ppo_model_path(res, code)
    # モデルの状態を読み込み
    checkpoint = torch.load(model_path)
    policy_net = PolicyNetwork(obs_dim, act_dim)
    # ネットワークにパラメータを復元
    policy_net.load_state_dict(checkpoint["policy_state_dict"])
    # .eval() によって推論モードに切り替え（DropoutやBatchNormが無効化）
    policy_net.eval()

    # 推論ループ
    obs, info = env.reset()
    done = False
    while not done:
        # 状態とマスクをテンソル化（PyTorchネットワークに渡すため）
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        mask_tensor = torch.tensor(info["action_mask"], dtype=torch.float32)
        # マスク付きで行動分布を生成し、サンプリング
        # log_prob は推論では使わないが、ログや分析に活用可能
        action, log_prob = select_action(policy_net, obs_tensor, mask_tensor)

        """
        選択された行動がマスクで禁止されていないかを確認
        mask_tensor[action] == 0 の場合は違反行動（設計ミスやバグの検出に有効）
        select_action() 内部でマスクが正しく適用されていれば、通常は違反は起きない
        """
        if mask_tensor[action] == 0:
            print(f"⚠️ 違反行動: {action}, Mask: {mask_tensor.tolist()}")
        else:
            print(f"✅ 行動: {action}, Mask: {mask_tensor.tolist()}")

        obs, reward, done, _, info = env.step(action)

    df_transaction = pd.DataFrame(env.transman.dict_transaction)
    print(df_transaction)
    print(f"一株当りの損益 : {df_transaction['損益'].sum()} 円")

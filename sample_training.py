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

    """
    ネットワークとオプティマイザの初期化
    """
    # 行動分布を出力する方策ネットワーク
    policy_net = PolicyNetwork(obs_dim, act_dim)
    # 状態価値を推定するネットワーク
    # ValueNetwork は 学習時のAdvantage計算専用
    value_net = ValueNetwork(obs_dim)
    # 両ネットワークのパラメータを同時に更新
    optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=3e-4)

    num_epochs = 3
    gamma = 0.99

    # 学習ループ（PPO）
    for epoch in range(num_epochs):
        obs_list, action_list, logprob_list, reward_list, mask_list = [], [], [], [], []

        # 初期状態とマスク取得
        obs, info = env.reset()
        done = False

        while not done:
            # 環境から得られた観測データ obs を PyTorch のテンソルに変換
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            # 行動マスク（action_mask）をテンソルに変換
            mask_tensor = torch.tensor(info["action_mask"], dtype=torch.float32)
            # マスク付きで行動分布を生成し、サンプリング、log_prob はPPOの損失計算に必要
            action, log_prob = select_action(policy_net, obs_tensor, mask_tensor)

            """
            1エピソード分の履歴を後でバッチ化して、PPOの損失関数に渡す
            このようにするとエピソード全体を1つのテンソルバッチとして扱えるようになります。
            """
            # 現在の観測（状態）を保存
            # 後で torch.stack(obs_list) によってバッチ化され、方策・価値ネットワークの入力になる
            obs_list.append(obs_tensor)
            # 選択した行動を保存
            # torch.tensor(action) によってテンソル化し、後で損失計算に使用
            action_list.append(torch.tensor(action))
            # 選択した行動の対数確率を保存
            # log_prob は Categorical(...).log_prob(action) で得られた値
            logprob_list.append(log_prob)
            # 行動マスクを保存
            # PPOの損失計算で、マスクされた行動を除外するために使用
            mask_list.append(mask_tensor)

            # 状態遷移と報酬取得
            obs, reward, done, _, info = env.step(action)
            reward_list.append(torch.tensor(reward, dtype=torch.float32))

        """
        PPO（Proximal Policy Optimization）における「割引報酬（Return）」の計算処理
        エピソード全体の報酬履歴から、各時点での累積報酬（Return）を逆順で計算。
        
        PPOでは、状態の価値（value）と実際のReturnとの差分（Advantage）を使って学習
        𝐴_𝑡 = 𝐺_𝑡 − 𝑉(𝑠_𝑡)
        そのため、各ステップのReturn 𝐺_𝑡 を正確に計算しておく必要がある
        """
        # 強化学習では、ある時点 𝑡 における「Return」𝐺_𝑡 は、
        # その時点から将来にわたって得られる報酬の合計
        returns = []
        G = 0
        # 報酬リストを後ろから前へ処理
        for r in reversed(reward_list):
            # 現在の報酬 𝑟 に、次のステップの累積報酬 𝐺 を割引して加える
            # これにより、未来の報酬を考慮した累積値が得られる
            G = r + gamma * G
            # returns の先頭に 𝐺 を挿入することで、元の時間順に戻す
            returns.insert(0, G)

        """
        PPOの損失計算に向けた「バッチ化と前処理」
        """
        # 方策ネットワーク・価値ネットワークの入力
        obs_batch = torch.stack(obs_list)
        # 各ステップで選択した行動（整数）をテンソル化
        # PPOの損失関数では、行動ごとの確率や価値を比較するために必要
        action_batch = torch.stack(action_list)
        # 各行動の対数確率（log_prob）をまとめる
        # PPOでは「新しい方策と古い方策の確率比」を使って損失を計算するため、
        # 過去の確率を記録しておく必要がある
        logprob_batch = torch.stack(logprob_list)
        # 各ステップの割引報酬（Return）をまとめる
        # これは「実際に得られた報酬の累積」であり、
        # 価値ネットワークとの比較に使う
        return_batch = torch.stack(returns)
        # 状態ベクトル obs_batch を価値ネットワークに通して、各ステップの状態価値 𝑉(𝑠_𝑡) を取得
        # .squeeze() によって [T, 1] → [T] に変形（損失計算のため）
        #value_batch = value_net(obs_batch).squeeze()
        value_batch = value_net(obs_batch).squeeze(-1)  # 最後の次元だけを潰す
        # Advantage（利得）の計算
        # 𝐴_𝑡 = 𝐺_𝑡 − 𝑉(𝑠_𝑡)
        # detach() によって、価値ネットワークの勾配計算を切り離す。
        # （方策ネットワークだけを更新するため）
        adv_batch = return_batch - value_batch.detach()
        # 各ステップの行動マスクをまとめる
        # PPOの損失計算で、選択不可能な行動を除外するために使う
        mask_batch = torch.stack(mask_list)
        """
        PPOの損失関数を計算（方策・価値・エントロピー項を含む）
        """
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
        # 勾配をゼロに初期化
        # 前回の勾配情報を消去して、今回の学習ステップに備えるための必須操作
        optimizer.zero_grad()
        # 損失関数から勾配を計算
        loss.backward()
        # 勾配に基づいてパラメータを更新
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

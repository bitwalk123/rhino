import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Categorical

from modules.env_mask import TradingEnv


def compute_ppo_loss(
        policy_net,
        value_net,
        obs, actions,
        old_log_probs,
        returns,
        advantages,
        action_masks
):
    """
    📦 PPO損失関数（Clip付き）
    この関数は、以下の2つの損失を計算して合成します：
    - 方策損失（policy_loss）：確率比率のクリップ付き損失
    - 価値損失（value_loss）：状態価値とReturnの誤差（MSE）
    """
    # 方策ネットワークに状態とマスクを渡して、各行動のロジット（未正規化スコア）を取得
    logits = policy_net(obs, action_masks)
    # ロジットから 確率分布（Categorical） を構築
    dist = Categorical(logits=logits)
    # 現在の方策で、過去に選択された行動の対数確率を取得
    new_log_probs = dist.log_prob(actions)

    # PPOの中核：確率比率（新旧方策の確率の変化率）を計算
    ratio = torch.exp(new_log_probs - old_log_probs)
    # PPOの「Clip付き損失」のために、確率比率を上下に制限
    # 1 ± 0.2 は クリップ範囲（ε）：この範囲を超える更新は抑制される
    # これにより、方策の急激な変化を防ぎ、安定した学習が可能になる
    clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
    # PPOの「Surrogate Objective」
    # advantages は [T] のテンソルで、行動の良さを表す重み
    # torch.min(...) によって、クリップされた方策損失が選ばれる
    # - を付けることで、損失関数として最小化対象にする
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    # 状態ベクトル obs を価値ネットワークに通して、状態価値 𝑉(𝑠_𝑡) を取得
    # .squeeze() によって [T, 1] → [T] に変形（損失計算のため）
    values = value_net(obs).squeeze()
    # 状態価値と実際の Return の誤差を MSE（平均二乗誤差）で計算
    value_loss = nn.functional.mse_loss(values, returns)
    # 最終的な損失は、方策損失 + 価値損失（重み付き）
    # 0.5 はハイパーパラメータで、価値損失の影響度を調整
    # この合成損失を loss.backward() に渡して、両ネットワークを同時に更新
    return policy_loss + 0.5 * value_loss


def select_action(policy_net, obs, action_mask):
    """
    🚦 行動選択関数
    現在の状態 obs と行動マスク action_mask を使って、方策ネットワークから行動をサンプリングする
    PPOでは、確率的方策に基づいて行動を選び、その確率（log_prob）も記録する必要があります
    """
    # 方策ネットワークに状態とマスクを渡して、行動のロジット（未正規化スコア）を取得
    logits = policy_net(obs, action_mask)
    # Categorical 分布を使って、ロジットから確率分布を構築
    dist = Categorical(logits=logits)
    # 分布 dist から 確率的に行動をサンプリング
    action = dist.sample()
    # 選択した行動の対数確率（log_prob）を取得
    log_prob = dist.log_prob(action)
    # action.item() によって、テンソルからPythonの整数に変換
    # log_prob はそのままテンソルとして返す（後で loss.backward() に使うため）
    return action.item(), log_prob


class PolicyNetwork(nn.Module):
    # 🎯 方策ネットワーク（マスク対応）
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
    # 🧠 ValueNetwork の基本構造
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 出力はスカラー（状態価値）
        )

    def forward(self, obs):
        return self.net(obs)  # shape: [batch_size, 1]


class PPOAgent:
    def __init__(self):
        pass

    def infer(self, df: pd.DataFrame, model_path: str):
        env = TradingEnv(df)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        # モデルの状態を読み込み
        checkpoint = torch.load(model_path)
        policy_net = PolicyNetwork(obs_dim, act_dim)
        # ネットワークにパラメータを復元
        policy_net.load_state_dict(checkpoint["policy_state_dict"])
        # .eval() によって推論モードに切り替え（DropoutやBatchNormが無効化）
        policy_net.eval()

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 推論ループ
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        obs, info = env.reset()
        done = False
        while not done:
            # 状態とマスクをテンソル化（PyTorchネットワークに渡すため）
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            mask_tensor = torch.tensor(info["action_mask"], dtype=torch.float32)
            # マスク付きで行動分布を生成しサンプリング、log_prob は推論では使わないが、ログや分析に活用可能
            action, log_prob = select_action(policy_net, obs_tensor, mask_tensor)

            """
            選択された行動がマスクで禁止されていないかを確認
            mask_tensor[action] == 0 の場合は違反行動（設計ミスやバグの検出に有効）
            select_action() 内部でマスクが正しく適用されていれば、通常は違反は起きない
            """
            if mask_tensor[action] == 0:
                print(f"❌ 違反行動: {action}, Mask: {mask_tensor.tolist()}")
            """
            else:
                print(f"✔ 行動: {action}, Mask: {mask_tensor.tolist()}")
            """

            obs, reward, done, _, info = env.step(action)

        # 取引明細の出力
        df_transaction = pd.DataFrame(env.transman.dict_transaction)
        print(df_transaction)
        print(f"一株当りの損益 : {df_transaction['損益'].sum()} 円")

    def train(self, df: pd.DataFrame, model_path: str):
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

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 学習ループ
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
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
                obs_list.append(obs_tensor)
                # 選択した行動を保存
                action_list.append(torch.tensor(action))
                # 選択した行動の対数確率を保存
                logprob_list.append(log_prob)
                # 行動マスクを保存
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
            # ある時点 𝑡 における「Return」𝐺_𝑡 は、その時点から将来にわたって得られる報酬の合計
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
            action_batch = torch.stack(action_list)
            # 各行動の対数確率（log_prob）をまとめる
            logprob_batch = torch.stack(logprob_list)
            # 各ステップの割引報酬（Return）をまとめる
            return_batch = torch.stack(returns)
            # 状態ベクトル obs_batch を価値ネットワークに通して、各ステップの状態価値 𝑉(𝑠_𝑡) を取得
            value_batch = value_net(obs_batch).squeeze(-1)  # 最後の次元だけを潰す
            # Advantage（利得）の計算
            adv_batch = return_batch - value_batch.detach()
            # 各ステップの行動マスクをまとめる
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
            optimizer.zero_grad()
            # 損失関数から勾配を計算
            loss.backward()
            # 勾配に基づいてパラメータを更新
            optimizer.step()

            print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

        # 学習モデルの保存
        # https://docs.pytorch.org/docs/stable/generated/torch.save.html
        obj = {
            "policy_state_dict": policy_net.state_dict(),
            "value_state_dict": value_net.state_dict()
        }
        torch.save(obj, model_path)
        print(f"✅ モデルを保存しました: {model_path}")

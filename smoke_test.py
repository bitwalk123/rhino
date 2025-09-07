import pandas as pd
import numpy as np

# TradingEnv クラスはすでに定義済みとする
from trading_env import TradingEnv, ActionType


# 1. ダミーデータの作成（Time, Price, Volume）
def create_dummy_data(n: int):
    np.random.seed(42)
    time = np.arange(n)
    price = np.linspace(1100, 1200, n) - np.random.normal(0, 5, n)
    price = [float(int(p)) for p in price]
    volume = np.cumsum(np.random.randint(1, 500, n))
    volume = [v * 100 for v in volume]
    df = pd.DataFrame({"Time": time, "Price": price, "Volume": volume})
    return df


# 2. 環境の初期化と動作確認
def run_smoke_test():
    actions = [
        ActionType.HOLD,
        ActionType.BUY,
        ActionType.HOLD,
        ActionType.HOLD,
        ActionType.BUY,
        ActionType.SELL,
        ActionType.BUY,
        ActionType.HOLD,
        ActionType.REPAY,
        ActionType.SELL,
        ActionType.HOLD,
        ActionType.SELL,
        ActionType.HOLD,
        ActionType.BUY,
        ActionType.SELL,
        ActionType.HOLD,
        ActionType.REPAY,
    ]
    df = create_dummy_data(len(actions))
    print(df)
    env = TradingEnv(df)
    obs, _ = env.reset()
    print("\n📈 初期観測:", obs)

    for step, act in enumerate(actions):
        obs, reward, done, _, dict_info = env.step(act.value)
        # 確認しやすいように整形して標準出力
        print(
            f"🕒 Step {step:02d}: "
            f"Obs=[{obs[0]:.1f}, {obs[1]:5.2f}, "
            f"{int(obs[2]):d}, {int(obs[3]):d}, {int(obs[4]):d}], "
            f"Action={ActionType(act).name}, "
            f"Reward={reward:.2f}, "
            f"Done={done}"
        )

        if done:
            print(f"総収益: {dict_info["pnl_total"]}")
            break


# 実行
if __name__ == "__main__":
    run_smoke_test()

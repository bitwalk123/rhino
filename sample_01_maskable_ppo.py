import os

from sb3_contrib import MaskablePPO

from funcs.ios import get_excel_sheet
from modules.env import TrainingEnv
from structs.res import AppRes

if __name__ == "__main__":
    # 過去のティックデータをサンプルとして読み込む処理
    res = AppRes()
    file = "ticks_20250819.xlsx"
    code = "7011"
    path_excel = os.path.join(res.dir_collection, file)  # フルパスを生成
    df = get_excel_sheet(path_excel, code)  # 銘柄コードがシート名
    print("ヘッダー")
    print(df.columns)
    print(f"行数 : {len(df)}")

    """
    TrainingEnv
    gymnasium.Env を継承した、過去のティックデータを用いた学習用環境クラス
    MaskablePPO に対応させるため、マスク情報を返す action_masks()　を実装
    """
    env = TrainingEnv(df)

    # =========
    #  学習処理
    # =========
    # 新しいモデルを生成
    model = MaskablePPO("MlpPolicy", env, verbose=1)
    print("学習を開始します。")
    model.learn(total_timesteps=100_000)
    print("学習が終了しました。")

    print("モデルを保存します。")
    model.save("ppo_mask")
    del model  # remove to demonstrate saving and loading

    # =========
    #  推論処理
    # =========
    print("モデルを読み込みます。")
    model = MaskablePPO.load("ppo_mask")

    print("推論を開始します。")
    obs, _ = env.reset()
    terminated = False
    while not terminated:
        action_masks = env.action_masks()
        action, _states = model.predict(obs, action_masks=action_masks)
        obs, reward, terminated, truncated, info = env.step(action)
    print("推論が終了しました。")

    # 取引履歴を所得
    print("取引詳細")
    df_transaction = env.getTransaction()
    print(df_transaction)
    print(f"一株当りの損益 : {df_transaction['損益'].sum()} 円")

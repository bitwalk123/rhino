import os

from funcs.ios import get_excel_sheet
from funcs.models import get_trained_ppo_model_path
from modules.agent_mask import PPOAgent
from structs.res import AppRes

if __name__ == "__main__":
    res = AppRes()
    # 学習用データフレーム
    file_excel = "ticks_20250819.xlsx"
    code = "7011"
    path_excel = os.path.join(res.dir_collection, file_excel)
    df = get_excel_sheet(path_excel, code)

    # モデルの保存先
    model_path = get_trained_ppo_model_path(res, code)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # PPO エージェントのインスタンスと学習
    agent = PPOAgent()
    agent.train(df, model_path, num_epochs=10, new_model=False)

import os

from funcs.ios import get_excel_sheet
from funcs.models import get_trained_ppo_model_path
from modules.agent_mask import PPOAgent
from structs.res import AppRes

if __name__ == "__main__":
    res = AppRes()
    file_excel = "ticks_20250819.xlsx"
    code = "7011"
    path_excel = os.path.join(res.dir_collection, file_excel)
    df = get_excel_sheet(path_excel, code)

    # モデル保存先のパス
    model_path = get_trained_ppo_model_path(res, code)

    # PPO エージェントのインスタンスと推論
    agent = PPOAgent()
    agent.infer(df, model_path)

import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from funcs.ios import get_excel_sheet
from modules.env import TrainingEnv
from structs.res import AppRes

res = AppRes()
file = "ticks_20250819.xlsx"
code = "7011"
path_excel = os.path.join(res.dir_collection, file)  # フルパスを生成
df = get_excel_sheet(path_excel, code)  # 銘柄コードがシート名

env_raw = TrainingEnv(df)  # gymnasium.Env を継承した自作の環境
env = ActionMasker(env_raw, lambda obs: env_raw.getActionMask())

model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

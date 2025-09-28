import os

from stable_baselines3 import PPO

from funcs.ios import get_excel_sheet
from modules.env import TradingEnv
from structs.res import AppRes


class PPOAgent:
    def __init__(self, res: AppRes):
        self.res = res
        self.total_timesteps = 128000

    def train(self, file:str, code:str):
        path_excel = os.path.join(self.res.dir_collection, file)
        df = get_excel_sheet(path_excel, code)
        env = TradingEnv(df)
        model = PPO("MlpPolicy", env, verbose=True)
        model.learn(total_timesteps=self.total_timesteps)

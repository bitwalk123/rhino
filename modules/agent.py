from stable_baselines3 import PPO

from modules.env import TradingEnv


class PPOAgent:
    def __init__(self, env: TradingEnv):
        self.model = PPO("MlpPolicy", env, verbose=True)
        self.total_timesteps = 128000

    def train(self):
        self.model.learn(total_timesteps=self.total_timesteps)

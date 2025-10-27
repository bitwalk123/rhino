import os

from modules.agent import PPOAgentSB3
from structs.res import AppRes

if __name__ == "__main__":
    res = AppRes()
    agent = PPOAgentSB3(res)

    # 学習用データ
    code = "7011"
    # list_file = ["ticks_20250819.xlsx"]
    list_file = sorted(os.listdir(res.dir_collection))
    for file in list_file:
        print(file)
        agent.train(file, code)

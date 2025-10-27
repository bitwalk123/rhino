from modules.agent import PPOAgentSB3
from structs.res import AppRes

if __name__ == "__main__":
    res = AppRes()
    agent = PPOAgentSB3(res)

    # 学習用データ
    file = "ticks_20250819.xlsx"
    code = "7011"

    agent.infer(file, code)
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
    flag_new_model = True
    for file in list_file:
        print(f"学習するティックデータ : {file}")
        agent.train(file, code, new_model=flag_new_model)
        if flag_new_model:
            flag_new_model = False

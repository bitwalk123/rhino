from funcs.commons import get_collection_path
from funcs.ios import get_excel_sheet
from funcs.models import get_ppo_model_path
from modules.agent import PPOAgentSB3
from structs.res import AppRes

if __name__ == "__main__":
    res = AppRes()
    agent = PPOAgentSB3()

    # 学習用データ
    code = "7011"
    list_file = ["ticks_20250819.xlsx"]
    # list_file = sorted(os.listdir(res.dir_collection))
    flag_new_model = True
    for file in list_file:
        print(f"学習するティックデータ : {file}")
        # Excel ファイルのフルパス
        path_excel = get_collection_path(res, file)
        # Excel ファイルをデータフレームに読み込む
        df = get_excel_sheet(path_excel, code)
        path_model = get_ppo_model_path(res, code)
        # モデルの学習
        agent.train(df, path_model, new_model=flag_new_model)
        if flag_new_model:
            flag_new_model = False

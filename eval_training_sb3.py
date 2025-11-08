import datetime
import os
import re

from funcs.commons import get_collection_path
from funcs.ios import get_excel_sheet
from funcs.models import get_ppo_model_path
from modules.agent import PPOAgentSB3
from structs.res import AppRes

if __name__ == "__main__":
    pattern = re.compile(r".+(\d{8})\.xlsx")

    res = AppRes()
    agent = PPOAgentSB3()

    dt = datetime.datetime.now()
    date_str = f"{dt.year:04d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"

    # 学習用データ
    code = "7011"
    # list_file = ["ticks_20250819.xlsx"]
    list_file = sorted(os.listdir(res.dir_collection))
    flag_new_model = True
    for file in list_file:
        print(f"学習するティックデータ : {file}")
        # Excel ファイルのフルパス
        path_excel = get_collection_path(res, file)
        # Excel ファイルをデータフレームに読み込む
        df = get_excel_sheet(path_excel, code)
        path_model = get_ppo_model_path(res, code)
        # ログ出力先ディレクトリ（例: logs/ticks_20250819）
        if m := pattern.match(file):
            file_date_str = m.group(1)
        else:
            file_date_str = "unknown"
        log_dir = os.path.join(res.dir_log, code, date_str, file_date_str)
        # モデルの学習
        agent.train(df, path_model, log_dir, new_model=flag_new_model)
        if flag_new_model:
            flag_new_model = False

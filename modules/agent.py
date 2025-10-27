from stable_baselines3.common.env_checker import check_env

from funcs.commons import get_collection_path
from funcs.ios import get_excel_sheet
from modules.agent_auxiliary import ActionMaskWrapper
from modules.env import TrainingEnv
from structs.res import AppRes


class PPOAgentSB3:
    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

    def get_env(self, file: str, code: str) -> DummyVecEnv:
        # Excel ファイルをフルパスに
        path_excel = get_collection_path(self.res, file)
        # Excel ファイルをデータフレームに読み込む
        df = get_excel_sheet(path_excel, code)

        # 環境のインスタンスを生成
        env_raw = TrainingEnv(df)
        env = ActionMaskWrapper(env_raw)
        # SB3の環境チェック（オプション）
        check_env(env, warn=True)

        env_monitor = Monitor(env, self.res.dir_log)  # Monitorの利用
        env_vec = DummyVecEnv([lambda: env_monitor])

        return env_vec

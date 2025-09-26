import pandas as pd

from structs.res import AppRes


def get_jpx_ticker_list(self, res: AppRes) -> pd.DataFrame:
    return pd.read_excel(res.tse)

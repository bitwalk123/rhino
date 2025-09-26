import os

import pandas as pd


def get_excel_sheet_list(path_excel: str) -> list:
    if os.path.isfile(path_excel):
        wb = pd.ExcelFile(path_excel)
        list_name: list = wb.sheet_names
        if len(list_name) > 0:
            return sorted(list_name)
        else:
            return list()
    else:
        return list()


def get_excel_sheet(path_excel: str, sheet: str) -> pd.DataFrame:
    """
    指定したExcelファイルの指定したシートをデータフレームに読み込む
    :param path_excel:
    :param sheet:
    :return:
    """
    if os.path.isfile(path_excel):
        wb = pd.ExcelFile(path_excel)
        list_sheet = wb.sheet_names
        if sheet in list_sheet:
            return wb.parse(sheet_name=sheet)
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

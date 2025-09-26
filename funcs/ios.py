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

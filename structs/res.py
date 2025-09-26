import os


class AppRes:
    dir_collection = 'collection'
    dir_font = 'fonts'
    dir_image = 'images'
    dir_model = 'models'
    dir_output = 'output'

    tse = 'https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls'

    path_monospace = "fonts/RictyDiminished-Regular.ttf"

    def __init__(self):
        # システムディレクトリのチェック
        list_dir = [
            self.dir_collection,
            self.dir_model,
            self.dir_output,
        ]
        for dirname in list_dir:
            self.check_system_dir(dirname)

    @staticmethod
    def check_system_dir(dirname: str):
        if not os.path.exists(dirname):
            os.mkdir(dirname)



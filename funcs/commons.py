import datetime
import os
import re

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QStyle

from structs.res import AppRes


def get_builtin_icon(parent: QWidget, name: str) -> QIcon:
    pixmap_icon = getattr(QStyle.StandardPixmap, 'SP_%s' % name)
    return parent.style().standardIcon(pixmap_icon)


def get_date_str_from_filename(path: str) -> str:
    p = re.compile(r".+(\d{4})(\d{2})(\d{2})\..*")
    m = p.match(path)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    else:
        return "1970-01-01"


def get_icon(res: AppRes, img_name: str) -> QIcon:
    return QIcon(os.path.join(res.dir_image, img_name))


def get_name_15min_chart(code: str, dt: datetime.datetime) -> str:
    year = dt.year
    month = dt.month
    day = dt.day
    return f"{year:4d}/{month:02d}{day:02d}_15min_chart_{code}.png"


def get_name_15min_chart_now(code: str) -> str:
    dt_now = datetime.datetime.now()
    year = dt_now.year
    month = dt_now.month
    day = dt_now.day
    return f"{year:4d}/{month:02d}{day:02d}_15min_chart_{code}.png"


def get_name_15min_chart_usd(code: str, dt: datetime.datetime) -> str:
    year = dt.year
    month = dt.month
    day = dt.day
    return f"{year:4d}/{month:02d}{day:02d}_15min_chart_{code}_usd.png"

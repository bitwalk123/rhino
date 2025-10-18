from enum import Enum, auto


class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
    REPAY = 3


class AppMode(Enum):
    TRAIN = auto()
    INFER = auto()


class PositionType(Enum):
    NONE = 0
    LONG = 1
    SHORT = 2

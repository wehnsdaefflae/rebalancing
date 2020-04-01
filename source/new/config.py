
PATH_DIRECTORY_DATA = "C:/Users/Mark/PycharmProjects/rebalancing/data/"
RAW_BINANCE_DIR = PATH_DIRECTORY_DATA + "binance/"


STATS_NO_TIME = (
    # "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    # "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
)


STATS_TYPES = int, float, float, float, float, float, int, float, int, float, float, float

STAT_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
)

stat_empty = -1, -1., -1., -1., -1., -1., -1, -1., -1, -1., -1., -1.

indices_ints = 0, 6, 8


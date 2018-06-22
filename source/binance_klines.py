from binance.client import Client


# https://sammchardy.github.io/binance/2018/01/08/historical-data-download-binance.html
"""
 [
    1499040000000,      # Open time
    "0.01634790",       # Open
    "0.80000000",       # High
    "0.01575800",       # Low
    "0.01577100",       # Close
    "148976.11427815",  # Volume
    1499644799999,      # Close time
    "2434.19055334",    # Quote asset volume
    308,                # Number of trades
    "1756.87402397",    # Taker buy base asset volume
    "28.46694368",      # Taker buy quote asset volume
    "17928899.62484339" # Ignore
  ]
 """

# pair = "ETHBTC"

client = Client("", "")
b = client.get_exchange_info()

pairs = sorted(set(y for y in [x["symbol"] for x in b["symbols"]] if y[-3:] == "ETH"))

for pair_index, each_pair in enumerate(pairs):
    print("getting {:s} ({:d}/{:d})".format(each_pair, pair_index + 1, len(pairs)))
    # klines = client.get_historical_klines(each_pair, Client.KLINE_INTERVAL_1MINUTE, "20 Jun, 2017", "20 Jun, 2018")
    try:
        klines = client.get_historical_klines(each_pair, Client.KLINE_INTERVAL_1MINUTE, "20 May, 2018", "20 Jun, 2018")
        with open("../data/binance/{}.csv".format(each_pair), mode='w') as f:
            for each_kline in klines:
                f.write("\t".join([str(x) for x in each_kline]) + "\n")
    except Exception as e:
        print("{} didn't work.".format(each_pair))
        print(e)

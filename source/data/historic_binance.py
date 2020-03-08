import os
import time

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

# start_date = "01 Jan, 2010"
start_date = "01 May, 2018"
end_date = "01 Jan, 2020"
interval = Client.KLINE_INTERVAL_1MINUTE

encoded_dates = start_date.replace(" ", "").replace(",", "") + "-" + end_date.replace(" ", "").replace(",", "")
data_dir = "../../data/binance/" + encoded_dates + "-" + interval + "/"
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

client = Client("", "")
while True:
    while True:
        try:
            b = client.get_exchange_info()
            break
        except Exception as e:
            print("Cannot connect, retrying...")
            time.sleep(1)

    exchange_pairs = {y for y in [x["symbol"] for x in b["symbols"]] if y[-3:] == "ETH"}

    present_pairs = {os.path.splitext(f)[0] for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))}

    missing_pairs = exchange_pairs - present_pairs
    if len(missing_pairs) < 1:
        print("all pairs retrieved")
        exit()

    print("{:d} pairs missing".format(len(missing_pairs)))
    for pair_index, each_pair in enumerate(sorted(missing_pairs)):
        print("getting {:s} ({:d}/{:d})".format(each_pair, pair_index + 1, len(missing_pairs)))
        try:
            klines = client.get_historical_klines(each_pair, interval, start_date, end_date)
            with open(data_dir + "{}.csv".format(each_pair), mode='w') as f:
                for i, each_kline in enumerate(klines):
                    f.write("\t".join([str(x) for x in each_kline]) + "\n")
                    if i % 100 == 99:
                        print(f"finished {i * 100 / len(klines):5.2f}% of klines...")
        except Exception as e:
            print("{} didn't work, continuing.".format(each_pair))
            print(e)
            time.sleep(1)

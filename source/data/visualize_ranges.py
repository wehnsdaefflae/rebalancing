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
import glob
import os
from typing import Sequence, Tuple

from matplotlib import pyplot


def get_timestamps(filename: str) -> Sequence[int]:
    print(f"reading {filename:s}...")
    sequence = []
    with open(filename, mode="r") as file:
        for line in file:
            ts_str = line.split("\t", 1)[0]
            sequence.append(int(ts_str))

    return sequence


def check_sequence(sequence: Sequence[int]) -> Tuple[int, int]:
    d = -1
    for i in range(1, len(sequence)):
        delta = sequence[i] - sequence[i-1]
        if d < 0:
            d = delta
        elif delta != d:
            return delta, i
    return -1, -1


def main():
    directory = "../../data/binance/"
    files = sorted(os.path.basename(_name) for _name in glob.glob(directory + "*.csv"))

    files_fine = [
        "ADAETH.csv",
        "ADXETH.csv",
        "AMBETH.csv",
        "ARKETH.csv",
        "ARNETH.csv",
        "ASTETH.csv",
        "BATETH.csv",
        "BCCETH.csv",
        "BCDETH.csv",
        "BCPTETH.csv",
        "BQXETH.csv",
        "BTGETH.csv",
        "BTSETH.csv",
        "CDTETH.csv",
        "DASHETH.csv",
        "DGDETH.csv",
        "DLTETH.csv",
        "ENGETH.csv",
        "ENJETH.csv",
        "ETCETH.csv",
        "EVXETH.csv",
        "FUELETH.csv",
        "FUNETH.csv",
        "GVTETH.csv",
        "GXSETH.csv",
        "HSRETH.csv",
        "IOTAETH.csv",
        "KMDETH.csv",
        "KNCETH.csv",
        "LINKETH.csv",
        "LSKETH.csv",
        "MANAETH.csv",
        "MDAETH.csv",
        "MODETH.csv",
        "MTHETH.csv",
        "MTLETH.csv",
        "NEOETH.csv",
        "NULSETH.csv",
        "OMGETH.csv",
        "POEETH.csv",
        "POWRETH.csv",
        "PPTETH.csv",
        "QSPETH.csv",
        "RCNETH.csv",
        "RDNETH.csv",
        "REQETH.csv",
        "SALTETH.csv",
        "SNGLSETH.csv",
        "SNMETH.csv",
        "STORJETH.csv",
        "STRATETH.csv",
        "SUBETH.csv",
        "TNTETH.csv",
        "TRXETH.csv",
        "VENETH.csv",
        "VIBETH.csv",
        "XMRETH.csv",
        "XRPETH.csv",
        "XVGETH.csv",
        "XZCETH.csv",
        "YOYOETH.csv",
        "ZECETH.csv",
        "ZRXETH.csv",
    ]
    files_fine = []
    for name in files:
        sequence = get_timestamps(directory + name)
        gap_irregular, index = check_sequence(sequence)
        if 60000 < gap_irregular:
            print(f"Sequence {name:s} has a hole of size {gap_irregular:d} at index {index:d}. Skipping...")
            continue
        else:
            print(f"Sequence {name:s} is fine.")
            files_fine.append(name)

    print(", ".join(files_fine))

        # pyplot.scatter(seq, [i] * len(seq), label=name)

    pyplot.show()


if __name__ == "__main__":
    main()

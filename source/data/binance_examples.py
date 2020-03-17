import glob
import os
from typing import Tuple, Sequence

from source.tactics.optimal_trading import split_time_and_data, make_source_matrix, make_path_from_sourcematrix, get_crypto_rates
from source.tools.timer import Timer


def get_pairs() -> Sequence[Tuple[str, str]]:
    pairs = (
        ("bcc", "eth"), ("bnb", "eth"), ("dash", "eth"), ("icx", "eth"),
        ("iota", "eth"), ("ltc", "eth"), ("nano", "eth"), ("poa", "eth"),
        ("qtum", "eth"), ("theta", "eth"), ("tusd", "eth"), ("xmr", "eth")
    )

    directory = "../../data/binance/"
    files = sorted(glob.glob(directory + "*.csv"))

    names_base = (os.path.basename(x) for x in files)
    names_first = (os.path.splitext(x)[0] for x in names_base)
    pairs = tuple((x[:-3], x[-3:]) for x in names_first)
    return pairs


def main():
    pairs = get_pairs()[:2]

    stats = (
        ## "open_time",
        #"open",
        #"high",
        #"low",
        "close",
        #"volume",
        ## "close_time",
        #"quote_asset_volume",
        #"number_of_trades",
        #"taker_buy_base_asset_volume",
        #"taker_buy_quote_asset_volume",
        #"ignore",
    )

    names_pairs = tuple(f"{x[0]:s}-{x[1]}" for x in pairs)

    timestamps = []
    no_assets = len(pairs)
    generate_rates_a = (split_time_and_data(x, timestamp_storage=timestamps) for x in get_crypto_rates(pairs, ("close", ), interval_minutes=1))

    matrix = make_source_matrix(no_assets, generate_rates_a, fees=.01)
    print("fixing matrix...")
    matrix_fix = tuple(matrix)
    path = make_path_from_sourcematrix(matrix_fix)

    generate_rates_b = (split_time_and_data(x) for x in get_crypto_rates(pairs, stats, interval_minutes=1))

    print("writing examples...")
    header = ("timestamp",) + tuple(f"{each_pair:s}_{each_column:s}" for each_pair in names_pairs for each_column in stats) + ("target",)  # todo: reward at some point?
    with open("../../data/examples/binance.csv", mode="a") as file:
        file.write("\t".join(header) + "\n")

        for i, (ts, rates, target) in enumerate(zip(timestamps, generate_rates_b, path)):
            line = [f"{ts:d}"] + [f"{x:.8f}" for x in rates] + [names_pairs[target]]
            file.write("\t".join(line) + "\n")

            if Timer.time_passed(2000):
                print(f"finished {i * 100. / len(timestamps):5.2f}% of writing examples...")


if __name__ == "__main__":
    main()

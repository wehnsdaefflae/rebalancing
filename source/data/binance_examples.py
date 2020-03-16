from source.tactics.optimal_trading import get_selected_crypto_rates, split_time_and_data, make_source_matrix, make_path_from_sourcematrix, simulate, get_crypto_rates
from source.tools.timer import Timer


def main():
    pairs = (
        ("bcc", "eth"), ("bnb", "eth"), ("dash", "eth"), ("icx", "eth"),
        ("iota", "eth"), ("ltc", "eth"), ("nano", "eth"), ("poa", "eth"),
        ("qtum", "eth"), ("theta", "eth"), ("tusd", "eth"), ("xmr", "eth")
    )

    names_pairs = tuple(f"{x[0]:s}-{x[1]}" for x in pairs)

    timestamps = []
    no_assets = 12
    fees = .01
    generate_rates_a = (split_time_and_data(x, timestamp_storage=timestamps) for x in get_crypto_rates(interval_minutes=1, pairs=pairs))

    matrix = make_source_matrix(no_assets, generate_rates_a, fees=.01)
    print("fixing matrix...")
    matrix_fix = tuple(matrix)

    path = make_path_from_sourcematrix(matrix_fix)

    generate_rates_b = (split_time_and_data(x) for x in get_selected_crypto_rates())
    roi_path = simulate(generate_rates_b, path, fees)

    generate_rates_c = (split_time_and_data(x) for x in get_selected_crypto_rates())

    print("writing examples...")
    with open("../../data/examples/binance.csv", mode="a") as file:
        header = ("timestamp",) + names_pairs + ("target", "gain")
        file.write("\t".join(header) + "\n")
        next(roi_path)  # skip the first gain
        for i, (ts, rates, target, gain) in enumerate(zip(timestamps, generate_rates_c, path, roi_path)):
            line = [f"{ts:d}"] + [f"{x:.8f}" for x in rates] + [names_pairs[target], f"{gain:.8f}"]
            file.write("\t".join(line) + "\n")

            if Timer.time_passed(2000):
                print(f"finished {i * 100. / len(timestamps):5.2f}% of writing examples...")


if __name__ == "__main__":
    main()

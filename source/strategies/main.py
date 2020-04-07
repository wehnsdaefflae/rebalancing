from typing import Tuple, Sequence, Optional, List, Generator

from source.data.generators.snapshots_debug import get_random_rates, get_debug_rates
from source.strategies.infer_investment_path.optimal_trading_memory import generate_multiple_changes, generate_matrix, make_path_from_sourcematrix
from source.strategies.simulations.simulation import simulate


def split_time_and_data(
        input_data: Tuple[int, Sequence[float]],
        timestamp_storage: Optional[List[int]] = None,
        rate_storage: Optional[List[Sequence[float]]] = None) -> Sequence[float]:

    timestamp, data = input_data

    if timestamp_storage is not None:
        timestamp_storage.append(timestamp)

    if rate_storage is not None:
        rate_storage.append(data)

    return data


def compare():
    no_assets = 3
    fees = .01

    # source
    timestamps = []
    rates = []
    generate_rates = get_debug_rates()

    print("tick    " + "".join(f"{i: 9d}" for i in range(len(rates))))
    print()
    for i, each_rate in enumerate(zip(*rates)):
        print(f"ass_{i:03d} " + "".join(f"{x:9.2f}" for x in each_rate))
    print()

    matrix_change = generate_multiple_changes(generate_rates)
    matrix_change_fix = list(matrix_change)
    print("change matrix source")
    print("\n".join(["    ".join(["        "] + [f"{max(0., v):5.2f}" for v in x]) for x in zip(*[y for y in matrix_change_fix])]))
    print()

    matrix = generate_matrix(no_assets, matrix_change_fix, .01, bound=100)
    matrix_fix = tuple(matrix)

    print("asset deprecated matrix source")
    print("\n".join(["".join(["     "] + [f"{v: 9d}" for v in x]) for x in zip(*[y[0] for y in matrix_fix])]))
    print()
    print(f"roi: {matrix_fix[-1][1][1]:5.5f}")
    print()

    path_new = make_path_from_sourcematrix(matrix_fix)

    roi_path = simulate(rates, path_new, fees)
    next(roi_path)
    print("path    " + "".join(f"  ass_{x:03d}" for x in path_new))
    print("reward  " + "".join(f"{x:9.2f}" for x in roi_path))
    print()


def stack():
    import random
    from typing import Iterator

    def get_random_sequence(start_value: float) -> Iterator[float]:
        value = start_value
        yield value

        while True:
            r = random.uniform(-.1, .1) * value
            value = value + r
            yield value

    g = get_random_sequence(1.)
    a = [next(g) for _ in range(5)]
    print(a)

    def get_market(no_assets: int = 10) -> Iterator[Sequence[float]]:
        rg = tuple(get_random_sequence(random.uniform(10., 60.)) for _ in range(no_assets))
        yield from zip(*rg)

    gm = get_market(2)
    b = [next(gm) for _ in range(5)]
    print(b)

    def ratio_generator() -> Generator[float, Optional[float], None]:
        value_last = yield
        value = yield
        while True:
            ratio = 0. if value_last == 0. else value / value_last
            value_last = value
            value = yield ratio

    gr = ratio_generator()
    next(gr)    # move to the first yield
    g = get_random_sequence(1.)
    a = []
    for _v in g:
        _r = gr.send(_v)
        if _r is None:
            # two values are required for a ratio
            continue
        a.append(_r)
        if len(a) >= 5:
            break
    print(a)

    def ratio_generator_multiple(no_values: int) -> Generator[Sequence[float], Optional[Sequence[float]], None]:
        gs = tuple(ratio_generator() for _ in range(no_values))
        for each_g in gs:
            next(each_g)

        values = yield
        ratios = tuple(g.send(v) for g, v in zip(gs, values))

        yield from zip(*(g.send(v) for g, v in zip(gs, values))) #?
        #while True:
        #    values = yield None if None in ratios else ratios
        #    ratios = tuple(g.send(v) for g, v in zip(gs, values))

    rgm = ratio_generator_multiple(2)
    next(rgm)    # move to the first yield
    gm = get_market(2)
    b = []
    for _v in gm:
        _r = rgm.send(_v)
        if _r is None:
            # two values are required for a ratio
            continue
        b.append(_r)
        if len(b) >= 5:
            break
    print(b)

"""
def f():
	yield from g()
def g():
	x = yield 1
	y = yield x
	yield y
 
gen = f()
print(gen.send(None))
print(gen.send(2))
print(gen.send(3))
"""

def main():
    no_assets = 2

    rates = get_debug_rates()
    ratios = generate_multiple_changes(rates)
    matrix = generate_matrix(no_assets, ratios, .01, bound=100)
    path = make_path_from_sourcematrix(list(matrix))
    print(path)


if __name__ == "__main__":
    # compare()
    main()
    # stack()

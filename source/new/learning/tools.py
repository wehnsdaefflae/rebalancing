from __future__ import annotations

import itertools
import json
from typing import TypeVar, Sequence, Generic, Dict, Any, Iterable, Tuple, Generator, Optional

from matplotlib import pyplot


T = TypeVar("T")


def z_score_generator(drag: int = -1, offset: float = 0., scale: float = 1., clamp: Optional[Tuple[float, float]] = None) -> Generator[float, float, None]:
    # use to normalize input, enables more precise error value calculation for recurrent and failure approximations
    iteration = 0

    value = yield
    average = value
    deviation = 0.

    if clamp is not None:
        assert clamp[0] < clamp[1]

    while True:
        if deviation == 0.:
            value = yield 0.

        elif clamp is None:
            value = yield ((value - average) / deviation) * scale + offset

        else:
            r = ((value - average) / deviation) * scale + offset
            value = yield max(clamp[0], min(clamp[1], r))

        d = drag if drag >= 0 else iteration
        average = smear(average, value, d)
        deviation = smear(deviation, abs(value - average), d)

        iteration += 1


def z_score_normalized_generator() -> Generator[float, float, None]:
    yield from z_score_generator(drag=-1, scale=.25, offset=.5, clamp=(0., 1.))


def z_score_multiple_normalized_generator(no_values: int) -> Generator[Sequence[float], Sequence[float], None]:
    gs = tuple(z_score_normalized_generator() for _ in range(no_values))
    values = yield tuple(next(each_g) for each_g in gs)

    while True:
        values = yield tuple(each_g.send(x) for each_g, x in zip(gs, values))


def ratio_generator() -> Generator[float, Optional[float], None]:
    value_last = yield  # dont do an initial next?
    value = yield
    while True:
        ratio = value / value_last
        value_last = value
        value = yield ratio


def ratio_generator_multiple(no_values: int) -> Generator[Sequence[float], Optional[Sequence[float]], None]:
    gs = tuple(ratio_generator() for _ in range(no_values))
    for each_g in gs:
        next(each_g)

    values = yield
    ratios = tuple(g.send(v) for g, v in zip(gs, values))

    while True:
        values = yield None if None in ratios else ratios
        ratios = tuple(g.send(v) for g, v in zip(gs, values))


def smear(average: float, value: float, inertia: int) -> float:
    return (inertia * average + value) / (inertia + 1.)


def accumulating_combinations_with_replacement(elements: Iterable[T], repetitions: int) -> Generator[Tuple[T, ...], None, None]:
    yield from (c for _r in range(repetitions) for c in itertools.combinations_with_replacement(elements, _r + 1))


def product(values: Sequence[float]) -> float:
    output = 1.
    for _v in values:
        output *= _v
    return output


class JsonSerializable(Generic[T]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> T:
        """
        name_class = d["name_class"]
        name_module = d["name_module"]
        this_module = importlib.import_module(name_module)
        this_class = getattr(this_module, name_class)
        """
        raise NotImplementedError()

    def to_dict(self) -> Dict[str, Any]:
        """
        this_class = self.__class__
        d = {
            "name_class": this_class.__name__,
            "name_module": this_class.__module__,
        }
        """
        raise NotImplementedError()

    @staticmethod
    def load_from(path_name: str) -> T:
        with open(path_name, mode="r") as file:
            d = json.load(file)
            return JsonSerializable.from_dict(d)

    def save_as(self, path: str):
        with open(path, mode="w") as file:
            d = self.to_dict()
            json.dump(d, file, indent=2, sort_keys=True)


class MovingGraph:
    def __init__(self, no_plots: int, size_window: int):
        self.plots = tuple([] for _ in range(no_plots))
        self.size_window = size_window
        self.fig, self.ax = pyplot.subplots()
        self.iteration = 0

    def add_snapshot(self, points: Sequence[float]):
        for each_plot, each_value in zip(self.plots, points):
            each_plot.append(each_value)
            del(each_plot[:-self.size_window])
        self.iteration += 1

    def draw(self):
        self.ax.clear()
        x_coordinates = list(range(max(0, self.iteration - self.size_window), self.iteration))
        for i, each_plot in enumerate(self.plots):
            self.ax.plot(x_coordinates, each_plot, label=f"{i:d}")

        val_min = min(min(each_plot) for each_plot in self.plots)
        val_max = max(max(each_plot) for each_plot in self.plots)

        self.ax.set_ylim([val_min - .2 * (val_max - val_min),  val_max + .2 * (val_max - val_min)])

        pyplot.legend()
        pyplot.tight_layout()
        pyplot.pause(.05)


if __name__ == "__main__":
    g_single = ratio_generator()
    next(g_single)
    print(g_single.send(2))
    print(g_single.send(3))
    print()

    g_multiple = ratio_generator_multiple(2)
    next(g_multiple)
    print(g_multiple.send([2, 3]))
    print(g_multiple.send([3, 2]))
    print()

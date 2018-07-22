from typing import Tuple


class Regressor:
    def __init__(self, drag: int):
        self.drag = drag
        self.mean_x = 0.
        self.mean_y = 0.
        self.var_x = 0.
        self.cov_xy = 0.
        self.initial = True

    def fit(self, x: float, y: float):
        dx = x - self.mean_x
        dy = y - self.mean_y

        self.var_x = (self.drag * self.var_x + dx ** 2) / (self.drag + 1)
        self.cov_xy = (self.drag * self.cov_xy + dx * dy) / (self.drag + 1)

        if self.initial:
            self.mean_x = x
            self.mean_y = y
            self.initial = False

        else:
            self.mean_x = (self.drag * self.mean_x + x) / (self.drag + 1)
            self.mean_y = (self.drag * self.mean_y + y) / (self.drag + 1)

    def _get_parameters(self) -> Tuple[float, float]:
        a = 0. if self.var_x == 0. else self.cov_xy / self.var_x
        t = self.mean_y - a * self.mean_x
        return a, t

    def output(self, x: float) -> float:
        a, t = self._get_parameters()
        return x * a + t

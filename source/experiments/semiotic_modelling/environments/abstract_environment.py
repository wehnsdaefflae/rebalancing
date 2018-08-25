from typing import Generic, TypeVar

INPUT = TypeVar("INPUT")
OUTPUT = TypeVar("OUTPUT")


class Environment(Generic[INPUT, OUTPUT]):
    pass


class EnvTest(Environment[int, int]):
    def __init__(self):
        self.iterations = 0   # type: int

    def __next__(self) -> int:
        return_value = self.a
        self.a, self.b = self.b, self.a + self.b
        return return_value

    def __iter__(self):
        return self

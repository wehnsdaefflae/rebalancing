# https://www.dropbox.com/s/ed5hm4rd0b8bz18/optimal.pdf?dl=0
import random
from typing import Sequence

from matplotlib import pyplot


def get_sequence(length: int) -> Sequence[float]:
    random.seed(235235)
    s = [10.]

    for _ in range(length - 1):
        r = random.random() * .1
        if random.random() < .5 and s[-1] - r >= 0.:
            s.append(s[-1] - r)
        else:
            s.append(s[-1] + r)

    return s


def main():
    size = 1000
    seq = get_sequence(size)

    t = [False for _ in seq]
    r = [0. for _ in seq]

    pyplot.plot(seq)
    pyplot.show()


if __name__ == "__main__":
    main()

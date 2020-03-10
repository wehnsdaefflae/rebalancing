# https://www.dropbox.com/s/ed5hm4rd0b8bz18/optimal.pdf?dl=0
import math
import random
from typing import Sequence

# from matplotlib import pyplot


def get_sequence(start_value: float, length: int) -> Sequence[float]:
    s = [-1. for _ in range(length)]
    s[0] = start_value

    for i in range(1, length):
        value_last = s[i-1]
        r = random.uniform(-.1, .1) * value_last
        s[i] = value_last + r

    return s


def main():
    random.seed(235235)

    size = 10
    seq_sec = get_sequence(12.2, size)
    seq_ass = get_sequence(53.5, size)

    sec = [False for _ in range(size)]
    ass = [False for _ in range(size)]
    val = [-1. for _ in range(size)]

    sec[0] = True
    ass[0] = False
    val[0] = 1.

    for i in range(size - 1):
        change_sec = seq_sec[i+1] / seq_sec[i]
        change_ass = seq_ass[i+1] / seq_ass[i]

        if change_sec < change_ass and sec[i]:
            ass[i+1] = True
            sec[i+1] = False
            val[i+1] = val[i] * change_ass

        elif change_ass < change_sec and ass[i]:
            ass[i+1] = False
            sec[i+1] = True
            val[i+1] = val[i] * change_sec

        else:
            ass[i+1] = ass[i]
            sec[i+1] = sec[i]
            val[i+1] = val[i]

    sec[-1] = True
    ass[-1] = False
    change_sec = seq_sec[-1] / seq_sec[-2]
    val[-1] = val[-2] * change_sec

    print("tick  " + "".join(f"{i: 7d}" for i in range(size)))
    print()
    print("ass_v " + "".join(f"  {x:5.2f}" for x in seq_ass))
    print("sec_v " + "".join(f"  {x:5.2f}" for x in seq_sec))
    print()
    print("ass   " + "".join(f"{str(x)[0]:>7s}" for x in ass))
    print("sec   " + "".join(f"{str(x)[0]:>7s}" for x in sec))
    print()
    print("value " + "".join(f"  {x:5.2f}" for x in val))

    r = [0. for _ in range(size)]

    # pyplot.plot(seq)
    # pyplot.show()


if __name__ == "__main__":
    main()

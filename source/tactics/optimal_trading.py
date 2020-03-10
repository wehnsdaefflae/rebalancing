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
    seq_ass = get_sequence(53.5, size)
    seq_sec = get_sequence(12.2, size)

    origin_ass = [False for _ in range(size-1)]  # [False: held, True: bought]
    origin_sec = [False for _ in range(size-1)]  # [False: held, True: bought]

    val_ass = [-1. for _ in range(size)]
    val_sec = [-1. for _ in range(size)]

    val_ass[0] = 0.
    val_sec[0] = 1.

    # keep two alternative accumulative value sequences
    # split on each switch, keep alternative, discard worse sequence
    # remember actions, not states

    # todo: generalize to n assets

    for i in range(size - 1):
        # next sec
        change_sec = seq_sec[i+1] / seq_sec[i]

        hold_sec = val_sec[i] * change_sec
        sell_ass = val_ass[i] * change_sec
        if hold_sec < sell_ass or i >= size - 2:
            origin_sec[i] = True           # set origin of sec i + 1 to from ass
            val_sec[i+1] = sell_ass

        else:
            origin_sec[i] = False          # set origin of sec i + 1 to from ass
            val_sec[i+1] = hold_sec

        # next ass
        change_ass = seq_ass[i+1] / seq_ass[i]

        hold_ass = val_ass[i] * change_ass
        sell_sec = val_sec[i] * change_ass
        if hold_ass < sell_sec and i < size - 2:
            origin_ass[i] = True           # set origin of ass i + 1 to from sec
            val_ass[i+1] = sell_sec

        else:
            origin_ass[i] = False          # set origin of ass i + 1 to from ass
            val_ass[i+1] = hold_ass

    origin = origin_sec
    storage_inv = ["sec"]
    for i in range(size - 2, -1, -1):
        if origin[i]:
            if storage_inv[-1] == "sec":
                storage_inv.append("ass")
            else:
                storage_inv.append("sec")
            origin = origin_ass if origin == origin_sec else origin_sec
        else:
            storage_inv.append(storage_inv[-1])

    storage_path = storage_inv[::-1]

    print("tick    " + "".join(f"{i: 9d}" for i in range(size)))
    print()
    print("seq_ass " + "".join(f"    {x:5.2f}" for x in seq_ass))
    print("seq_sec " + "".join(f"    {x:5.2f}" for x in seq_sec))
    print()
    print("val_ass " + "".join(f"    {x:5.2f}" for x in val_ass))
    print("origin  " + "".join(["  initial"] + [f"   bought" if x else "     held" for x in origin_ass]))
    print()
    print("val_sec " + "".join(f"    {x:5.2f}" for x in val_sec))
    print("origin  " + "".join(["  initial"] + [f"   bought" if x else "     held" for x in origin_sec]))
    print()
    print("path    " + "".join(f"{x:>9s}" for x in storage_path[1:]))
    print()

    print(f"final value: {val_sec[-1]:5.2f}")

    # pyplot.plot(seq)
    # pyplot.show()


if __name__ == "__main__":
    main()

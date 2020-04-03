from typing import Iterable, Sequence, Generator


def simulate(rates: Iterable[Sequence[float]], path: Sequence[int], fees: float) -> Generator[float, None, None]:
    len_path = len(path)

    amount_asset = -1.
    asset_current = -1
    last_rate = -1.

    amount_last = -1.

    for i, rates_current in enumerate(rates):
        if i < len_path:
            asset_next = path[i]

            # first iteration, initialize stuff
            if i == 0:
                asset_current = asset_next
                amount_asset = 1. / rates_current[asset_current]

            # if hold
            rate_this = rates_current[asset_current]
            if asset_next == asset_current:
                if rate_this < 0.:
                    amount = -1. if last_rate < 0. else amount_asset * last_rate
                    yield 0. if amount < 0. or amount_last < 0. else amount - amount_last
                    amount_last = amount
                else:
                    amount = amount_asset * rate_this
                    yield 0. if amount < 0. or amount_last < 0. else amount - amount_last
                    amount_last = amount

            # if switch
            else:
                amount = amount_asset * rate_this
                amount = amount * (1. - fees)
                yield 0. if amount < 0. or amount_last < 0. else amount - amount_last
                amount_last = amount

                rate_other = rates_current[asset_next]
                if rate_other >= 0.:
                    amount_asset = amount / rate_other
                    rate_this = rate_other
                    asset_current = asset_next

                else:
                    # should actually never switch into unknown asset
                    print(f"switching into unknown asset at rate {i:d}! why?!")

            last_rate = rate_this if rate_this >= 0. else last_rate

        elif i == len_path:
            asset_current = path[-1]
            rate_this = rates_current[asset_current]
            if rate_this < 0.:
                amount = amount_asset * last_rate
                yield 0. if amount < 0. or amount_last < 0. else amount - amount_last
                amount_last = amount

            else:
                amount = amount_asset * rate_this
                yield 0. if amount < 0. or amount_last < 0. else amount - amount_last
                amount_last = amount

            break

        elif len_path < i:
            break

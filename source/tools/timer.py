import time


class Timer:
    last_time = -1  # type: int

    @staticmethod
    def time_passed(passed_time: int) -> bool:
        if 0 >= passed_time:
            raise ValueError("Only positive millisecond values allowed.")

        this_time = round(time.time() * 1000.)

        if Timer.last_time < 0:
            Timer.last_time = this_time
            return False

        elif this_time - Timer.last_time < passed_time:
            return False

        Timer.last_time = this_time
        return True

    @staticmethod
    def update_time():
        Timer.last_time = round(time.time() * 1000.)
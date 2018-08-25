import string
from typing import Generator, Optional


def text_generator(file_path: str) -> Generator[str, None, None]:
    permissible_non_letter = string.digits + string.punctuation + " "
    with open(file_path, mode="r") as file:
        for line in file:
            for character in line:
                if character in string.ascii_letters:
                    yield character.lower()

                elif character in permissible_non_letter:
                    yield character


def generator_1() -> Generator[int, Optional[int], None]:
    a = 0
    while a < 10:
        add = yield a
        add = 0 if add is None else add
        a += add


if __name__ == "__main__":
    text_path = "C:/Users/Mark/Daten/Texts/pride_prejudice.txt"
    g = text_generator(text_path)
    for v in g:
        print(v, end="")

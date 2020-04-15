from typing import Sequence, Tuple, Generator

TIMESTAMP = int
VECTOR = Sequence[float]
INPUT_VALUE = VECTOR
TARGET_VALUE = VECTOR
EXAMPLE = Tuple[TIMESTAMP, TARGET_VALUE, INPUT_VALUE]
OFFSET_EXAMPLES = Generator[EXAMPLE, None, None]

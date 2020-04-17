from typing import Sequence, Tuple, Generator

TIMESTAMP = int
VECTOR = Sequence[float]
INPUT_VALUE = VECTOR
OUTPUT_VALUE = VECTOR
EXAMPLE = Tuple[TIMESTAMP, OUTPUT_VALUE, INPUT_VALUE]
OFFSET_EXAMPLES = Generator[EXAMPLE, None, None]

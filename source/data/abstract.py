from typing import Sequence, Tuple, Generator, Dict, Any

TIMESTAMP = int
VECTOR = Sequence[float]
INPUT_VALUE = VECTOR
OUTPUT_VALUE = VECTOR
EXAMPLE = Tuple[TIMESTAMP, OUTPUT_VALUE, INPUT_VALUE]
OFFSET_EXAMPLES = Generator[EXAMPLE, None, None]

STATE = Dict[str, Any]
GENERATOR_STATES = Generator[STATE, None, None]

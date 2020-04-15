from typing import Generator, Sequence, Tuple, Dict, Any

SNAPSHOT = Dict[str, Any]
STREAM_SNAPSHOTS = Generator[SNAPSHOT, None, None]

TIMESTAMP = int
VECTOR = Sequence[float]
INPUT_VALUE = VECTOR
TARGET_VALUE = VECTOR
EXAMPLE = Tuple[TARGET_VALUE, INPUT_VALUE]
# STREAM_EXAMPLES = Generator[Tuple[TIMESTAMP, EXAMPLE], None, None]

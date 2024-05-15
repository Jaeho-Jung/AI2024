from typing import NewType, Tuple

Color = NewType("Color", Tuple[int, int, int])

RED = Color((255, 0, 0))
GREEN = Color((0, 255, 0))
BLUE = Color((0, 0, 255))
ORANGE = Color((255, 165, 0))
WHITE = Color((255, 255, 255))
YELLOW = Color((255, 255, 0))
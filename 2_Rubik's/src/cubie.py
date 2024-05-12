from dataclasses import dataclass


@dataclass
class Cubie:
    position: str
    orientation: int

    def __eq__(self, other):
        return self.position == other.position and self.orientation == other.orientation
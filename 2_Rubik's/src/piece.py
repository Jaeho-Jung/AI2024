from typing import NewType, Dict

from color import Color

Corner = NewType('Corner', Dict[str, Color])

CORNER_TO_URF = {
    'URF': '',
    'URB': 'U',
    'ULB': 'U U',
    'ULF': "U'",
    'DRF': 'R',
    'DRB': 'R R',
    'DLF': 'D R',
    'DLB': 'D D R'
}
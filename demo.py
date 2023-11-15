# from rtm import RTM

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "src"
sys.path.append(str(ROOT))

from juxtapose import Tapnet


tapnet = Tapnet()
tapnet("./asset/bike.mp4", save=True)

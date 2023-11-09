# from rtm import RTM

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "src"
sys.path.append(str(ROOT))

# from juxtapose import RTM

# model = RTM(det="rtmdet-m", tracker="bytetrack", pose="rtmpose-l")
# for x in model("asset/bike.mp4", show=False, stream=True):
#     print(x.dtype)
# model("https://c8.alamy.com/comp/2B6JDCF/kids-play-football-on-outdoor-stadium-field-children-score-a-goal-during-soccer-game-little-boy-kicking-ball-school-sports-club-training-for-young-2B6JDCF.jpg")

from juxtapose import Tapnet


tapnet = Tapnet()
tapnet("./asset/track.mp4", show=True, save=False)

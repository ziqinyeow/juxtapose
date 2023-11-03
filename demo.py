# from rtm import RTM

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "src"
sys.path.append(str(ROOT))

from juxtapose import RTM

model = RTM(det="rtmdet-m", tracker="bytetrack", pose="rtmpose-l")
model("data/bike.mp4", show=True, stream=False)
# model("https://c8.alamy.com/comp/2B6JDCF/kids-play-football-on-outdoor-stadium-field-children-score-a-goal-during-soccer-game-little-boy-kicking-ball-school-sports-club-training-for-young-2B6JDCF.jpg")

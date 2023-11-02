# from rtm import RTM

# import sys
# from pathlib import Path

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0] / "src"
# sys.path.append(str(ROOT))

from juxtapose import RTM

model = RTM(det="rtmdet-m", tracker="bytetrack", pose="rtmpose-l")
model("data/bike.mp4")

import sys

sys.path.insert(0, "src")

from juxtapose import RTM, RTMDet


model = RTM(det="rtmdet-m", tracker="bytetrack", pose="rtmpose-l")
model("./asset/shortput.MP4", show=False, save=True)

import sys

sys.path.insert(0, "src")

from juxtapose import RTM, RTMDet


model = RTM(det="rtmdet-m", tracker="botsort", pose="rtmpose-l")
print(model("./asset/football.jpeg", show=False, return_im=False))

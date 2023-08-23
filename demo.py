from rtm import RTM
from rtm.detectors import DET_MAP

model = RTM(det="rtmdet-l", tracker="", pose="rtmpose-l")
# model("data/football.jpeg", roi="rect")
model("data/bike.mp4", roi="rect")
# model("ignore/Men 100m Heat-1.mp4", roi='rect')

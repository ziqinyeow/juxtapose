from rtm import RTM

model = RTM(det="yolov8-l", rtmpose="l", tracker="n")
model("data/track_1.png", show=True, save=False)

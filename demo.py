from rtm import RTM

model = RTM(det="groundingdino", rtmpose="l", tracker="n")
model("data/track_2.png", show=True, save=False)

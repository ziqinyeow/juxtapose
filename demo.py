from rtm import RTM

model = RTM(rtmdet="l", rtmpose="l", tracker="botsort", device="cuda")
model("data", show=False, save=True)

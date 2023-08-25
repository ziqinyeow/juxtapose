from rtm import RTM

model = RTM(det="rtmdet-l", tracker="bytetrack", pose="rtmpose-l")
model("ignore/Men 100m Heat-1.mp4", roi="rect")

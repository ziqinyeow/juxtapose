# from rtm import RTM


from pose import RTM

model = RTM(det="rtmdet-l", tracker="bytetrack", pose="rtmpose-l")
model("data/bike.mp4")

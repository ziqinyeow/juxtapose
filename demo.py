# from rtm import RTM


from juxtapose import RTM

model = RTM(det="rtmdet-l", tracker="bytetrack", pose="rtmpose-l")
model("data/bike.mp4")

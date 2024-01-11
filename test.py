from juxtapose import RTM


model = RTM(det="rtmdet-m", tracker="bytetrack", pose="rtmpose-l", device="cuda")
model("./asset/football.jpeg", show=False, save=False)

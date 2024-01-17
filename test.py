import sys

sys.path.insert(0, "src")

from juxtapose import RTM


model = RTM(det="rtmdet-m", tracker="bytetrack", pose="rtmpose-l", device="cuda")
for i in model("./asset/bike2.mp4", show=False, save=False, stream=True, verbose=False):
    print(i.persons)

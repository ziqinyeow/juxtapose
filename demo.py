# from juxtapose import RTM
import sys

sys.path.insert(0, "src")

from juxtapose import RTM

model = RTM(det="rtmdet-m", tracker="bytetrack", pose="rtmpose-l")
# model(
#     "https://i.pinimg.com/736x/fc/2d/01/fc2d0133e6bac631d8493cab1969e17c.jpg",
#     show=False,
#     save=True,
# )
model("asset/bike2.mp4", show=False, save=True)

# from rtm import RTM

import sys

sys.path.insert(
    0,
    "/Users/ziqin/University/Sem 6/WIA3002 Academic Project/juxtapose/packages/sdk/src",
)

from juxtapose import RTM

# print(get_user_config_dir())

# from juxtapose import RTM

model = RTM(det="rtmdet-s", tracker="bytetrack", pose="rtmpose-s")
model("data/bike.mp4")

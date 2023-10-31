from typing import Literal

DETECTOR_TYPES = Literal[
    "rtmdet-s",
    "rtmdet-m",
    "rtmdet-l",
    "groundingdino",
    "yolov8s",
    "yolov8m",
    "yolov8l",
]

POSE_ESTIMATOR_TYPES = Literal["rtmpose-s", "rtmpose-m", "rtmpose-l"]

TRACKER_TYPES = Literal["bytetrack", "botsort", ""]

DEVICE_TYPES = Literal["cpu", "cuda"]

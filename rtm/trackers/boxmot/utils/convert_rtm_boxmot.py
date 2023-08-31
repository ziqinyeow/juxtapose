import numpy as np
from rtm.utils.core import Detections

def convert_boxmot_tracker_to_rtm(outputs):
    if len(outputs) > 0:
        return Detections(
            xyxy=outputs[:, :4],
            track_id=outputs[:, 4],
            confidence=outputs[:, 5],
            labels=outputs[:, 6],
        )
    else:
        # return empty Detections object
        return Detections(
            xyxy=np.array([]),
            track_id=np.array([]),
            confidence=np.array([]),
            labels=np.array([]),
        )

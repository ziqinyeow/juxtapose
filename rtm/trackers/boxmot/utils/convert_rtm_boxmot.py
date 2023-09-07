import numpy as np
from rtm.utils.core import Detections


def convert_boxmot_tracker_to_rtm(outputs, detections) -> Detections:
    if len(outputs) > 0:
        return Detections(
            xyxy=outputs[:, :4],
            track_id=outputs[:, 4],
            confidence=outputs[:, 5],
            labels=outputs[:, 6],
        )
    else:
        # return empty Detections object
        detections.track_id = np.array([""] * len(detections.xyxy))
        return detections

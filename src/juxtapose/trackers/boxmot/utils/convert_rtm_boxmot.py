import numpy as np
from juxtapose.utils.core import Detections


def convert_boxmot_tracker_to_rtm(outputs, detections) -> Detections:
    if len(outputs) > 0:
        # get the index of track_id that is "", remove it from outputs, don't add to detections
        outputs = outputs[outputs[:, 4] != ""]
        return Detections(
            xyxy=outputs[:, :4],
            track_id=outputs[:, 4].astype("float").astype("int"),
            confidence=outputs[:, 5],
            labels=outputs[:, 6],
        )
    else:
        # return empty Detections object
        detections.track_id = np.array([""] * len(detections.xyxy))
        return detections

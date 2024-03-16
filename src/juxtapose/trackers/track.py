import torch
import numpy as np
import inspect
from pathlib import Path
from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker
from .boxmot import StrongSORT, DeepOCSORT, OCSORT
from juxtapose.utils.core import Detections
from juxtapose.trackers.boxmot.utils.convert_rtm_boxmot import (
    convert_boxmot_tracker_to_rtm,
)
from typing import Union

ARGS_MAP = {
    "bytetrack": dict(
        tracker_type="bytetrack",  # tracker type, ['botsort', 'bytetrack']
        track_high_thresh=0.5,  # threshold for the first association
        track_low_thresh=0.1,  # threshold for the second association
        new_track_thresh=0.2,  # threshold for init new track if the detection does not match any tracks
        track_buffer=30,  # buffer to calculate the time when to remove tracks
        match_thresh=0.8,  # threshold for matching tracks
    ),
    "botsort": dict(
        tracker_type="botsort",  # tracker type, ['botsort', 'bytetrack']
        track_high_thresh=0.5,  # threshold for the first association
        track_low_thresh=0.1,  # threshold for the second association
        new_track_thresh=0.6,  # threshold for init new track if the detection does not match any tracks
        track_buffer=30,  # buffer to calculate the time when to remove tracks
        match_thresh=0.8,  # threshold for matching tracks
        # min_box_area: 10  # threshold for min box areas(for tracker evaluation, not used for now)
        # mot20: False  # for tracker evaluation(not used for now)
        # BoT-SORT settings
        cmc_method="sparseOptFlow",  # method of global motion compensation
        # ReID model related thresh (not supported yet)
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=False,
    ),
}

TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
BOXMOT_TRACKER_MAP = {
    "strongsort": StrongSORT,
    "deepocsort": DeepOCSORT,
    "ocsort": OCSORT,
}


class Tracker:
    def __init__(self, type: str = "bytetrack", device: str = "cpu") -> None:
        self.type = type
        if (
            type in BOXMOT_TRACKER_MAP.keys()
            and "model_weights"
            in inspect.signature(BOXMOT_TRACKER_MAP[type].__init__).parameters
        ):
            self.tracker = BOXMOT_TRACKER_MAP[type](
                model_weights=Path("model/osnet_x0_25_msmt17.pt"),
                device=device,
                fp16=torch.cuda.is_available() and device == "cuda",
            )
            self.tracker.model.warmup()
        elif type in BOXMOT_TRACKER_MAP.keys():
            self.tracker = BOXMOT_TRACKER_MAP[type]()
        elif type in TRACKER_MAP.keys():
            self.tracker: Union[BYTETracker, BOTSORT] = TRACKER_MAP[type](
                args=ARGS_MAP[type], frame_rate=30
            )

    def update(self, detections, im):
        # Triggered only if self.tracker_type == True
        # If self.tracker_type is strongsort, deepocsort, ocsort, transform detections into 2D dets array format
        if self.type in BOXMOT_TRACKER_MAP.keys():
            # clamp to image size and 0
            detections.xyxy[:, 0] = np.clip(detections.xyxy[:, 0], 0, im.shape[1])
            detections.xyxy[:, 1] = np.clip(detections.xyxy[:, 1], 0, im.shape[0])
            detections.xyxy[:, 2] = np.clip(detections.xyxy[:, 2], 0, im.shape[1])
            detections.xyxy[:, 3] = np.clip(detections.xyxy[:, 3], 0, im.shape[0])
            dets = np.array(
                [
                    detections.xyxy[:, 0],
                    detections.xyxy[:, 1],
                    detections.xyxy[:, 2],
                    detections.xyxy[:, 3],
                    detections.confidence,
                    detections.labels,
                ]
            ).T
            # if dets.xyxy is None:
            #     dets = np.empty((0, 6))
            temp_outputs: Detections = self.tracker.update(dets, im)
            detections = convert_boxmot_tracker_to_rtm(temp_outputs, detections)
            return detections

        # If self.tracker_type is bortsort of byte tracker, just put in detections and no need
        elif self.type in TRACKER_MAP.keys():
            detections: Detections = self.tracker.update(
                bboxes=detections.xyxy,
                confidence=detections.confidence,
                labels=detections.labels,
                img=im,
            )
            # return detection type
            return detections

        # base case
        else:
            return detections

import torch

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

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


class Tracker:
    def __init__(self, type: str = "bytetrack") -> None:
        self.tracker: Union[BYTETracker, BOTSORT] = TRACKER_MAP[type](
            args=ARGS_MAP[type], frame_rate=30
        )

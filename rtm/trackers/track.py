import torch
from pathlib import Path

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

from typing import Union

from boxmot import OCSORT, StrongSORT, DeepOCSORT

ARGS_MAP = {
    "bytetrack": dict(
        tracker_type="bytetrack",  # tracker type, ['botsort', 'bytetrack']
        track_high_thresh=0.5,  # threshold for the first association
        track_low_thresh=0.1,  # threshold for the second association
        new_track_thresh=0.6,  # threshold for init new track if the detection does not match any tracks
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
    "strongsort": dict(
        model_weights=Path("osnet_x0_25_msmt17.pt"),
        fp16=False,  # explicitly set to false to avoid fp16 error
        ema_alpha=0.8,
        max_age=30,
        max_dist=0.2,
        max_iou_dist=0.7,
        mc_lambda=0.995,
        n_init=1,
        nn_budget=100,
    ),
    "deepocsort": dict(
        model_weights=Path("osnet_x0_25_msmt17.pt"),
        alpha_fixed_emb=0.95,
        asso_func="giou",
        aw_off=False,
        aw_param=0.5,
        cmc_off=False,
        conf=0.5,
        delta_t=3,
        det_thresh=0,
        embedding_off=False,
        inertia=0.2,
        iou_thresh=0.3,
        max_age=30,
        min_hits=1,
        new_kf_off=False,
        w_association_emb=0.75,
        fp16=False,  # explicitly set to false to avoid fp16 error
    ),
    "ocsort": dict(
        det_thresh=0.2,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        use_byte=False,
    ),
}

TRACKER_MAP = {
    "bytetrack": BYTETracker,
    "botsort": BOTSORT,
    "strongsort": StrongSORT,
    "deepocsort": DeepOCSORT,
    "ocsort": OCSORT,
}

native = ["bytetrack", "botsort"]
third_party = ["strongsort", "deepocsort", "ocsort"]


class Tracker:
    def __init__(self, type: str = "bytetrack", device="cpu") -> None:
        args = ARGS_MAP[type]
        if type not in native:
            if type != "ocsort":
                args["device"] = device  # for strongsort and deepocsort only
            self.tracker: Union[
                BYTETracker, BOTSORT, StrongSORT, DeepOCSORT, OCSORT
            ] = TRACKER_MAP[type](**args)
        else:
            self.tracker: Union[
                BYTETracker, BOTSORT, StrongSORT, DeepOCSORT, OCSORT
            ] = TRACKER_MAP[type](args, frame_rate=30)

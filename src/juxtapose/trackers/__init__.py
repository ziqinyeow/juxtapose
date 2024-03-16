from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker
from .track import Tracker, TRACKER_MAP, BOXMOT_TRACKER_MAP
from .boxmot import StrongSORT, DeepOCSORT


# from .track import register_tracker

__all__ = (
    "Tracker",
    "TRACKER_MAP",
    "BOXMOT_TRACKER_MAP",
    "BOTSORT",
    "BYTETracker",
    "StrongSORT",
    "DeepOCSORT",
)  # allow simpler import

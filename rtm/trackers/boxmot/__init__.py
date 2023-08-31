# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = "10.0.41"

from rtm.trackers.boxmot.postprocessing.gsi import gsi
from rtm.trackers.boxmot.tracker_zoo import create_tracker, get_tracker_config

# from rtm.trackers.boxmot.trackers.botsort.bot_sort import BoTSORT
# from rtm.trackers.boxmot.trackers.bytetrack.byte_tracker import BYTETracker
from rtm.trackers.boxmot.trackers.deepocsort.deep_ocsort import DeepOCSort as DeepOCSORT
from rtm.trackers.boxmot.trackers.hybridsort.hybridsort import HybridSORT
from rtm.trackers.boxmot.trackers.ocsort.ocsort import OCSort as OCSORT
from rtm.trackers.boxmot.trackers.strongsort.strong_sort import StrongSORT

TRACKERS = [
    # "bytetrack",
    # "botsort",
    "strongsort",
    "ocsort",
    "deepocsort",
    "hybridsort",
]

__all__ = (
    "__version__",
    "StrongSORT",
    "OCSORT",
    # "BYTETracker",
    # "BoTSORT",
    "DeepOCSORT",
    "HybridSORT",
    "create_tracker",
    "get_tracker_config",
    "gsi",
)

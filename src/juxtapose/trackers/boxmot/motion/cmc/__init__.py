# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from juxtapose.trackers.boxmot.motion.cmc.ecc import ECC
from juxtapose.trackers.boxmot.motion.cmc.orb import ORB
from juxtapose.trackers.boxmot.motion.cmc.sift import SIFT
from juxtapose.trackers.boxmot.motion.cmc.sof import SparseOptFlow


def get_cmc_method(cmc_method):
    if cmc_method == "ecc":
        return ECC
    elif cmc_method == "orb":
        return ORB
    elif cmc_method == "sof":
        return SparseOptFlow
    elif cmc_method == "sift":
        return SIFT
    else:
        return None

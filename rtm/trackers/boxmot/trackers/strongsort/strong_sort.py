# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np

from rtm.trackers.boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
from rtm.trackers.boxmot.motion.cmc import get_cmc_method
from rtm.trackers.boxmot.trackers.strongsort.sort.detection import Detection
from rtm.trackers.boxmot.trackers.strongsort.sort.tracker import Tracker
from rtm.trackers.boxmot.utils.matching import NearestNeighborDistanceMetric
from rtm.trackers.boxmot.utils.ops import xyxy2tlwh
from rtm.trackers.boxmot.utils.convert_rtm_boxmot import convert_boxmot_tracker_to_rtm


class StrongSORT(object):
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        max_dist=0.2,
        max_iou_dist=0.7,
        max_age=30,
        n_init=1,
        nn_budget=100,
        mc_lambda=0.995,
        ema_alpha=0.9,
    ):
        self.model = ReIDDetectMultiBackend(
            weights=model_weights, device=device, fp16=fp16
        )
        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
        )
        self.cmc = get_cmc_method("ecc")()

    def update(self, dets, img):
        # def update(self, bboxes, confidence, labels, img):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        xyxy = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        det_ind = dets[:, 6]

        # assert isinstance(
        #     bboxes, np.ndarray
        # ), f"Unsupported 'dets' input format '{type(bboxes)}', valid format is np.ndarray"
        # assert isinstance(
        #     img, np.ndarray
        # ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        # assert (
        #     len(bboxes) == len(confidence) == len(labels)
        # ), "bboxes and confidence and labels should have same dimensions"

        # xyxy = bboxes
        # confs = confidence
        # clss = labels
        # det_ind = np.arange(len(bboxes))

        if len(self.tracker.tracks) >= 1:
            warp_matrix = self.cmc.apply(img, xyxy)
            for track in self.tracker.tracks:
                track.camera_update(warp_matrix)

        # extract appearance information for each detection
        features = self.model.get_features(xyxy, img)

        tlwh = xyxy2tlwh(xyxy)
        detections = [
            Detection(box, conf, cls, det_ind, feat)
            for box, conf, cls, det_ind, feat in zip(
                tlwh, confs, clss, det_ind, features
            )
        ]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue

            x1, y1, x2, y2 = track.to_tlbr()

            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind

            outputs.append(
                np.concatenate(
                    ([x1, y1, x2, y2], [id], [conf], [cls], [det_ind])
                ).reshape(1, -1)
            )
        if len(outputs) > 0:
            return np.concatenate(outputs)
        return np.array([])
        # if len(outputs) > 0:
        #     outputs = np.concatenate(outputs)
        # return convert_boxmot_tracker_to_rtm(outputs)

from pathlib import Path
from ultralytics import YOLO
import numpy as np

from juxtapose.utils import LOGGER
from juxtapose.utils.core import Detections

from juxtapose.utils.downloads import safe_download

base = "juxtapose"


class YOLOv8:
    """YOLOv8 model (s, m, l) to detect only person (class 0)"""

    def __init__(self, type: str = "m", device: str = "cpu", conf_thres: float = 0.3):
        self.model = YOLO(f"model/yolov8{type}.pt")
        self.model.to(device)
        self.conf_thres = conf_thres
        LOGGER.info(f"Loaded yolov8{type} pt model.")

    def __call__(self, im):
        """Return List of xyxy coordinates based on im frame (cv2.imread)
        im -> (h, w, c)
        return -> [[x, y, x, y], [x, y, x, y], ...] -> two persons detected
        """
        result = self.model(im, verbose=False, conf=self.conf_thres)[0].boxes
        result = result[result.cls == 0].cpu().numpy()
        result = Detections(
            xyxy=result.xyxy,
            confidence=result.conf,
            labels=np.array([0 for _ in range(len(result.xyxy))]),
        )
        # print(result)

        return result

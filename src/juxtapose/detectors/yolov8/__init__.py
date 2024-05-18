from ultralytics import YOLO
import numpy as np

from pathlib import Path
from juxtapose.utils import LOGGER
from juxtapose.utils.core import Detections

import torch
from juxtapose.utils.downloads import safe_download

base = "juxtapose"


class YOLOv8:
    """YOLOv8 model (s, m, l) to detect only person (class 0)"""

    def __init__(self, type: str = "m", device: str = "cpu", conf_thres: float = 0.3):

        download_dir = Path("model")
        onnx_model = download_dir / f"yolov8{type}.onnx"

        if not onnx_model.exists():
            safe_download(
                f"https://huggingface.co/ziq/rtm/resolve/main/yolov8{type}.onnx",
                file=f"yolov8{type}",
                dir=download_dir,
            )

        self.model = YOLO(f"model/yolov8{type}.onnx", task="detect")
        self.device = device
        # if device == "cuda" and torch.cuda.is_available():
        #     self.model.to(device)
        # else:
        #     self.model.to("cpu")
        self.conf_thres = conf_thres
        LOGGER.info(f"Loaded yolov8{type} onnx model.")

    def __call__(self, im):
        """Return List of xyxy coordinates based on im frame (cv2.imread)
        im -> (h, w, c)
        return -> [[x, y, x, y], [x, y, x, y], ...] -> two persons detected
        """
        result = self.model(im, verbose=False, conf=self.conf_thres, device=0)[0].boxes
        result = result[result.cls == 0].cpu().numpy()
        result = Detections(
            xyxy=result.xyxy,
            confidence=result.conf,
            labels=np.array([0 for _ in range(len(result.xyxy))]),
        )
        # print(result)

        return result

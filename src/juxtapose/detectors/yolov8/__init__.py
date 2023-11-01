from pathlib import Path
from ultralytics import YOLO
import supervision as sv

from juxtapose.utils.downloads import safe_download

base = "pose"


class YOLOv8:
    """YOLOv8 model (s, m, l) to detect only person (class 0)"""

    def __init__(self, type: str = "m", device: str = "cpu", conf_thres: float = 0.3):
        # onnx_file = Path(f"model/rtmdet-{type}.onnx")
        # if not onnx_file.exists():
        #     safe_download(
        #         f"https://huggingface.co/ziq/rtm/resolve/main/rtmdet-{type}.onnx",
        #         file=f"rtmdet-{type}",
        #         dir=Path(f"model/"),
        #     )
        self.model = YOLO(f"yolov8{type}.pt")
        self.model.to(device)
        self.conf_thres = conf_thres

    def __call__(self, im):
        """Return List of xyxy coordinates based on im frame (cv2.imread)
        im -> (h, w, c)
        return -> [[x, y, x, y], [x, y, x, y], ...] -> two persons detected
        """
        result = self.model(im, verbose=False, conf=self.conf_thres)[0].boxes
        result = result[result.cls == 0].cpu().numpy()
        result = sv.Detections(
            xyxy=result.xyxy,
            confidence=result.conf,
        )
        result.labels = [0 for _ in range(len(result.xyxy))]

        return result

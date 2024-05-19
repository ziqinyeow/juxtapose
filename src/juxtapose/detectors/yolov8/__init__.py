import numpy as np

from pathlib import Path
from juxtapose.utils import LOGGER
from juxtapose.utils.core import Detections
import cv2

import onnxruntime as ort

from juxtapose.utils.downloads import safe_download

base = "juxtapose"


class YOLOv8:
    """YOLOv8 model (s, m, l) to detect only person (class 0)"""

    def __init__(
        self,
        type: str = "m",
        device: str = "cpu",
        conf_thres: float = 0.3,
        iou_thres: float = 0.45,
    ):

        download_dir = Path("model")
        onnx_model = download_dir / f"yolov8{type}.onnx"

        if not onnx_model.exists():
            safe_download(
                f"https://huggingface.co/ziq/rtm/resolve/main/yolov8{type}.onnx",
                file=f"yolov8{type}",
                dir=download_dir,
            )

        # self.model = YOLO(, task="detect")
        # self.device = device
        # if device == "cuda" and torch.cuda.is_available():
        #     self.model.to(device)
        # else:
        #     self.model.to("cpu")
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        providers = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}[
            device
        ]

        # Create an inference session using the ONNX model and specify execution providers
        self.session = ort.InferenceSession(
            f"model/yolov8{type}.onnx",
            providers=[providers],
        )

        # Get the model inputs
        self.model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        LOGGER.info(f"Loaded yolov8{type} onnx model into {providers}.")

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def postprocess(self, input_image, output):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]

        boxes = []
        scores = []
        class_ids = []

        img_height, img_width = input_image.shape[:2]
        x_factor = img_width / self.input_width
        y_factor = img_height / self.input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.conf_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        box_res = []
        score_res = []
        class_id_res = []

        for i in indices:
            # if it is human
            if class_ids[i] == 0:
                box = boxes[i]
                xyxy = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                box_res.append(xyxy)
                score_res.append(scores[i])
                class_id_res.append(class_ids[i])

        return box_res, score_res, class_id_res

    def inference(self, im):
        # Preprocess the image data
        img_data = self.preprocess(im)
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
        return self.postprocess(im, outputs)  # output image

    def __call__(self, im):
        """Return List of xyxy coordinates based on im frame (cv2.imread)
        im -> (h, w, c)
        return -> [[x, y, x, y], [x, y, x, y], ...] -> two persons detected
        """
        boxes, scores, class_id = self.inference(im)
        result = Detections(
            xyxy=np.array(boxes),
            confidence=np.array(scores),
            labels=np.array([0 for _ in range(len(boxes))]),
        )

        return result

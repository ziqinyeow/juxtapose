# https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmpose.md

from typing import List, Tuple
import numpy as np
from pathlib import Path
import onnxruntime as ort

from .postprocessing import get_simcc_maximum
from .preprocessing import bbox_xyxy2cs, top_down_affine
from juxtapose.utils import LOGGER
from juxtapose.utils.downloads import safe_download

base = "juxtapose"


class RTMPose:
    """RTMPose model (s, m, l) to detect multi-person poses (class 0) based on bboxes"""

    def __init__(self, type: str = "m", device: str = "cpu") -> None:
        download_dir = Path("model")

        onnx_model = download_dir / f"rtmpose-{type}.onnx"

        if not onnx_model.exists():
            safe_download(
                f"https://huggingface.co/ziq/rtm/resolve/main/rtmpose-{type}.onnx",
                file=f"rtmpose-{type}",
                dir=download_dir,
            )

        providers = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}[
            device
        ]

        self.session = ort.InferenceSession(
            path_or_bytes=onnx_model, providers=[providers]
        )

        self.onnx_model = onnx_model
        self.model_input_size = (192, 256)
        self.mean = (123.675, 116.28, 103.53)
        self.std = (58.395, 57.12, 57.375)
        self.device = device
        LOGGER.info(f"Loaded rtmpose-{type} onnx model into {providers}.")
        # self.conf_thres = conf_thres

    def __call__(self, im: np.ndarray, bboxes: list = []):
        """Return List of 17 xy coordinates based on im frame (cv2.imread)
        im -> (h, w, c)
        bboxes -> [[x, y, x, y], [x, y, x, y], ...] -> two persons detected
        return -> np.ndarray([[x, y, ... 17 times], [x, y, ... 17 times], ...]) -> (2 or more, 17, 2)
        """

        if len(bboxes) == 0:
            bboxes = [[0, 0, im.shape[1], im.shape[0]]]

        keypoints, scores = [], []
        for bbox in bboxes:
            img, center, scale = self.preprocess(im, bbox)
            outputs = self.inference(img)
            kpts, score = self.postprocess(outputs, center, scale)

            keypoints.append(kpts)
            scores.append(score)

        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)

        return keypoints, scores

    def inference(self, im: np.array):
        im = im.transpose(2, 0, 1)
        im = np.ascontiguousarray(im, dtype=np.float32)
        input = im[None, :, :, :]

        sess_input = {self.session.get_inputs()[0].name: input}
        sess_output = []
        for out in self.session.get_outputs():
            sess_output.append(out.name)

        outputs = self.session.run(sess_output, sess_input)
        return outputs

    def preprocess(self, img: np.ndarray, bbox: list):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            bbox (list):  xyxy-format bounding box of target.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        bbox = np.array(bbox)

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = top_down_affine(self.model_input_size, scale, center, img)
        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            resized_img = (resized_img - self.mean) / self.std

        return resized_img, center, scale

    def postprocess(
        self,
        outputs: List[np.ndarray],
        center: Tuple[int, int],
        scale: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # decode simcc
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        # rescale keypoints
        keypoints = keypoints / self.model_input_size * scale
        keypoints = keypoints + center - scale / 2

        return keypoints, scores

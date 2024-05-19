# https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmdet.md


from pathlib import Path
from juxtapose.utils import LOGGER
from juxtapose.utils.downloads import safe_download
from juxtapose.utils.core import Detections

import cv2
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from .utils import multiclass_nms


base = "juxtapose"


class RTMDet:
    """RTMDet model (s, m, l) to detect only person (class 0)"""

    def __init__(self, type: str = "m", device: str = "cpu", conf_thres: float = 0.3):
        download_dir = Path("model")
        onnx_model = download_dir / f"rtmdet-{type}.onnx"

        if not onnx_model.exists():
            safe_download(
                f"https://huggingface.co/ziq/rtm/resolve/main/rtmdet-{type}.onnx",
                file=f"rtmdet-{type}",
                dir=download_dir,
            )

        providers = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}[
            device
        ]

        self.session = ort.InferenceSession(
            path_or_bytes=onnx_model, providers=[providers]
        )

        self.onnx_model = onnx_model
        self.model_input_size = (640, 640)
        self.mean = (103.53, 116.28, 123.675)
        self.std = (57.375, 57.12, 58.395)
        self.device = device
        self.conf_thres = conf_thres
        LOGGER.info(f"Loaded rtmdet-{type} onnx model into {providers}.")

    def inference(self, im: np.ndarray):
        im = im.transpose(2, 0, 1)
        im = np.ascontiguousarray(im, dtype=np.float32)
        input = im[None, :, :, :]

        sess_input = {self.session.get_inputs()[0].name: input}
        sess_output = []
        for out in self.session.get_outputs():
            sess_output.append(out.name)

        outputs = self.session.run(sess_output, sess_input)
        # print(outputs[0].shape, outputs[1].shape)
        return outputs

    def __call__(self, im):
        """Return List of xyxy coordinates based on im frame (cv2.imread)
        im -> (h, w, c)
        return -> [[x, y, x, y], [x, y, x, y], ...] -> two persons detected
        """
        im, ratio = self.preprocess(im)
        outputs = self.inference(im)
        preds, scores, labels = self.postprocess(outputs, ratio)
        results = Detections(
            xyxy=preds,
            confidence=scores,
            labels=labels,
        )

        return results

    def preprocess(self, img: np.ndarray):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        # print("pre", img.shape)
        if len(img.shape) == 3:
            padded_img = (
                np.ones(
                    (self.model_input_size[0], self.model_input_size[1], 3),
                    dtype=np.uint8,
                )
                * 114
            )
        else:
            padded_img = np.ones(self.model_input_size, dtype=np.uint8) * 114

        ratio = min(
            self.model_input_size[0] / img.shape[0],
            self.model_input_size[1] / img.shape[1],
        )
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[: padded_shape[0], : padded_shape[1]] = resized_img
        # print(padded_shape, resized_img.shape, padded_img.shape)

        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            padded_img = (padded_img - self.mean) / self.std

        return padded_img, ratio

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for RTMDet model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of RTMDet model.
            ratio (float): Ratio of preprocessing.

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_scores (np.ndarray): Final scores.
        """
        outputs, [labels] = outputs

        if outputs.shape[-1] == 4:
            # onnx without nms module

            grids = []
            expanded_strides = []
            strides = [8, 16, 32]

            hsizes = [self.model_input_size[0] // stride for stride in strides]
            wsizes = [self.model_input_size[1] // stride for stride in strides]

            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                expanded_strides.append(np.full((*shape, 1), stride))

            grids = np.concatenate(grids, 1)
            expanded_strides = np.concatenate(expanded_strides, 1)
            outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
            outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

            predictions = outputs[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
            boxes_xyxy /= ratio
            dets = multiclass_nms(
                boxes_xyxy, scores, nms_thr=self.nms_thr, score_thr=self.conf_thres
            )
            if dets is not None:
                pack_dets = (dets[:, :4], dets[:, 4], dets[:, 5])
                final_boxes, final_scores, final_cls_inds = pack_dets
                isscore = final_scores > 0.3
                iscat = final_cls_inds == 0
                isbbox = [i and j for (i, j) in zip(isscore, iscat)]
                final_boxes = final_boxes[isbbox]

        elif outputs.shape[-1] == 5:
            # onnx contains nms module

            pack_dets = (outputs[0, :, :4], outputs[0, :, 4])
            # print(outputs.shape, pack_dets[0].shape, pack_dets[1].shape)
            final_boxes, final_scores = pack_dets
            final_boxes /= ratio
            isscore = final_scores > self.conf_thres
            isperson = labels == 0
            isbbox = [i and j for (i, j) in zip(isscore, isperson)]
            final_boxes = final_boxes[isbbox]

        return final_boxes, final_scores[isbbox], labels[isbbox]

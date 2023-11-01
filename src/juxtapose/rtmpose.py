# https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmpose.md

import numpy as np

from pathlib import Path

from juxtapose.utils.downloads import safe_download
from juxtapose.mmdeploy.apis.utils import build_task_processor
from juxtapose.mmdeploy.utils import get_input_shape, load_config
from mmpose.structures import merge_data_samples


base = "pose"


class RTMPose:
    """RTMPose model (s, m, l) to detect multi-person poses (class 0) based on bboxes"""

    def __init__(self, type: str = "m", device: str = "cpu") -> None:
        model_cfg = f"model/rtmpose-{type}.py"
        onnx_file = Path(f"model/rtmpose-{type}.onnx")
        deploy_cfg = f"model/pose-detection_simcc_onnxruntime_dynamic.py"

        if not Path(model_cfg).exists():
            safe_download(
                f"https://huggingface.co/ziq/rtm/resolve/main/rtmpose-{type}.py",
                file=f"rtmpose-{type}",
                dir=Path(f"model/"),
            )
        if not onnx_file.exists():
            safe_download(
                f"https://huggingface.co/ziq/rtm/resolve/main/rtmpose-{type}.onnx",
                file=f"rtmpose-{type}",
                dir=Path(f"model/"),
            )
        if not Path(deploy_cfg).exists():
            safe_download(
                f"https://huggingface.co/ziq/rtm/resolve/main/pose-detection_simcc_onnxruntime_dynamic.py",
                file=f"detection_simcc_onnxruntime_dynamic",
                dir=Path(f"model/"),
            )

        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

        # build task and backend model
        self.task_processor = build_task_processor(model_cfg, deploy_cfg, device)
        self.model = self.task_processor.build_backend_model([onnx_file])

        # process input image
        self.input_shape = get_input_shape(deploy_cfg)

    def create_input(self, im, bboxes=None):
        model_inputs, _ = self.task_processor.create_input(
            im, self.input_shape, bboxes=bboxes
        )
        return model_inputs

    def __call__(self, im, bboxes=None):
        """Return List of 17 xy coordinates based on im frame (cv2.imread)
        im -> (h, w, c)
        bboxes -> [[x, y, x, y], [x, y, x, y], ...] -> two persons detected
        return -> np.ndarray([[x, y, ... 17 times], [x, y, ... 17 times], ...]) -> (2 or more, 17, 2)
        """

        if bboxes is not None:
            bboxes = [np.array([bbox]) for bbox in bboxes]

        model_inputs = self.create_input(im, bboxes)
        result = self.model.test_step(model_inputs)
        result = merge_data_samples(result)
        kpts = result.pred_instances.keypoints

        return kpts

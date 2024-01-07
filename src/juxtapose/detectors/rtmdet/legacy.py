# https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmdet.md


from pathlib import Path
from juxtapose.mmdeploy.apis.utils import build_task_processor
from juxtapose.mmdeploy.utils import get_input_shape, load_config
from juxtapose.utils import get_user_config_dir
from juxtapose.utils.downloads import safe_download
from juxtapose.utils.core import Detections

base = "juxtapose"


class RTMDet:
    """RTMDet model (s, m, l) to detect only person (class 0)"""

    def __init__(self, type: str = "m", device: str = "cpu", conf_thres: float = 0.3):
        download_dir = Path("model")

        model_cfg = download_dir / f"rtmdet-{type}.py"
        onnx_file = download_dir / f"rtmdet-{type}.onnx"
        deploy_cfg = download_dir / f"detection_onnxruntime_static.py"

        if not model_cfg.exists():
            safe_download(
                f"https://huggingface.co/ziq/rtm/resolve/main/rtmdet-{type}.py",
                file=f"rtmdet-{type}",
                dir=download_dir,
            )
        if not onnx_file.exists():
            safe_download(
                f"https://huggingface.co/ziq/rtm/resolve/main/rtmdet-{type}.onnx",
                file=f"rtmdet-{type}",
                dir=download_dir,
            )
        if not deploy_cfg.exists():
            safe_download(
                f"https://huggingface.co/ziq/rtm/resolve/main/detection_onnxruntime_static.py",
                file=f"detection_onnxruntime_static",
                dir=download_dir,
            )

        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

        # build task and backend model
        self.task_processor = build_task_processor(model_cfg, deploy_cfg, device)
        self.model = self.task_processor.build_backend_model([onnx_file])

        # process input image
        self.input_shape = get_input_shape(deploy_cfg)

        self.conf_thres = conf_thres

    def __call__(self, im):
        """Return List of xyxy coordinates based on im frame (cv2.imread)
        im -> (h, w, c)
        return -> [[x, y, x, y], [x, y, x, y], ...] -> two persons detected
        """

        model_inputs, _ = self.task_processor.create_input(im, self.input_shape)
        result = self.model.test_step(model_inputs)

        pred_instances = result[0].pred_instances

        # filter confidence threshold
        pred_instances = pred_instances[pred_instances.scores > self.conf_thres]

        # get only class 0 (person)
        pred_instances = pred_instances[pred_instances.labels == 0].cpu().numpy()

        result = Detections(
            xyxy=pred_instances.bboxes,
            confidence=pred_instances.scores,
            labels=pred_instances.labels,
        )

        return result

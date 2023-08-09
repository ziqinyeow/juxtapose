# https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmdet.md


from charset_normalizer import detect
from rtm.mmdeploy.apis.utils import build_task_processor
from rtm.mmdeploy.utils import get_input_shape, load_config
from rtm.utils.downloads import safe_download
from pathlib import Path

base = "rtm"


class RTMDet:
    """RTMDet model (s, m, l) to detect only person (class 0)"""

    def __init__(self, type: str = "m", device: str = "cpu", conf_thres: float = 0.3):
        model_cfg = f"{base}/conf/rtmdet-{type}.py"
        onnx_file = Path(f"model/rtmdet-{type}.onnx")
        if not onnx_file.exists():
            safe_download(
                f"https://huggingface.co/ziq/rtm/resolve/main/rtmdet-{type}.onnx",
                file=f"rtmdet-{type}",
                dir=Path(f"model/"),
            )

        deploy_cfg = "rtm/mmdeploy/configs/mmdet/detection_onnxruntime_static.py"

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
        bboxes = pred_instances.bboxes
        scores = pred_instances.scores
        labels = pred_instances.labels

        detections = []
        for bbox, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = bbox
            detections.append([x1, y1, x2, y2, score, label])
        return (
            bboxes,
            scores,
            labels,
            detections,
        )

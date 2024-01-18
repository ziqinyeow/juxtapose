import cv2
import torch
import bisect
import numpy as np
from pathlib import Path
from juxtapose.utils.core import Detections
from juxtapose.utils import ops


from juxtapose.utils.downloads import safe_download
import juxtapose.detectors.groundingdino.datasets.transforms as T
from juxtapose.detectors.groundingdino.util.inference import (
    load_model,
    preprocess_caption,
    get_phrases_from_posmap,
    box_convert,
)


class GroundingDino:
    def __init__(
        self,
        type="",
        device: str = "cpu",
        captions: str = "person .",
        conf_thres: float = 0.35,
        text_thres: float = 0.25,
    ):
        model_path = Path(f"model/groundingdino_swint_ogc.pth")
        config_path = Path(f"model/gd-ogc.py")
        if not model_path.exists():
            safe_download(
                f"https://huggingface.co/ziq/rtm/resolve/main/groundingdino_swint_ogc.pth",
                file=f"groundingdino_swint_ogc.pth",
                dir=Path(f"model/"),
            )
        if not config_path.exists():
            safe_download(
                f"https://huggingface.co/ziq/rtm/resolve/main/gd-ogc.py",
                file=f"gd-ogc.py",
                dir=Path(f"model/"),
            )
        self.model = load_model(str(config_path), model_path, captions=captions)
        self.model.to(device)
        self.device = device
        self.captions = captions
        self.conf_thres = conf_thres
        self.text_thres = text_thres

    def transform(self, im):
        _transform = T.Compose(
            [
                T.ToPILImage(),
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        im, _ = _transform(im, None)
        return im

    def predict(self, im, remove_combined: bool = False):
        with torch.no_grad():
            outputs = self.model(im[None])
            # outputs = self.model(im[None], captions=[self.captions])

        prediction_logits = (
            outputs["pred_logits"].cpu().sigmoid()[0]
        )  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[
            0
        ]  # prediction_boxes.shape = (nq, 4)

        mask = prediction_logits.max(dim=1)[0] > self.conf_thres
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        tokenizer = self.model.tokenizer
        tokenized = tokenizer(self.captions)

        if remove_combined:
            sep_idx = [
                i
                for i in range(len(tokenized["input_ids"]))
                if tokenized["input_ids"][i] in [101, 102, 1012]
            ]

            phrases = []
            for logit in logits:
                max_idx = logit.argmax()
                insert_idx = bisect.bisect_left(sep_idx, max_idx)
                right_idx = sep_idx[insert_idx]
                left_idx = sep_idx[insert_idx - 1]
                phrases.append(
                    get_phrases_from_posmap(
                        logit > self.text_thres,
                        tokenized,
                        tokenizer,
                        left_idx,
                        right_idx,
                    ).replace(".", "")
                )
        else:
            phrases = [
                get_phrases_from_posmap(
                    logit > self.text_thres, tokenized, tokenizer
                ).replace(".", "")
                for logit in logits
            ]

        return boxes, logits.max(dim=1)[0], phrases

    def __call__(self, im):
        h, w, _ = im.shape
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = self.transform(im)
        bboxes, conf, labels = self.predict(im)
        bboxes = bboxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=bboxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        result = Detections(
            xyxy=xyxy,
            confidence=conf.numpy(),
            labels=np.array([0 if label == "person" else 0 for label in labels]),
            # labels=np.array([0 if label == "person" else -1 for label in labels]),
        )
        # print(result)
        return result

"""Main class to perform inference using RTMDet and RTMPose (ONNX)"""

import cv2
import csv
from pathlib import Path
import numpy as np
import supervision as sv

from typing import List, Union, Generator, Literal, Optional

from juxtapose.data import load_inference_source
from juxtapose.detectors import get_detector, RTMDet, GroundingDino, YOLOv8
from juxtapose.rtmpose import RTMPose

from juxtapose.utils.core import Detections
from juxtapose.utils.plotting import Annotator
from juxtapose.utils.roi import select_roi
from juxtapose.utils.checks import check_imshow
from juxtapose.utils.torch_utils import smart_inference_mode
from juxtapose.utils.ops import xyxy2xyxyxyxy
from juxtapose.utils import (
    LOGGER,
    MACOS,
    WINDOWS,
    colorstr,
    ops,
    get_time,
)
from juxtapose.trackers import Tracker, TRACKER_MAP

from pydantic import BaseModel


class Result(BaseModel):
    im: np.ndarray  # shape -> (h, w, c)
    kpts: List  # shape -> (number of humans, 17, 2)
    bboxes: List  # shape -> (number of humans, 4)
    speed: dict  # {'bboxes': ... ms, 'kpts': ... ms} -> used to record the milliseconds of the inference time

    save_dirs: str  # save directory
    name: str  # file name

    class Config:
        arbitrary_types_allowed = True


class RTM:
    """"""

    def __init__(
        self,
        device="cpu",
        annotator=Annotator(),
    ) -> None:
        self.dataset = None
        self.vid_path, self.vid_writer = None, None

    def setup_source(self, source, imgsz=640, vid_stride=1) -> None:
        self.dataset = load_inference_source(
            source=source, imgsz=imgsz, vid_stride=vid_stride
        )
        self.source_type = self.dataset.source_type
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [
            None
        ] * self.dataset.bs

    def save_preds(self, im0, vid_cap, idx, save_path) -> None:
        """Save video predictions as mp4 at specified path."""
        # Save imgs
        if self.dataset.mode == "image":
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(
                        vid_cap.get(cv2.CAP_PROP_FPS)
                    )  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                suffix = ".mp4" if MACOS else ".avi" if WINDOWS else ".avi"
                fourcc = "avc1" if MACOS else "WMV2" if WINDOWS else "MJPG"
                save_path = str(Path(save_path).with_suffix(suffix))
                self.vid_writer[idx] = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
                )
            self.vid_writer[idx].write(im0)

    def save_csv(self, path, data):
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def get_labels(self, detections: Detections):
        mp = {0: "person"}
        return np.array(
            [
                f"{mp[label]} {id} {score:.2f}"
                for score, label, id in zip(
                    detections.confidence, detections.labels, detections.track_id
                )
            ]
        )

    @smart_inference_mode
    def stream_inference(
        self,
        source,
        # interest point
        timer: List = [],
        poi: Literal["point", "box", "text", ""] = "",
        roi: Literal["rect", ""] = "",
        # panels
        show=True,
        plot=True,
        plot_bboxes=True,
        save=False,
        save_dirs="",
        verbose=True,
    ) -> Result:
        if show:
            check_imshow(warn=True)

        if save:
            save_dirs = Path(get_time() if not save_dirs else save_dirs)
            save_dirs.mkdir(exist_ok=True)

        profilers = (ops.Profile(), ops.Profile(), ops.Profile())  # count the time
        self.setup_source(source)

        if self.tracker_type:
            self.setup_tracker()

        current_source, index = None, 0

        for _, batch in enumerate(self.dataset):
            path, im0s, vid_cap, s = batch
            is_image = s.startswith("image")

            p, im, im0 = Path(path[0]), im0s[0], im0s[0].copy()

            index += 1

            # reset tracker when source changed
            if current_source is None:
                current_source = p
                if roi:
                    zones = self.setup_roi(im)
            elif current_source != p:
                if roi:
                    zones = self.setup_roi(im)
                if self.tracker_type:
                    self.setup_tracker()
                current_source = p
                index = 1

            # multi object detection (detect only person)
            with profilers[0]:
                detections: Detections = self.det(im)

            if plot:
                labels = self.get_labels(detections)

                if plot_bboxes:
                    self.box_annotator.annotate(im, detections, labels=labels)
                # self.annotator.draw_kpts(im, kpts)
                # self.annotator.draw_skeletons(im, kpts)

            result = Result(
                im=im,
                bboxes=detections.xyxy,  # detections.xyxy,
                speed={
                    "bboxes": profilers[0].dt * 1e3 / 1,
                    "track": profilers[1].dt * 1e3 / 1,
                    "kpts": profilers[2].dt * 1e3 / 1,
                },
                save_dirs=str(save_dirs) if save else "",
                name=p.name,
            )

            yield result

            if show:
                if is_image:
                    cv2.imshow(result.name, result.im)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("q"):
                        cv2.destroyWindow(result.name)
                        continue
                else:
                    cv2.imshow(result.name, result.im)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == 32:
                        zones = self.setup_roi(im0)

            if verbose:
                LOGGER.info(
                    f"{s}Detected {len(detections.xyxy)} person(s) ({colorstr('green', f'{profilers[0].dt * 1E3:.1f}ms')} | {colorstr('green', f'{profilers[1].dt * 1E3:.1f}ms')} | {colorstr('green', f'{profilers[2].dt * 1E3:.1f}ms')})"
                )

            if save:
                self.save_preds(im, vid_cap, 0, str(save_dirs / p.name))
                self.save_csv(
                    str(save_dirs / p.with_suffix(".csv").name),
                    [
                        str(index),
                        # str([{i: kpt} for i, kpt in zip(detections.track_id, kpts)]),
                    ],
                )

        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()

        cv2.destroyAllWindows()

    def __call__(
        self,
        source,
        stream=False,
        # interest point
        timer: List = [],
        poi: Literal["point", "box", "text", ""] = "",
        roi: Literal["rect", ""] = "",
        # panels
        show=True,
        plot=True,
        plot_bboxes=True,
        save=False,
        save_dirs="",
        verbose=True,
    ) -> Union[List[Result], Generator[Result, None, None]]:
        if stream:
            return self.stream_inference(
                source,
                timer=timer,
                poi=poi,
                roi=roi,
                show=show,
                plot=plot,
                plot_bboxes=plot_bboxes,
                save=save,
                save_dirs=save_dirs,
                verbose=verbose,
            )
        else:
            return list(
                self.stream_inference(
                    source,
                    timer=timer,
                    poi=poi,
                    roi=roi,
                    show=show,
                    plot=plot,
                    plot_bboxes=plot_bboxes,
                    save=save,
                    save_dirs=save_dirs,
                    verbose=verbose,
                )
            )

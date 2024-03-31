import functools
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import mediapy as media
from pathlib import Path

import cv2
import csv
import supervision as sv

from typing import List, Union, Generator, Literal

from juxtapose.data import load_inference_source
from juxtapose.trackers.tapnet import tapir_model
from juxtapose.trackers.tapnet.utils import transforms
from juxtapose.trackers.tapnet.utils import viz_utils
from juxtapose.utils.downloads import safe_download
from juxtapose.utils.checks import check_imshow
from juxtapose.utils.plotting import Annotator
from juxtapose.utils import (
    LOGGER,
    MACOS,
    WINDOWS,
    colorstr,
    ops,
    get_time,
)

from dataclasses import dataclass


@dataclass
class Result:
    im: np.ndarray  # shape -> (h, w, c)
    tracks: List  # shape -> (number of humans, 17, 2)
    speed: dict  # {'bboxes': ... ms, 'kpts': ... ms} -> used to record the milliseconds of the inference time

    save_dirs: str  # save directory
    name: str  # file name


class Tapnet:
    def __init__(
        self,
        points: List = [],
        annotator=Annotator(),
    ):
        self.points = [np.array(point) for point in points]
        download_dir = Path("model")
        checkpoint_path = download_dir / f"causal_tapir_checkpoint.npy"

        if not checkpoint_path.exists():
            safe_download(
                f"https://huggingface.co/ziq/rtm/resolve/main/causal_tapir_checkpoint.npy",
                file=f"causal_tapir_checkpoint",
                dir=download_dir,
            )

        ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
        params, state = ckpt_state["params"], ckpt_state["state"]

        online_init = hk.transform_with_state(Tapnet.build_online_model_init)
        online_init_apply = jax.jit(online_init.apply)

        online_predict = hk.transform_with_state(Tapnet.build_online_model_predict)
        online_predict_apply = jax.jit(online_predict.apply)

        rng = jax.random.PRNGKey(42)
        self.online_init_apply = functools.partial(
            online_init_apply, params=params, state=state, rng=rng
        )
        self.online_predict_apply = functools.partial(
            online_predict_apply, params=params, state=state, rng=rng
        )
        self.resize_height, self.resize_width = 256, 256

        self.annotator = annotator
        self.box_annotator = sv.BoxAnnotator()

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

    def get_query_features(self, im, resized_im: np.ndarray, select_points):
        height, width = im.shape[0], im.shape[1]
        query_points = Tapnet.convert_select_points_to_query_points(0, select_points)
        query_points = transforms.convert_grid_coordinates(
            query_points,
            (1, height, width),
            (1, self.resize_height, self.resize_width),
            coordinate_format="tyx",
        )
        query_features, _ = self.online_init_apply(
            frames=Tapnet.preprocess_frames(resized_im),
            query_points=query_points[None],
        )
        causal_state = Tapnet.construct_initial_causal_state(
            query_points.shape[0], len(query_features.resolutions) - 1
        )
        return query_features, causal_state

    def stream_inference(
        self,
        source,
        # interest point
        timer: List = [],
        poi: Literal["point", "box", "text", ""] = "",
        roi: Literal["rect", ""] = "",
        points: List = [],
        startFrame: int = 0,
        # panels
        show=False,
        plot=False,
        save=False,
        save_dirs="",
        verbose=True,
    ) -> Result:  # type: ignore

        if save:
            save_dirs = Path(get_time() if not save_dirs else save_dirs)
            save_dirs.mkdir(exist_ok=True)

        profilers = (ops.Profile(), ops.Profile(), ops.Profile())  # count the time
        self.setup_source(source)

        current_source, index = None, 0

        for _, batch in enumerate(self.dataset):
            path, im0s, vid_cap, s = batch
            is_image = s.startswith("image")

            index += 1

            if startFrame >= index:
                continue

            p, im, im0 = Path(path[0]), im0s[0], im0s[0].copy()

            height, width = im.shape[0], im.shape[1]
            resized_im = media.resize_image(im, (self.resize_height, self.resize_width))
            resized_im = np.expand_dims(resized_im, axis=[0, 1])

            # reset tracker when source changed
            if current_source is None:
                current_source = p
                query_features, causal_state = self.get_query_features(
                    im, resized_im, self.points
                )
            elif current_source != p:
                query_features, causal_state = self.get_query_features(
                    im, resized_im, self.points
                )
                current_source = p
                index = 1

            # multi object detection (detect only person)
            with profilers[0]:
                (prediction, causal_state), _ = self.online_predict_apply(
                    frames=resized_im,
                    query_features=query_features,
                    causal_context=causal_state,
                )
                tracks = prediction["tracks"][0]
                occlusions = prediction["occlusion"][0]
                expected_dist = prediction["expected_dist"][0]
                visibles = Tapnet.postprocess_occlusions(occlusions, expected_dist)
                tracks = transforms.convert_grid_coordinates(
                    tracks, (self.resize_width, self.resize_height), (width, height)
                )

            if plot:
                for i, t in enumerate(tracks):
                    if visibles[i]:
                        x, y = t[0][0], t[0][1]
                        cv2.circle(im, (int(x), int(y)), 8, (0, 0, 255), -1)

            result = Result(
                im=im,
                tracks=tracks.tolist(),  # detections.xyxy,
                speed={
                    "tracks": profilers[0].dt * 1e3 / 1,
                },
                save_dirs=str(save_dirs) if save else "",
                name=p.name,
            )

            yield result

            if verbose:
                LOGGER.info(
                    f"{s}Tracking {len(tracks)} point(s) ({colorstr('green', f'{profilers[0].dt * 1E3:.1f}ms')})"
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
        points: List = [],
        startFrame: int = 0,
        # panels
        show=False,
        plot=False,
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
                startFrame=startFrame,
                points=points,
                show=show,
                plot=plot,
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
                    points=points,
                    startFrame=startFrame,
                    show=show,
                    plot=plot,
                    save=save,
                    save_dirs=save_dirs,
                    verbose=verbose,
                )
            )

    @staticmethod
    def build_online_model_init(frames, query_points):
        """Initialize query features for the query points."""
        model = tapir_model.TAPIR(
            use_causal_conv=True, bilinear_interp_with_depthwise_conv=False
        )

        feature_grids = model.get_feature_grids(frames, is_training=False)
        query_features = model.get_query_features(
            frames,
            is_training=False,
            query_points=query_points,
            feature_grids=feature_grids,
        )
        return query_features

    @staticmethod
    def build_online_model_predict(frames, query_features, causal_context):
        """Compute point tracks and occlusions given frames and query points."""
        model = tapir_model.TAPIR(
            use_causal_conv=True, bilinear_interp_with_depthwise_conv=False
        )
        feature_grids = model.get_feature_grids(frames, is_training=False)
        trajectories = model.estimate_trajectories(
            frames.shape[-3:-1],
            is_training=False,
            feature_grids=feature_grids,
            query_features=query_features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=causal_context,
            get_causal_context=True,
        )
        causal_context = trajectories["causal_context"]
        del trajectories["causal_context"]
        return {k: v[-1] for k, v in trajectories.items()}, causal_context

    @staticmethod
    def preprocess_frames(frames):
        """Preprocess frames to model inputs.

        Args:
        frames: [num_frames, height, width, 3], [0, 255], np.uint8

        Returns:
        frames: [num_frames, height, width, 3], [-1, 1], np.float32
        """
        frames = frames.astype(np.float32)
        frames = frames / 255 * 2 - 1
        return frames

    @staticmethod
    def postprocess_occlusions(occlusions, expected_dist):
        """Postprocess occlusions to boolean visible flag.

        Args:
        occlusions: [num_points, num_frames], [-inf, inf], np.float32

        Returns:
        visibles: [num_points, num_frames], bool
        """
        pred_occ = jax.nn.sigmoid(occlusions)
        pred_occ = 1 - (1 - pred_occ) * (1 - jax.nn.sigmoid(expected_dist))
        visibles = pred_occ < 0.5  # threshold
        return visibles

    @staticmethod
    def sample_random_points(frame_max_idx, height, width, num_points):
        """Sample random points with (time, height, width) order."""
        y = np.random.randint(0, height, (num_points, 1))
        x = np.random.randint(0, width, (num_points, 1))
        t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
        points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
        return points

    @staticmethod
    def construct_initial_causal_state(num_points, num_resolutions):
        value_shapes = {
            "tapir/~/pips_mlp_mixer/block_1_causal_1": (1, num_points, 2, 512),
            "tapir/~/pips_mlp_mixer/block_1_causal_2": (1, num_points, 2, 2048),
            "tapir/~/pips_mlp_mixer/block_2_causal_1": (1, num_points, 2, 512),
            "tapir/~/pips_mlp_mixer/block_2_causal_2": (1, num_points, 2, 2048),
            "tapir/~/pips_mlp_mixer/block_3_causal_1": (1, num_points, 2, 512),
            "tapir/~/pips_mlp_mixer/block_3_causal_2": (1, num_points, 2, 2048),
            "tapir/~/pips_mlp_mixer/block_4_causal_1": (1, num_points, 2, 512),
            "tapir/~/pips_mlp_mixer/block_4_causal_2": (1, num_points, 2, 2048),
            "tapir/~/pips_mlp_mixer/block_5_causal_1": (1, num_points, 2, 512),
            "tapir/~/pips_mlp_mixer/block_5_causal_2": (1, num_points, 2, 2048),
            "tapir/~/pips_mlp_mixer/block_6_causal_1": (1, num_points, 2, 512),
            "tapir/~/pips_mlp_mixer/block_6_causal_2": (1, num_points, 2, 2048),
            "tapir/~/pips_mlp_mixer/block_7_causal_1": (1, num_points, 2, 512),
            "tapir/~/pips_mlp_mixer/block_7_causal_2": (1, num_points, 2, 2048),
            "tapir/~/pips_mlp_mixer/block_8_causal_1": (1, num_points, 2, 512),
            "tapir/~/pips_mlp_mixer/block_8_causal_2": (1, num_points, 2, 2048),
            "tapir/~/pips_mlp_mixer/block_9_causal_1": (1, num_points, 2, 512),
            "tapir/~/pips_mlp_mixer/block_9_causal_2": (1, num_points, 2, 2048),
            "tapir/~/pips_mlp_mixer/block_10_causal_1": (1, num_points, 2, 512),
            "tapir/~/pips_mlp_mixer/block_10_causal_2": (1, num_points, 2, 2048),
            "tapir/~/pips_mlp_mixer/block_11_causal_1": (1, num_points, 2, 512),
            "tapir/~/pips_mlp_mixer/block_11_causal_2": (1, num_points, 2, 2048),
            "tapir/~/pips_mlp_mixer/block_causal_1": (1, num_points, 2, 512),
            "tapir/~/pips_mlp_mixer/block_causal_2": (1, num_points, 2, 2048),
        }
        fake_ret = {k: jnp.zeros(v, dtype=jnp.float32) for k, v in value_shapes.items()}
        return [fake_ret] * num_resolutions * 4

    @staticmethod
    def convert_select_points_to_query_points(frame, points):
        """Convert select points to query points.

        Args:
        points: [num_points, 2], [t, y, x]
        Returns:
        query_points: [num_points, 3], [t, y, x]
        """
        points = np.stack(points)
        query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
        query_points[:, 0] = frame
        query_points[:, 1] = points[:, 1]
        query_points[:, 2] = points[:, 0]
        return query_points

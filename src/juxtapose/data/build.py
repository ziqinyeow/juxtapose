import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from juxtapose.data.loaders import (
    LOADERS,
    LoadImages,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)

from juxtapose.data.utils import IMG_FORMATS, VID_FORMATS
from juxtapose.utils import RANK, colorstr
from juxtapose.utils.checks import check_file


def check_source(source):
    """Check source type and return corresponding flag values."""
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(
            ("https://", "http://", "rtsp://", "rtmp://")
        )
        webcam = (
            source.isnumeric()
            or source.endswith(".streams")
            or (is_url and not is_file)
        )
        screenshot = source.lower() == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, tuple(LOADERS)):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError(
            "Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict"
        )

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(source=None, imgsz=640, vid_stride=1):
    """
    Loads an inference source for object detection and applies necessary transformations.

    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference.
        imgsz (int, optional): The size of the image for inference. Default is 640.
        vid_stride (int, optional): The frame interval for video sources. Default is 1.

    Returns:
        dataset (Dataset): A dataset object for the specified input source.
    """
    source, webcam, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = (
        source.source_type
        if in_memory
        else SourceTypes(webcam, screenshot, from_img, tensor)
    )

    # Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif webcam:
        dataset = LoadStreams(source, imgsz=imgsz, vid_stride=vid_stride)
    elif screenshot:
        dataset = LoadScreenshots(source, imgsz=imgsz)
    elif from_img:
        dataset = LoadPilAndNumpy(source, imgsz=imgsz)
    else:
        dataset = LoadImages(source, imgsz=imgsz, vid_stride=vid_stride)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset

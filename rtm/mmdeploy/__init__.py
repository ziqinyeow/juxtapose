# Copyright (c) OpenMMLab. All rights reserved.
import importlib

from rtm.mmdeploy.utils import get_root_logger
from .version import __version__, version_info  # noqa F401

if importlib.util.find_spec("torch"):
    importlib.import_module("rtm.mmdeploy.pytorch")
else:
    logger = get_root_logger()
    logger.debug("torch is not installed.")

if importlib.util.find_spec("mmcv"):
    importlib.import_module("rtm.mmdeploy.mmcv")
else:
    logger = get_root_logger()
    logger.debug("mmcv is not installed.")

# Copyright (c) OpenMMLab. All rights reserved.
from rtm.mmdeploy.backend.tensorrt import is_available
from ..core import PIPELINE_MANAGER

__all__ = ["is_available"]

if is_available():
    from rtm.mmdeploy.backend.tensorrt import from_onnx as _from_onnx
    from rtm.mmdeploy.backend.tensorrt import load, save

    from_onnx = PIPELINE_MANAGER.register_pipeline()(_from_onnx)
    __all__ += ["from_onnx", "save", "load"]
    try:
        from rtm.mmdeploy.backend.tensorrt.onnx2tensorrt import (
            onnx2tensorrt as _onnx2tensorrt,
        )

        onnx2tensorrt = PIPELINE_MANAGER.register_pipeline()(_onnx2tensorrt)
        __all__ += ["onnx2tensorrt"]
    except Exception:
        pass

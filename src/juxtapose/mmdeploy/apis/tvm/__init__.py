# Copyright (c) OpenMMLab. All rights reserved.
from juxtapose.mmdeploy.backend.tvm import get_library_ext, is_available
from ..core import PIPELINE_MANAGER

__all__ = ["is_available", "get_library_ext"]

if is_available():
    from juxtapose.mmdeploy.backend.tvm import HDF5Dataset
    from juxtapose.mmdeploy.backend.tvm import from_onnx as _from_onnx

    from_onnx = PIPELINE_MANAGER.register_pipeline()(_from_onnx)

    __all__ += ["from_onnx", "HDF5Dataset"]

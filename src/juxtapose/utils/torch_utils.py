import torch

from juxtapose.utils.checks import check_version


TORCH_1_9 = check_version(torch.__version__, "1.9.0")
TORCH_1_11 = check_version(torch.__version__, "1.11.0")
TORCH_1_12 = check_version(torch.__version__, "1.12.0")
TORCH_2_0 = check_version(torch.__version__, "2.0.0")


def smart_inference_mode(fn):
    """Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator."""

    def decorate(*args, **kwargs):
        """Applies appropriate torch decorator for inference mode based on torch version."""
        return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)(
            *args, **kwargs
        )

    return decorate

import contextlib
import time
import re

import torch
import numpy as np


class Profile(contextlib.ContextDecorator):
    """
    POSE Profile class.
    Usage: as a decorator with @Profile() or as a context manager with 'with Profile():'
    """

    def __init__(self, t=0.0):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
        """
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        """
        Start timing.
        """
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """
        Stop timing.
        """
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """
        Get current time.
        """
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


def clean_str(s):
    """
    Cleans a string by replacing special characters with underscore _

    Args:
      s (str): a string needing special characters replaced

    Returns:
      (str): a string with special characters replaced by an underscore _
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def xyxy2xyxyxyxy(xyxy):
    """
    xyxy: [(x1, y1, x2, y2), ...]
    """
    xyxyxyxy = []
    for _xyxy in xyxy:
        x1, y1, x2, y2 = _xyxy

        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)

        xyxyxyxy.append(
            np.array([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
        )
    return xyxyxyxy

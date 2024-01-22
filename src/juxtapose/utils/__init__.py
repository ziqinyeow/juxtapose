import contextlib
import inspect
import logging.config
import os
import platform
import re
import subprocess
import sys
import threading
import urllib
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from juxtapose import __version__

# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

# Other Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
# DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
NUM_THREADS = min(
    8, max(1, os.cpu_count() - 1)
)  # number of YOLOv5 multiprocessing threads
AUTOINSTALL = (
    str(os.getenv("JUXTAPOSE_AUTOINSTALL", True)).lower() == "true"
)  # global auto-install mode
VERBOSE = (
    str(os.getenv("JUXTAPOSE_VERBOSE", True)).lower() == "true"
)  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
LOGGING_NAME = "JUXTAPOSE"
MACOS, LINUX, WINDOWS = (
    platform.system() == x for x in ["Darwin", "Linux", "Windows"]
)  # environment booleans
ARM64 = platform.machine() in ("arm64", "aarch64")  # ARM64 booleans


# Settings
torch.set_printoptions(linewidth=320, precision=4, profile="default")
np.set_printoptions(
    linewidth=320, formatter={"float_kind": "{:11.5g}".format}
)  # format short g, %precision=5
cv2.setNumThreads(
    0
)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr max threads
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for deterministic training
os.environ[
    "TF_CPP_MIN_LOG_LEVEL"
] = "2"  # suppress verbose TF compiler warnings in Colab


class SimpleClass:
    """
    POSE SimpleClass is a base class providing helpful string representation, error reporting, and attribute
    access methods for easier debugging and usage.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return (
            f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n"
            + "\n".join(attr)
        )

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}"
        )


class IterableSimpleNamespace(SimpleNamespace):
    """
    POSE IterableSimpleNamespace is an extension class of SimpleNamespace that adds iterable functionality and
    enables usage with dict() and for loops.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date JUXTAPOSE
            'default.yaml' file.\nPlease update your code with 'pip install -U juxtapose' and if necessary replace
            with the latest version from
            """
        )

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)


def plt_settings(rcparams=None, backend="Agg"):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Usage:
        decorator: @plt_settings({"font.size": 12})
        context manager: with plt_settings({"font.size": 12}):

    Args:
        rcparams (dict): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend. This decorator can be
            applied to any function that needs to have specific matplotlib rc parameters and backend for its execution.
    """

    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Sets rc parameters and backend, calls the original function, and restores the settings."""
            original_backend = plt.get_backend()
            plt.switch_backend(backend)

            with plt.rc_context(rcparams):
                result = func(*args, **kwargs)

            plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


def set_logging(name=LOGGING_NAME, verbose=True):
    """Sets up logging for the given name."""
    rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {name: {"format": "%(message)s"}},
            "handlers": {
                name: {
                    "class": "logging.StreamHandler",
                    "formatter": name,
                    "level": level,
                }
            },
            "loggers": {name: {"level": level, "handlers": [name], "propagate": False}},
        }
    )


def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


class EmojiFilter(logging.Filter):
    """
    A custom logging filter class for removing emojis in log messages.

    This filter is particularly useful for ensuring compatibility with Windows terminals
    that may not support the display of emojis in log messages.
    """

    def filter(self, record):
        """Filter logs by emoji unicode characters on windows."""
        record.msg = emojis(record.msg)
        return super().filter(record)


# Set logger
set_logging(LOGGING_NAME, verbose=VERBOSE)  # run before defining LOGGER
LOGGER = logging.getLogger(
    LOGGING_NAME
)  # define globally (used in train.py, val.py, detect.py, etc.)
if WINDOWS:  # emoji-safe logging
    LOGGER.addFilter(EmojiFilter())


class ThreadingLocked:
    """
    A decorator class for ensuring thread-safe execution of a function or method.
    This class can be used as a decorator to make sure that if the decorated function
    is called from multiple threads, only one thread at a time will be able to execute the function.

    Attributes:
        lock (threading.Lock): A lock object used to manage access to the decorated function.

    Usage:
        @ThreadingLocked()
        def my_function():
            # Your code here
            pass
    """

    def __init__(self):
        self.lock = threading.Lock()

    def __call__(self, f):
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            with self.lock:
                return f(*args, **kwargs)

        return decorated


def yaml_save(file="data.yaml", data=None):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.

    Returns:
        (None): Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    for k, v in data.items():
        if isinstance(v, Path):
            data[k] = str(v)

    # Dump data to file in YAML format
    with open(file, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(
                r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+",
                "",
                s,
            )

        # Add YAML filename to dict and return
        return (
            {**yaml.safe_load(s), "yaml_file": str(file)}
            if append_filename
            else yaml.safe_load(s)
        )


def yaml_print(yaml_file: Union[str, Path, dict]) -> None:
    """
    Pretty prints a yaml file or a yaml-formatted dictionary.

    Args:
        yaml_file: The file path of the yaml file or a yaml-formatted dictionary.

    Returns:
        None
    """
    yaml_dict = (
        yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    )
    dump = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True)
    LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


# Default configuration
# DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
# for k, v in DEFAULT_CFG_DICT.items():
#     if isinstance(v, str) and v.lower() == "none":
#         DEFAULT_CFG_DICT[k] = None
# DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
# DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def is_colab():
    """
    Check if the current script is running inside a Google Colab notebook.

    Returns:
        (bool): True if running inside a Colab notebook, False otherwise.
    """
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ


def is_kaggle():
    """
    Check if the current script is running inside a Kaggle kernel.

    Returns:
        (bool): True if running inside a Kaggle kernel, False otherwise.
    """
    return (
        os.environ.get("PWD") == "/kaggle/working"
        and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"
    )


def is_jupyter():
    """
    Check if the current script is running inside a Jupyter Notebook.
    Verified on Colab, Jupyterlab, Kaggle, Paperspace.

    Returns:
        (bool): True if running inside a Jupyter Notebook, False otherwise.
    """
    with contextlib.suppress(Exception):
        from IPython import get_ipython

        return get_ipython() is not None
    return False


def is_docker() -> bool:
    """
    Determine if the script is running inside a Docker container.

    Returns:
        (bool): True if the script is running inside a Docker container, False otherwise.
    """
    file = Path("/proc/self/cgroup")
    if file.exists():
        with open(file) as f:
            return "docker" in f.read()
    else:
        return False


def is_online() -> bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    import socket

    for host in "1.1.1.1", "8.8.8.8", "223.5.5.5":  # Cloudflare, Google, AliDNS:
        try:
            test_connection = socket.create_connection(address=(host, 53), timeout=2)
        except (socket.timeout, socket.gaierror, OSError):
            continue
        else:
            # If the connection was successful, close it to avoid a ResourceWarning
            test_connection.close()
            return True
    return False


ONLINE = is_online()


def is_pip_package(filepath: str = __name__) -> bool:
    """
    Determines if the file at the given filepath is part of a pip package.

    Args:
        filepath (str): The filepath to check.

    Returns:
        (bool): True if the file is part of a pip package, False otherwise.
    """
    import importlib.util

    # Get the spec for the module
    spec = importlib.util.find_spec(filepath)

    # Return whether the spec is not None and the origin is not None (indicating it is a package)
    return spec is not None and spec.origin is not None


def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)


def is_pytest_running():
    """
    Determines whether pytest is currently running or not.

    Returns:
        (bool): True if pytest is running, False otherwise.
    """
    return (
        ("PYTEST_CURRENT_TEST" in os.environ)
        or ("pytest" in sys.modules)
        or ("pytest" in Path(sys.argv[0]).stem)
    )


def is_github_actions_ci() -> bool:
    """
    Determine if the current environment is a GitHub Actions CI Python runner.

    Returns:
        (bool): True if the current environment is a GitHub Actions CI Python runner, False otherwise.
    """
    return (
        "GITHUB_ACTIONS" in os.environ
        and "RUNNER_OS" in os.environ
        and "RUNNER_TOOL_CACHE" in os.environ
    )


def is_git_dir():
    """
    Determines whether the current file is part of a git repository.
    If the current file is not part of a git repository, returns None.

    Returns:
        (bool): True if current file is part of a git repository.
    """
    return get_git_dir() is not None


def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory.
    If the current file is not part of a git repository, returns None.

    Returns:
        (Path | None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / ".git").is_dir():
            return d
    return None  # no .git dir found


def get_git_origin_url():
    """
    Retrieves the origin URL of a git repository.

    Returns:
        (str | None): The origin URL of the git repository.
    """
    if is_git_dir():
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"]
            )
            return origin.decode().strip()
    return None  # if not git dir or on error


def get_git_branch():
    """
    Returns the current git branch name. If not in a git repository, returns None.

    Returns:
        (str | None): The current git branch name.
    """
    if is_git_dir():
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"]
            )
            return origin.decode().strip()
    return None  # if not git dir or on error


def get_default_args(func):
    """Returns a dictionary of default arguments for a function.

    Args:
        func (callable): The function to inspect.

    Returns:
        (dict): A dictionary where each key is a parameter name, and each value is the default value of that parameter.
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_user_config_dir(sub_dir="juxtapose"):
    """
    Get the user config directory.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    # Return the appropriate config directory for each operating system
    if WINDOWS:
        path = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:  # macOS
        path = Path.home() / "Library" / "Application Support" / sub_dir
    elif LINUX:
        path = Path.home() / ".config" / sub_dir
    else:
        raise ValueError(f"Unsupported operating system: {platform.system()}")

    # GCP and AWS lambda fix, only /tmp is writeable
    if not is_dir_writeable(str(path.parent)):
        path = Path("/tmp") / sub_dir
        LOGGER.warning(
            f"WARNING ⚠️ user config directory is not writeable, defaulting to '{path}'."
        )

    # Create the subdirectory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path


USER_CONFIG_DIR = Path(
    os.getenv("YOLO_CONFIG_DIR", get_user_config_dir())
)  # POSE settings dir
SETTINGS_YAML = USER_CONFIG_DIR / "settings.yaml"


def colorstr(*input):
    """Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')."""
    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


class TryExcept(contextlib.ContextDecorator):
    """YOLOv8 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager."""

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """Executes when entering TryExcept context, initializes instance."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Defines behavior when exiting a 'with' block, prints error message if necessary."""
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    """Multi-threads a target function and returns thread. Usage: @threaded decorator."""

    def wrapper(*args, **kwargs):
        """Multi-threads a given function and returns the thread."""
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def update_dict_recursive(d, u):
    """
    Recursively updates the dictionary `d` with the key-value pairs from the dictionary `u` without overwriting
    entire sub-dictionaries. Note that function recursion is intended and not a problem, as this allows for updating
    nested dictionaries at any arbitrary depth.

    Args:
        d (dict): The dictionary to be updated.
        u (dict): The dictionary to update `d` with.

    Returns:
        (dict): The recursively updated dictionary.
    """
    for k, v in u.items():
        d[k] = update_dict_recursive(d.get(k, {}), v) if isinstance(v, dict) else v
    return d


def deprecation_warn(arg, new_arg, version=None):
    """Issue a deprecation warning when a deprecated argument is used, suggesting an updated argument."""
    if not version:
        version = float(__version__[:3]) + 0.2  # deprecate after 2nd major release
    LOGGER.warning(
        f"WARNING ⚠️ '{arg}' is deprecated and will be removed in 'juxtapose {version}' in the future. "
        f"Please use '{new_arg}' instead."
    )


def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = (
        Path(url).as_posix().replace(":/", "://")
    )  # Pathlib turns :// -> :/, as_posix() for Windows
    return urllib.parse.unquote(url).split("?")[
        0
    ]  # '%2F' to '/', split https://url.com/file.txt?auth


def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name


def get_time():
    # current date and time
    return datetime.now().strftime("%Y%m%d-%H%M%S")


DEFAULT_COLOR_PALETTE = [
    "#a351fb",
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]


# Run below code on utils init ------------------------------------------------------------------------------------

# Check first-install steps
PREFIX = colorstr("JUXTAPOSE: ")
ENVIRONMENT = (
    "Colab"
    if is_colab()
    else "Kaggle"
    if is_kaggle()
    else "Jupyter"
    if is_jupyter()
    else "Docker"
    if is_docker()
    else platform.system()
)
TESTS_RUNNING = is_pytest_running() or is_github_actions_ci()

# Apply monkey patches if the script is being run from within the parent directory of the script's location
from .patches import imread, imshow, imwrite

# torch.save = torch_save
if (
    Path(inspect.stack()[0].filename).parent.parent.as_posix()
    in inspect.stack()[-1].filename
):
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow

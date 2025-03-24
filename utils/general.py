import contextlib
import logging
import logging.config
import math
import os
import platform
import re
import time
from datetime import datetime
import random

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg

from pathlib import Path
from utils import emojis
from torchvision import transforms

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
RANK = int(os.getenv("RANK", -1))

# Settings
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of model multiprocessing threads

torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
os.environ['OMP_NUM_THREADS'] = "1" if platform.system() == "darwin" else str(NUM_THREADS)  # OpenMP (PyTorch and Scipy)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['TORCH_CPP_LOG_LEVEL'] = "ERROR"  # suppress "NNPACK.cpp could not initialize NNPACK" warning
os.environ['KINETO_LOG_LEVEL'] = "5"  # suppress verbose PyTorch profiler output computing FLOPS

LOGGING_NAME = "L2CS-Net"


def set_logging(name=LOGGING_NAME, verbose=True):
    """Configures logging with specified verbosity; `name` sets the logger's name, `verbose` controls logging level."""
    rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU training
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_logger': False,
            'formatters': {name: {'format': '%(message)s'}},
            'handlers': {
                name: {
                    'class': 'logging.StreamHandler',
                    'formatter': name,
                    'level': level,
                }
            },
            'loggers': {
                name: {
                    'level': level,
                    'handlers': [name],
                    'propagate': False,
                }
            }
        }
    )


set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)
if platform.system() == 'Windows':
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))

transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def prep_input_numpy(img: np.ndarray, device: str):
    """Preparing a Numpy Array as input to L2CS-Net."""

    if len(img.shape) == 4:
        imgs = []
        for im in img:
            imgs.append(transformations(im))
        img = torch.stack(imgs)
    else:
        img = transformations(img)

    img = img.to(device)

    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    return img


def check_version(current="0.0.0", minimum="0.0.0", name="version", pinned=False, hard=False, verbose=False):
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)
    s = f"WARNING {name}{minimum} is required by L2CS-Net, but {name}{current} is currently installed."
    if hard:
        assert result, emojis(s)
    if verbose and not result:
        # LOGGER.warning(s)
        pass
    return result


def file_date(path=__file__):
    """Returns a human-readable file modification date in 'YYYY-M-D' format, given a file path."""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def init_seeds(seed=42, deterministic=False):
    """
    Initializes RNG seeds and sets deterministic options if specified.

    See https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    if deterministic and check_version(torch.__version__, "1.12.0"):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order.
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def gazeto3d(gaze):
    gaze_gt = torch.zeros([3])
    gaze_gt[0] = -torch.cos(gaze[1]) * torch.sin(gaze[0])
    gaze_gt[1] = -torch.sin(gaze[1])
    gaze_gt[2] = -torch.cos(gaze[1]) * torch.cos(gaze[0])
    return gaze_gt


def angular(gaze, label):
    total = torch.sum(gaze * label)
    return torch.arccos(torch.clamp(total / torch.norm(gaze) * torch.norm(label), max=0.9999999)) * 180 / torch.pi


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Generates an incremented file or directory path if it exists, with optional mkdir; args: path, exist_ok=False,
    sep="", mkdir=False.

    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def check_imshow(warn=False):
    """Checks environment support for image display; warns on failure if `warn=True`."""
    try:
        cv2.imshow("test", np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f"WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}")
        return False


def clean_str(s):
    """Cleans a string by replacing special characters with underscore, e.g., `clean_str('#example!')` returns
    '_example_'.
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


class Profile(contextlib.ContextDecorator):
    def __init__(self, t=0.0, device: torch.device = None):
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """Initializes timing at the start of a profiling context block for performance measurement."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """Concludes timing, updating duration for profiling upon exiting a context block."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """Measures and returns the current time, synchronizing CUDA operations if `cuda` is True."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """
    Generates a lambda for a sinusoidal ramp from y1 to y2 over 'steps'.
    See https://arxiv.org/pdf/1812.01187.pdf for details.
    """
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def mean_gaze(gaze_pred):
    mean_pred1 = gaze_pred.mean(dim=[2, 3])
    # mean_pred2 = gaze_pred[1].mean(dim=[2, 3])
    # mean_pred3 = gaze_pred[2].mean(dim=[2, 3])

    means = [mean_pred1]
    global_mean = torch.mean(torch.stack(means, dim=0), dim=0)  # 结果形状为 [batch_size, num_channels, 2]
    global_mean = torch.mean(global_mean, dim=1)
    return global_mean


def gazeto3d_matrix(gaze):
    """
    输入: 形状为 [..., 2] 的视线角度 (pitch, yaw)
    输出: 形状为 [..., 3] 的 3D 方向向量
    """
    pitch = gaze[..., 0]
    yaw = gaze[..., 1]

    pitch = pitch * torch.pi / 180
    yaw = yaw * torch.pi / 180

    x = -torch.cos(yaw) * torch.sin(pitch)
    y = -torch.sin(yaw)
    z = -torch.cos(yaw) * torch.cos(pitch)

    gaze_3d = torch.stack([x, y, z], dim=-1)
    return gaze_3d


def angular_matrix(gaze, label):
    """
    输入:
    - gaze: 形状为 [..., 3] 的预测 3D 方向向量
    - label: 形状为 [..., 3] 的真实 3D 方向向量
    输出:
    - 形状为 [...] 的角度误差
    """
    # 计算内积
    total = torch.sum(gaze * label, dim=-1)

    # 计算模长
    gaze_norm = torch.norm(gaze, dim=-1)
    label_norm = torch.norm(label, dim=-1)

    # 计算角度误差
    cos_theta = total / (gaze_norm * label_norm)
    cos_theta = torch.clamp(cos_theta, max=0.9999999)     # 防止 arccos 的输入超出范围
    angular_error = torch.arccos(cos_theta) * 180 / torch.pi - 1.0
    return angular_error


def angular_error(gaze_2d, label_2d):
    gaze_3d = gazeto3d_matrix(gaze_2d)
    label_3d = gazeto3d_matrix(label_2d.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(gaze_2d))
    angular_error_matrix = angular_matrix(gaze_3d, label_3d)
    return angular_error_matrix


# if __name__ == '__main__':
#     gaze_pred = torch.tensor([[42.0, 42]])
#     gaze_label = torch.tensor([[-42.0, -42.0]])
#
#     gaze_pred_3d = gazeto3d_matrix(gaze_pred)
#     gaze_label_3d = gazeto3d_matrix(gaze_label)
#
#     angular_error = angular_matrix(gaze_pred_3d, gaze_label_3d)
#     print(f"最大角度误差: {angular_error.item()}°")
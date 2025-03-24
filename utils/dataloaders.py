import glob
import math
import os
import random
import time
from threading import Thread

import cv2
import numpy as np

import torch
from torch.utils.data import dataloader, distributed, Dataset, DataLoader
from utils.torch_utils import torch_distributed_zero_first
from torchvision import transforms
from PIL import Image, ImageFile
from pathlib import Path
from urllib.parse import urlparse
from utils.general import LOGGER

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", -1))
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # globel pin_memory for dataloaders

IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        """Initializes an InfiniteDataLoader that reuses workers with standard DataLoader syntax, augmenting with a
        repeating sampler.
        """
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler in the InfiniteDataLoader."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Yields batches of data indefinitely in a loop by resetting the sampler when exhausted."""
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        """Initializes a perpetual sampler wrapping a provided `Sampler` instance for endless data iteration."""
        self.sampler = sampler

    def __iter__(self):
        """Returns an infinite iterator over the dataset by repeatedly yielding from the given sampler."""
        while True:
            yield from iter(self.sampler)


# Inherit from distributedSampler and override iterator
# https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py
class SmartDistributedSampler(distributed.DistributedSampler):
    def __iter__(self):
        """ Yields indices for distributed data sampling, shuffled deterministically based on epoch and seed."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # determine the eventual size (n) of self.indices (DDP indices)
        n = int((len(self.dataset) - self.rank - 1) / self.num_replicas) + 1  # num_replices == WORLD_SIZE
        idx = torch.randperm(n, generator=g)
        if not self.shuffle:
            idx = idx.sort()[0]

        idx = idx.tolist()
        if self.drop_last:
            idx = idx[: self.num_samples]
        else:
            padding_size = self.num_samples - len(idx)
            if padding_size <= len(idx):
                idx += idx[:padding_size]
            else:
                idx += (idx * math.ceil(padding_size / len(idx)))[:padding_size]

        return iter(idx)


def seed_worker(work):
    """
    Sets the seed for a dataloader worker to ensure reproducibility, based on Pytorch's randomness notes.
    See https://pytorch.org/docs/stable/notes/randomness.html#dataloders.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Gaze360(Dataset):
    """ Dataset for Gaze360 """

    def __init__(self, path, root, transform, angle, binwidth, train=True, scaler=None):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.angle = angle
        if train == False:
            angle = 90
        self.binwidth = binwidth
        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[5]
                    label = np.array(gaze2d.split(',')).astype("float")
                    if abs((label[0] * 180 / np.pi)) < angle and abs((label[1] * 180 / np.pi)) < angle:
                        self.lines.append(line)
        LOGGER.info(
            "{} items removed from dataset that have an angle > {}".format(self.orig_list_len - len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        face = line[0]
        lefteye = line[1]
        righteye = line[2]
        name = line[3]
        gaze2d = line[5]
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0] * 180 / np.pi
        yaw = label[1] * 180 / np.pi

        img = Image.open(os.path.join(self.root, face))

        if self.transform:
            img = self.transform(img)

        # Bin values
        bins = np.array(range(-1 * self.angle, self.angle, self.binwidth))
        binned_pose = np.digitize([pitch, yaw], bins) - 1

        label = binned_pose
        cont_label = torch.FloatTensor([pitch, yaw])

        return img, label, cont_label, name

    @staticmethod
    def collate_fn(batch):
        imgs, labels, cont_labels, names = tuple(zip(*batch))

        imgs = torch.stack(imgs, dim=0)
        labels = torch.as_tensor(labels)
        cont_labels = torch.as_tensor(cont_labels)

        return imgs, labels, cont_labels, names


class Mpiigaze(Dataset):
    """ Dataset for MpiiGaze """

    def __init__(self, pathorg, root, transform, train, angle, fold=0):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.lines = []
        self.angle = angle
        folder = os.listdir(pathorg)
        folder.sort()
        self.pathorg = [os.path.join(pathorg, f) for f in folder]
        path = self.pathorg.copy()
        if train == True:
            # path.pop(fold)
            pass
        else:
            path = path[fold]
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    lines = f.readlines()
                    lines.pop(0)
                    self.orig_list_len += len(lines)
                    for line in lines:
                        line = line.strip().split(" ")
                        name = line[3]
                        gaze2d = line[7]
                        # head2d = line[8]
                        # lefteye = line[1]
                        # righteye = line[2]
                        face = line[0]
                        label = np.array(gaze2d.split(",")).astype("float")
                        if abs((label[0] * 180 / np.pi)) <= self.angle and abs((label[1] * 180 / np.pi)) <= self.angle:
                            label = torch.from_numpy(label).type(torch.FloatTensor)
                            pitch = label[0] * 180 / np.pi
                            yaw = label[1] * 180 / np.pi
                            label = torch.FloatTensor([pitch, yaw])
                            line = [label, face, name]
                            self.lines.append(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    line = line.strip().split(" ")
                    name = line[3]
                    gaze2d = line[7]
                    # head2d = line[8]
                    # lefteye = line[1]
                    # righteye = line[2]
                    face = line[0]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0] * 180 / np.pi)) <= self.angle and abs((label[1] * 180 / np.pi)) <= self.angle:
                        label = torch.from_numpy(label).type(torch.FloatTensor)
                        pitch = label[0] * 180 / np.pi
                        yaw = label[1] * 180 / np.pi
                        label = torch.FloatTensor([pitch, yaw])
                        line = [label, face, name]
                        self.lines.append(line)

        LOGGER.info(
            "{} items removed from dataset that have an angle > {}".format(self.orig_list_len - len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        label, face, name = line[0], line[1], line[2]
        img = Image.open(os.path.join(self.root, face))
        if self.transform:
            img = self.transform(img)
        return img, label, name

    @staticmethod
    def collate_fn(batch):
        imgs, labels, names = tuple(zip(*batch))

        imgs = torch.stack(imgs, dim=0)
        labels = torch.stack(labels, dim=0)

        return imgs, labels, names


class MpiigazeKFold(Dataset):
    """ KFold Dataset for mpiigaze """

    def __init__(self, pathorg, root, transform, train, angle, fold=0, scaler=False):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.labels = []
        self.faces = []
        folder = os.listdir(pathorg)
        folder.sort()
        self.pathorg = [os.path.join(pathorg, f) for f in folder]
        path = self.pathorg.copy()
        if train == True:
            path.pop(fold)
            pass
        else:
            path = path[fold]
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    lines = f.readlines()
                    lines.pop(0)
                    self.orig_list_len += len(lines)
                    for line in lines:
                        line = line.strip().split(" ")
                        name = line[3]
                        gaze2d = line[7]
                        # head2d = line[8]
                        # lefteye = line[1]
                        # righteye = line[2]
                        face = line[0]
                        label = np.array(gaze2d.split(",")).astype("float")
                        if abs((label[0] * 180 / np.pi)) <= 42 and abs((label[1] * 180 / np.pi)) <= 42:
                            # label = torch.from_numpy(label).type(torch.FloatTensor)
                            # pitch = label[0] * 180 / np.pi
                            # yaw = label[1] * 180 / np.pi
                            # label = torch.FloatTensor([pitch, yaw])
                            label = label * 180 / np.pi  # 弧度转为度
                            self.labels.append(label)
                            self.faces.append(face)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    line = line.strip().split(" ")
                    name = line[3]
                    gaze2d = line[7]
                    # head2d = line[8]
                    # lefteye = line[1]
                    # righteye = line[2]
                    face = line[0]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0] * 180 / np.pi)) <= 42 and abs((label[1] * 180 / np.pi)) <= 42:
                        label = label * 180 / np.pi
                        self.labels.append(label)
                        self.faces.append(face)
        self.labels = np.array(self.labels)
        self.faces = np.array(self.faces)
        LOGGER.info(
            "{} items removed from dataset that have an angle > {}".format(self.orig_list_len - len(self.labels),
                                                                           angle))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label, face = torch.from_numpy(self.labels[idx]).type(torch.FloatTensor), self.faces[idx]
        img = Image.open(os.path.join(self.root, face).replace('\\', '/'))
        if self.transform:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        imgs, labels = tuple(zip(*batch))

        imgs = torch.stack(imgs, dim=0)
        labels = torch.stack(labels, dim=0)

        return imgs, labels


def create_dataloader(
        root,
        angle,
        batch_size,
        dataset_name,
        path,
        rank=-1,
        workers=8,
        shuffle=False,
        seed=0,
        binwidth=None,
        transform=None,
        train=True,
        fold=0,
        scaler=None,
):
    """ param binwidth for gaze360. Param fold for mpiigaze"""
    with torch_distributed_zero_first(rank):
        if dataset_name == 'Gaze360':
            dataset = eval(dataset_name)(path, root, transform, angle, binwidth, train, scaler)
        else:
            dataset = eval(dataset_name)(path, root, transform, train, angle, fold, scaler)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else SmartDistributedSampler(dataset, shuffle=shuffle)
    loader = InfiniteDataLoader
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=dataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator
    ), dataset


class LoadStreams:
    # streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources="file.streams", auto=True, transforms=None, vid_stride=1):
        """Initializes a stream loader for processing video streams with L2CS, supporting various sources including
        YouTube.
        """
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = "stream"
        self.vid_stride = vid_stride  # video frame-rate stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        from utils.general import clean_str
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f"{i + 1}/{n}: {s}... "
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f"{st}Failed to open {s}"
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float("inf")  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        # new line
        LOGGER.info("")

        # check for common shapes
        self.transforms = transforms  # optional

    def update(self, i, cap, stream):
        """Reads frames from stream `i`, updating imgs array; handles stream reopening on signal loss."""
        n, f = 0, self.frames[i]  # frame number, frame array
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()  # .read() = .grab() followed by .retrieve()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning("WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.")
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time

    def __iter__(self):
        """Resets and returns the iterator for iterating over video frames or images in a dataset."""
        self.count = -1
        return self

    def __next__(self):
        """Iterates over video frames or images, halting on thread stop or 'q' key press, raising `StopIteration` when
        done.
        """
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.imgs.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack(im0)  # resize
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        return self.sources, im, im0, None, ""

    def __len__(self):
        """Returns the number of sources in the dataset, supporting up to 32 streams at 30 FPS over 30 years."""
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadImages:
    """L2CS image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`"""

    def __init__(self, path, transforms=None, vid_stride=1):
        """Initializes L2CS loader for images/videos, supporting glob patterns, directories, and lists of paths."""
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if "*" in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, "*.*"))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f"{p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        """Initializes iterator by resetting count and returns the iterator object itself."""
        self.count = 0
        return self

    def __next__(self):
        """Advances to the next file in the dataset, raising StopIteration if at the end."""
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: "

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f"Image Not Found {path}"
            s = f"image {self.count}/{self.nf} {path}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = im0
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def _new_video(self, path):
        """Initializes a new video capture object with path, frame count adjusted by stride, and orientation
        metadata.
        """
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees

    def _cv2_rotate(self, im):
        """Rotates a cv2 image based on its orientation; supports 0, 90, and 180 degrees rotations."""
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        """Returns the number of files in the dataset."""
        return self.nf  # number of files


if __name__ == "__main__":
    gazeMpiilabel_dir = "../datasets/FaceBased/MPIIFaceGaze/Label"
    gazeMpiimage_dir = "../datasets/FaceBased/MPIIFaceGaze/Image"
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_dataloader, dataset = create_dataloader(
        root=gazeMpiimage_dir,
        angle=42,
        batch_size=16,
        dataset_name="MpiigazeKFold",
        transform=transformations,
        path=gazeMpiilabel_dir,
        fold=0,
        scaler=True
    )
    for i, (images_gaze, labels_gaze) in enumerate(train_dataloader):
        continue

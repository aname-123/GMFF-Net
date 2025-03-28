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


class SmartDistributedSampler(distributed.DistributedSampler):
    def __iter__(self):
        """ Yields indices for distributed data sampling, shuffled deterministically based on epoch and seed."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # determine the eventual size (n) of self.indices (DDP indices)
        n = int((len(self.dataset) - self.rank - 1) / self.num_replicas) + 1
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
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



class MpiigazeKFold(Dataset):
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
                        gaze2d = line[7]
                        face = line[0]
                        label = np.array(gaze2d.split(",")).astype("float")
                        if abs((label[0] * 180 / np.pi)) <= 42 and abs((label[1] * 180 / np.pi)) <= 42:
                            label = label * 180 / np.pi
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
    with torch_distributed_zero_first(rank):
        if dataset_name == 'MpiigazeKFold':
            dataset = eval(dataset_name)(path, root, transform, train, angle, fold, scaler)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
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

import sys
import torch
import torch.nn as nn
import math
from copy import deepcopy
from utils.general import LOGGER
from utils.torch_utils import initialize_weights
from utils.plots import feature_visualization
from models.backbone import *
from models.common import Conv, Add, CEMA

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def parse_model(model_dict):
    nh, nl, pl, dl, backbone = model_dict['nh'], model_dict['nl'], model_dict['pl'], model_dict['dl'], model_dict['backbone']
    backbone = eval(backbone)()
    t = str(backbone)[0:6].replace('__main__', '')
    np = sum([x.numel() for x in backbone.parameters()])
    backbone.i, backbone.f, backbone.type, backbone.np = 0, -1, t, np

    layers, save = [backbone], []
    for i, (f, n, m, args) in enumerate(model_dict['Feature pyramid'], start=1):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except NameError:
                pass
        if m in [Conv, CEMA]:
            if m in [CEMA]:
                args.insert(2, n)
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__', '')
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)

    return nn.Sequential(*layers), sorted(save)


class Head(nn.Module):
    def __init__(self, ch, nh=1, nl=2, pl=7, dl=1024):
        super(Head, self).__init__()
        self.nh = nh
        self.nl = nl
        self.m = nn.Conv2d(ch, self.nh * self.nl, 1)
        self.fc1 = nn.Linear(dl * pl * pl, dl)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(dl, 2)

    def forward(self, x):
        bs, _, ny, nx = x.shape
        x = x.view(bs, -1)
        x = self.fc1(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(bs, 1, 2, 1, 1).permute(0, 1, 3, 4, 2).contiguous()
        return x


class XModel(nn.Module):
    def __init__(self, cfg='resnet50-GMSFF-EMABo_v4.yaml', nh=1, nl=2, pl=7, dl=1024):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, errors='ignore') as f:
                self.yaml = yaml.safe_load(f)
        self.nh = nh
        self.nl = nl
        if nh and nh != self.yaml['nh']:
            LOGGER.info(f"Overriding model.yaml nh={self.yaml['nh']} with nh={nh}")
            self.yaml['nh'] = nh
        if nl and nl != self.yaml['nl']:
            LOGGER.info(f"Overriding model.yaml nl={self.yaml['nl']} with nl={nl}")
            self.yaml['nl'] = nl
        if pl and pl != self.yaml['pl']:
            LOGGER.info(f"Overriding model.yaml pl={self.yaml['pl']} with pl={pl}")
            self.yaml['pl'] = pl
        if dl and dl != self.yaml['dl']:
            LOGGER.info(f"Overriding model.yaml dl={self.yaml['dl']} with nl={dl}")
            self.yaml['dl'] = dl
        self.model, self.save = parse_model(deepcopy(self.yaml))

        initialize_weights(self)

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi in m.m:
            b = mi.bias.view(m.ch, -1)
            pass

    def forward(self, x, visualize=False):
        return self._forward_once(x, visualize)

    def _forward_once(self, x, visualize=False):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[0][m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

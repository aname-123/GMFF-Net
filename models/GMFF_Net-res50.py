import sys
import torch
import torch.nn as nn
import math

from copy import deepcopy
from utils.general import LOGGER
from utils.torch_utils import initialize_weights
from utils.plots import feature_visualization

from models.backbone import *
from models.common import Concat, C3, Conv, Add, C2f, C2f_CBAM, CBAM, EMA, C2f_EMA, C2f_EMABo

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH


def parse_model(model_dict):     # model_dict
    nh, nl, pl, dl, backbone = model_dict['nh'], model_dict['nl'], model_dict['pl'], model_dict['dl'], model_dict['backbone']
    backbone = eval(backbone)()   # model backbone
    t = str(backbone)[0:6].replace('__main__', '')  # module type
    np = sum([x.numel() for x in backbone.parameters()])  # number params
    backbone.i, backbone.f, backbone.type, backbone.np = 0, -1, t, np

    layers, save = [backbone], []
    for i, (f, n, m, args) in enumerate(model_dict['Feature pyramid'], start=1):
        m = eval(m) if isinstance(m, str) else m   # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a   # eval strings
            except NameError:
                pass
        # 修改：if m in [Conv, C3]:
        if m in [Conv, C3, C2f, C2f_CBAM, C2f_EMA]:
            if m in [C3, C2f, C2f_CBAM]:
                args.insert(2, n)   # number of repeats
                n = 1
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__', '')    # module type
        np = sum([x.numel() for x in m_.parameters()])   # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)

    return nn.Sequential(*layers), sorted(save)


class Head(nn.Module):
    def __init__(self, ch, nh=1, nl=2, pl=7, dl=1024):
        super(Head, self).__init__()
        self.nh = nh   # number of channels
        self.nl = nl   # pitch yaw
        self.m = nn.Conv2d(ch, self.nh * self.nl, 1)  # output conv
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
    # 修改：def __init__(self, cfg='resnet50-FP.yaml', nh=1, nl=2, pl=4, dl=512):
    def __init__(self, cfg='resnet50-GMSFF-EMABo_v3.yaml', nh=1, nl=2, pl=7, dl=1024):
        super().__init__()
        if isinstance(cfg, dict):  # 检查变量 cfg 是否是一个字典类型
            self.yaml = cfg   # model dict
        else:
            import yaml
            self.yaml_file = Path(cfg).name  # 获取cfg路径的文件名部分
            with open(cfg, errors='ignore') as f:
                self.yaml = yaml.safe_load(f)
        self.nh = nh
        self.nl = nl
        if nh and nh != self.yaml['nh']:
            LOGGER.info(f"Overriding model.yaml nh={self.yaml['nh']} with nh={nh}")
            self.yaml['nh'] = nh  # override yaml value
        if nl and nl != self.yaml['nl']:
            LOGGER.info(f"Overriding model.yaml nl={self.yaml['nl']} with nl={nl}")
            self.yaml['nl'] = nl  # override yaml value
        if pl and pl != self.yaml['pl']:
            LOGGER.info(f"Overriding model.yaml pl={self.yaml['pl']} with pl={pl}")
            self.yaml['pl'] = pl  # override yaml value
        if dl and dl != self.yaml['dl']:
            LOGGER.info(f"Overriding model.yaml dl={self.yaml['dl']} with nl={dl}")
            self.yaml['dl'] = dl  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml))

        # Init weights, biases
        initialize_weights(self)

    def _initialize_biases(self, cf=None):   # initialize biases into Head()
        m = self.model[-1]
        for mi in m.m:
            b = mi.bias.view(m.ch, -1)
            pass

    def forward(self, x, visualize=False):
        return self._forward_once(x, visualize)

    def _forward_once(self, x, visualize=False):
        y = []   # output
        for m in self.model:
            if m.f != -1:    # if not from previous layer
                x = y[0][m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]   # from earlier layers。x从骨干网络不同的y[0][f]中得到输入
            x = m(x)   # run
            y.append(x if m.i in self.save else None)   # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x


if __name__ == "__main__":
    image = torch.randn([1, 3, 448, 448])
    model = XModel("./yaml/resnet50-GMSFF-EMABo_v3.yaml")
    pred = model(image)
    print("finish!")

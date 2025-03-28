from models.common import ResNet, BasicBlock, Bottleneck


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])



__all__ = [
    'resnet18', 'resnet50', 'resnet101'
]


if __name__ == "__main__":
    resnet50 = eval("resnet50")()
    resnet18 = eval("resnet18")()
    resnet101 = eval("resnet101")()

    print("finish!")

from crowdcount.utils.registry import Registry
from . import CSRNet, ResNet, MCNN, VGG

MODELS = Registry()


@MODELS.register("CSRNet")
def build_csrnet(cfg):
    return CSRNet(cfg)


@MODELS.register("res50")
@MODELS.register("res101")
def build_resnet(cfg):
    return ResNet(cfg)


def build_model(cfg):
    assert cfg.MODEL.NAME in MODELS, "{} not in model zoo".format(cfg.MODEL.NAME)
    return MODELS[cfg.MODEL.NAME](cfg)

from config.config import cfg
from models.autodecoer import AutoEncoder
from models.swin_transformer import CnnSwinTransformer
from models.DAUE import DeepAutoEncoder

def creat_model(model_name, **kwargs):

    if model_name == 'cnn_vit':
        net = AutoEncoder(P=cfg.TRAINING.P, L=cfg.TRAINING.L, size=cfg.TRAINING.COL,
                          patch=cfg.TRAINING.PATCH, dim=cfg.TRAINING.DIM).to(cfg.SYSTEM.DEVICE)
    elif model_name == 'swin_transformer':
        net = CnnSwinTransformer(P=cfg.TRAINING.P, L=cfg.TRAINING.L, size=cfg.TRAINING.COL,
                                patch=cfg.TRAINING.PATCH, dim=cfg.TRAINING.DIM, col=cfg.TRAINING.COL, checkpoint=True).to(cfg.SYSTEM.DEVICE)
    elif model_name == 'DAUE':
        net = DeepAutoEncoder(num_bands=cfg.TRAINING.L, end_members=cfg.TRAINING.P).to(cfg.SYSTEM.DEVICE)

    return net
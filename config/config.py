import os
import time

import yaml
import torch
# from yacs.config import C
from yacs.config import CfgNode as CN

_C = CN()

# 基本超参
_C.SYSTEM = CN()
_C.SYSTEM.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_C.SYSTEM.TIME = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())

# 训练过程超参
_C.TRAINING = CN()
# _C.TRAINING.MODEL = 'preresnet'
# apex samson urban jasper
# synthetic_legendre synthetic_RationalGF synthetic_sphericGF
# synthetic_maternGF synthetic_maternGFSNR20
_C.TRAINING.DATASET = 'jasper_dataset.mat'
# cnn_vit, swin_transformer, DAUE
_C.TRAINING.MODEL_NAME = 'cnn_vit'

_C.TRAINING.P = 5
_C.TRAINING.L = 228
_C.TRAINING.COL = 128
_C.TRAINING.LR = 1e-3
_C.TRAINING.COSLR_RATE = 0.01
_C.TRAINING.EPOCH = 1000
_C.TRAINING.PATCH = 4
_C.TRAINING.DIM = 30
_C.TRAINING.BETA = 5e3
_C.TRAINING.GAMMA = 5e-2
_C.TRAINING.WEIGHT_DECAY_PARAM = 0.0
_C.TRAINING.INIT = True
_C.TRAINING.BATCH = 20
# # autodecoder
# _C.TRAINING.ORDER_ABD = (0, 1, 2)
# _C.TRAINING.ORDER_ENDMEM = (0, 1, 2)

# #cnn_vit
# _C.TRAINING.ORDER_ABD = (2, 0, 1)
# _C.TRAINING.ORDER_ENDMEM = (2, 0, 1)
_C.TRAINING.ORDER_ABD = (0, 1, 2, 3, 4)
_C.TRAINING.ORDER_ENDMEM = (0, 1, 2, 3, 4)

# swintrans
# _C.TRAINING.ORDER_ABD = (2, 0, 1)
# _C.TRAINING.ORDER_ENDMEM = (2, 0, 1)
# _C.TRAINING.ORDER_ABD = (1, 0, 2)
# _C.TRAINING.ORDER_ENDMEM = (1, 0, 2)
# # swintrans init
# _C.TRAINING.ORDER_ABD = (0, 1, 2)
# _C.TRAINING.ORDER_ENDMEM = (0, 1, 2)

# PATH
_C.PATH = CN()
_C.PATH.DATASET = './data/datasets/jasper_dataset.mat'
_C.PATH.SAVE_DIR = './runs/logs/'

cfg = _C

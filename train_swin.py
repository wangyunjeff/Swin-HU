import os
import pickle
import time
import math

import scipy.io as sio
import torch
import torch.nn as nn
from torchsummary import summary
import random
import numpy as np
import yaml

from config.config import cfg
from models.autodecoer import AutoEncoder
from models.cnn_vit import CnnViT
from models.swin_transformer import CnnSwinTransformer
from models.creat_model import creat_model
from data.data_build import TrainData
from utils import utils
from engine import trainer, tester
from utils import plots
from utils.utils import vca


cfg_path = "./config/cfg_synthetic_maternGFSNR40_swin_transformer_init.yaml"
# cfg_path = r"H:\Code_F\HSU-Using-Transformer\runs\ok\20220928-16-16-00_swin_transformer_samson\cfg_samson_swin_transformer_init_best.yaml"
torch.cuda.empty_cache()
seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
cfg.merge_from_file(cfg_path)
exp_dir = os.path.join(cfg.PATH.SAVE_DIR, cfg.SYSTEM.TIME+f'_{cfg.TRAINING.MODEL_NAME}'+f'_{cfg.TRAINING.DATASET}/')
os.makedirs(exp_dir, exist_ok=True)
print(cfg)

with open(os.path.join(exp_dir, os.path.basename(cfg_path)), "w") as f:
    yaml.dump(yaml.load(cfg.dump(), Loader=yaml.FullLoader), f)


dataset = TrainData(cfg.PATH.DATASET)
gen = torch.utils.data.DataLoader(dataset=dataset,
                                  batch_size=cfg.TRAINING.COL**2,
                                  shuffle=False)
init_weight = dataset.get("init_weight").unsqueeze(2).unsqueeze(3).float()
net = creat_model(cfg.TRAINING.MODEL_NAME)
# summary(net, input_size=(285,110,110), batch_size=1)
# net = CnnViT(P=cfg.TRAINING.P, L=cfg.TRAINING.L, size=cfg.TRAINING.COL,
#                   patch=cfg.TRAINING.PATCH, dim=cfg.TRAINING.DIM, col=cfg.TRAINING.COL).to(cfg.SYSTEM.DEVICE)

# net = AutoEncoder(P=cfg.TRAINING.P, L=cfg.TRAINING.L, size=cfg.TRAINING.COL,
#                   patch=cfg.TRAINING.PATCH, dim=cfg.TRAINING.DIM).to(cfg.SYSTEM.DEVICE)
# net = CnnSwinTransformer(P=cfg.TRAINING.P, L=cfg.TRAINING.L, size=cfg.TRAINING.COL,
#                         patch=cfg.TRAINING.PATCH, dim=cfg.TRAINING.DIM, col=cfg.TRAINING.COL, checkpoint=True).to(cfg.SYSTEM.DEVICE)
net.apply(net.weights_init)
model_dict = net.state_dict()
if cfg.TRAINING.INIT:
    model_dict['decoder.0.weight'] = init_weight

# model_dict['decoder.0.weight'] = torch.from_numpy(sio.loadmat(r'H:\Code_F\HSU-Using-Transformer\data\datasets\samson_dataset.mat')['M']).unsqueeze(2).unsqueeze(3).float()

# for name, para in net.named_parameters():
#     if name == 'decoder.0.weight':
#         # b=para
#         para.requires_grad_(False)

# data_path2 = R'H:\Code_F\HSU-Using-Transformer\data\datasets\samson_dataset.mat'
# data2 = sio.loadmat(data_path2)
# him2 = data2['Y']
# init = torch.from_numpy(vca(him2, 3)[0]).unsqueeze(2).unsqueeze(3).float()
# model_dict['decoder.0.weight'] = init

net.load_state_dict(model_dict)

loss_func = nn.MSELoss(reduction='mean')
loss_func2 = utils.SAD(cfg.TRAINING.L)
loss_function = {'loss_mse': loss_func,
                 'loss_sad': loss_func2}
optimizer = torch.optim.Adam(net.parameters(),
                             lr=cfg.TRAINING.LR,
                             weight_decay=cfg.TRAINING.WEIGHT_DECAY_PARAM,
                             )
# optimizer = torch.optim.RMSprop(net.parameters(), lr=cfg.TRAINING.LR, weight_decay=0.0)
if cfg.TRAINING.INIT:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,181,182,183,190,192,194,196], gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 60,70,75, 90,100,120, 130,140], gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  milestones=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90,
    #                                                              100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150,
    #                                                              155, 160, 165, 170, 175, 180, 181, 182, 183, 190, 192,
    #                                                              194, 196], gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  milestones=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90,
    #                                                              100, 105, 110, 115], gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.845)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.90)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 120, 130, 140], gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.0001, 0.001, step_size_up=50, step_size_down=50,
    #                                              mode='triangular2', gamma=0, scale_fn=None, scale_mode='cycle',
    #                                              cycle_momentum=False, base_momentum=0.8, max_momentum=0.9,
    #                                              last_epoch=-1)
    # lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAINING.EPOCH)) / 2) * (
    #         1 - cfg.TRAINING.COSLR_RATE) + cfg.TRAINING.COSLR_RATE
    # # lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAINING.EPOCH)) / 2) * (1 - 0.5) + 0.5
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    flag = 50
else:
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAINING.EPOCH)) / 2) * (
                1 - cfg.TRAINING.COSLR_RATE) + cfg.TRAINING.COSLR_RATE
    # lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAINING.EPOCH)) / 2) * (1 - 0.5) + 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    flag = 20
# lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAINING.EPOCH)) / 2) * (
#                 1 - cfg.TRAINING.COSLR_RATE) + cfg.TRAINING.COSLR_RATE
# # lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAINING.EPOCH)) / 2) * (1 - 0.5) + 0.5
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

for epoch in range(cfg.TRAINING.EPOCH):
    trainer.train_loop(net, gen, gen, optimizer, loss_function, epoch, flag)
    # tester.test_new(net, dataset, exp_dir, epoch)
    scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    if epoch % flag == 0:
        tester.test_new(net, dataset, exp_dir, epoch)

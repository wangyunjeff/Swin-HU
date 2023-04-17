import time
import os

import torch
import torch.nn as nn
from torch import autograd
import scipy.io as sio
from tqdm import tqdm
from config.config import cfg
from utils import utils, plots
from data.data_build import TrainData


apply_clamp_inst1 = utils.NonZeroClipper()

def train_loop(net, gen, gen_test, optimizer, loss_function, epoch_i, flag=10):
    time_start = time.time()
    net.train()
    epo_vs_los = []

    for i, (x, _) in enumerate(gen):
        '''wy 
        loss以及BETA,GAMMA的作用
        BETA和GAMMA分别对丰度和端元进行约束、制衡，有几种可以调整的方案：
        1、BETA*loss_re 越接近 GAMME*loss_sad(训练结束时)，则效果越好；
        2、只对loss_re进行反向传播，则丰度图的结果会越好；因为loss_re制约了丰度，loss_sad制约了端元；
        3、只对loss_sad进行反向传播。
        '''
        optimizer.zero_grad()
        # with autograd.detect_anomaly():
        x = x.transpose(1, 0).view(1, -1, cfg.TRAINING.COL, cfg.TRAINING.COL)
        abu_est, re_result = net(x)

        if cfg.TRAINING.INIT:
                # 原方案
                loss_re = cfg.TRAINING.BETA * loss_function['loss_mse'](re_result, x)
                loss_sad = loss_function['loss_sad'](re_result.view(1, cfg.TRAINING.L, -1).transpose(1, 2),
                                      x.view(1, cfg.TRAINING.L, -1).transpose(1, 2))
                loss_sad = cfg.TRAINING.GAMMA * torch.sum(loss_sad).float()
                total_loss = loss_re + loss_sad
        else:
            # 方案3
            loss_re = loss_function['loss_mse'](re_result, x)
            loss_sad = loss_function['loss_sad'](re_result.view(1, cfg.TRAINING.L, -1).transpose(1, 2),
                                                 x.view(1, cfg.TRAINING.L, -1).transpose(1, 2))
            loss_sad = torch.sum(loss_sad).float()
            total_loss = loss_sad

        # loss_re = loss_function['loss_mse'](re_result, x)
        # loss_sad = loss_function['loss_sad'](re_result.view(1, cfg.TRAINING.L, -1).transpose(1, 2),
        #                                      x.view(1, cfg.TRAINING.L, -1).transpose(1, 2))
        # loss_sad = torch.sum(loss_sad).float()
        # total_loss = loss_sad
        # print('loss_c:', loss_c)


        # # 方案1 BETA=1 GAMMA=1
        # loss_re = 5000 * loss_function['loss_mse'](re_result, x)
        # loss_sad = loss_function['loss_sad'](re_result.view(1, cfg.TRAINING.L, -1).transpose(1, 2),
        #                                      x.view(1, cfg.TRAINING.L, -1).transpose(1, 2))
        # loss_sad = 1 * torch.sum(loss_sad).float()
        # total_loss = loss_re + loss_sad

        # # 方案2
        # loss_re = loss_function['loss_mse'](re_result, x)
        # loss_sad = loss_function['loss_sad'](re_result.view(1, cfg.TRAINING.L, -1).transpose(1, 2),
        #                                      x.view(1, cfg.TRAINING.L, -1).transpose(1, 2))
        # loss_sad = torch.sum(loss_sad).float()
        # total_loss = loss_re


        total_loss.backward()
        # scaler.scale(total_loss).backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=1, norm_type=1)

        optimizer.step()
        # scaler.step(optimizer)
        net.decoder.apply(apply_clamp_inst1)

        if epoch_i % flag == 0:
            print('Epoch:', epoch_i, '| train loss: %.4f' % total_loss.data,
                  '| re loss: %.4f' % loss_re.data,
                  '| sad loss: %.4f' % loss_sad.data)
        epo_vs_los.append(float(total_loss.data))

    exp_dir = os.path.join(cfg.PATH.SAVE_DIR, cfg.SYSTEM.TIME+f'_{cfg.TRAINING.MODEL_NAME}'+f'_{cfg.TRAINING.DATASET}/')
    if epoch_i % flag == 0:
        with open(exp_dir + "log1.csv", 'a') as file:
            file.write(f"Epoch: {epoch_i}, ")
            file.write(f"LR: {optimizer.state_dict()['param_groups'][0]['lr']}, ")
            file.write('| train loss: %.4f' % total_loss.data)
            file.write('| re loss: %.4f' % loss_re.data)
            file.write('| sad loss: %.4f\n' % loss_sad.data)

def train_loop_DAUE(net, gen, gen_test, optimizer, loss_function, epoch_i, flag=10):
    exp_dir = os.path.join(cfg.PATH.SAVE_DIR,
                           cfg.SYSTEM.TIME + f'_{cfg.TRAINING.MODEL_NAME}' + f'_{cfg.TRAINING.DATASET}/')

    for i, (x, _) in enumerate(gen):
        enc_out, dec_out = net(x.float())
        loss = loss_function(dec_out, x.float())
        loss = torch.sum(loss).float()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch_i+1)%flag==0:
        print(f'Epoch {epoch_i + 1:04d} / {cfg.TRAINING.EPOCH:04d}', end='\n=================\n')
        print("Loss: %.4f" %(loss.item()))
    if epoch_i % flag == 0:
        with open(exp_dir + "log1.csv", 'a') as file:
            file.write(f"Epoch: {epoch_i}, ")
            file.write(f"LR: {optimizer.state_dict()['param_groups'][0]['lr']}, ")
            file.write('| train loss: %.4f\n' % loss.data)
    pass


class SAD(nn.Module):
    def __init__(self, num_bands: int = 156):
        super(SAD, self).__init__()
        self.num_bands = num_bands

    def forward(self, input, target):
        """Spectral Angle Distance Objective
        Implementation based on the mathematical formulation presented in 'https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7061924'

        Params:
            input -> Output of the autoencoder corresponding to subsampled input
                    tensor shape: (batch_size, num_bands)
            target -> Subsampled input Hyperspectral image (batch_size, num_bands)

        Returns:
            angle: SAD between input and target
        """
        try:
            input_norm = torch.sqrt(torch.bmm(input.view(-1, 1, self.num_bands), input.view(-1, self.num_bands, 1)))
            target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1)))

            summation = torch.bmm(input.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
            angle = torch.acos(summation / (input_norm * target_norm))


        except ValueError:
            return 0.0

        return angle
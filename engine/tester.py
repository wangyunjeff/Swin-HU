import time
import os

import torch
import torch.nn as nn
import scipy.io as sio

from config.config import cfg
from utils import utils, plots
from data.data_build import TrainData





def test_or(net, dataset, exp_dir):
    result_dir = os.path.join(exp_dir, 'result/')
    os.makedirs(result_dir, exist_ok=True)
    net.eval()
    x = dataset.get("hs_img").transpose(1, 0).view(1, -1, cfg.TRAINING.COL, cfg.TRAINING.COL)
    abu_est, re_result = net(x)
    abu_est = abu_est / (torch.sum(abu_est, dim=1))
    abu_est = abu_est.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    target = torch.reshape(dataset.get("abd_map"), (cfg.TRAINING.COL, cfg.TRAINING.COL, cfg.TRAINING.P)).cpu().numpy()
    true_endmem = dataset.get("end_mem").numpy()
    est_endmem = net.state_dict()["decoder.0.weight"].cpu().numpy()
    est_endmem = est_endmem.reshape((cfg.TRAINING.L, cfg.TRAINING.P))

    abu_est = abu_est[:, :, cfg.TRAINING.ORDER_ABD]
    est_endmem = est_endmem[:, cfg.TRAINING.ORDER_ENDMEM]

    sio.savemat(result_dir + f"{cfg.TRAINING.DATASET}_abd_map.mat", {"A_est": abu_est})
    sio.savemat(result_dir + f"{cfg.TRAINING.DATASET}_endmem.mat", {"E_est": est_endmem})

    x = x.view(-1, cfg.TRAINING.COL, cfg.TRAINING.COL).permute(1, 2, 0).detach().cpu().numpy()
    re_result = re_result.view(-1, cfg.TRAINING.COL, cfg.TRAINING.COL).permute(1, 2, 0).detach().cpu().numpy()
    re = utils.compute_re(x, re_result)
    print("RE:", re)

    rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
    print("Class-wise RMSE value:")
    for i in range(cfg.TRAINING.P):
        print("Class", i + 1, ":", rmse_cls[i])
    print("Mean RMSE:", mean_rmse)

    sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)
    print("Class-wise SAD value:")
    for i in range(cfg.TRAINING.P):
        print("Class", i + 1, ":", sad_cls[i])
    print("Mean SAD:", mean_sad)

    with open(exp_dir + "log1.csv", 'a') as file:
        file.write(f"LR: {cfg.TRAINING.LR}, ")
        file.write(f"WD: {cfg.TRAINING.WEIGHT_DECAY_PARAM}, ")
        file.write(f"RE: {re:.4f}, ")
        file.write(f"SAD: {mean_sad:.4f}, ")
        file.write(f"RMSE: {mean_rmse:.4f}\n")

    plots.plot_abundance(target, abu_est, cfg.TRAINING.P, exp_dir)
    plots.plot_endmembers(true_endmem, est_endmem, cfg.TRAINING.P, exp_dir)
    time_end = time.time()


def test_new(net, dataset, exp_dir, epoch):
    result_dir = os.path.join(exp_dir, 'result/')
    os.makedirs(result_dir, exist_ok=True)
    net.eval()

    x = dataset.get("hs_img").transpose(1, 0).view(1, -1, cfg.TRAINING.COL, cfg.TRAINING.COL)
    # abu_est
    # net-> [1, P, COL, COL]
    # squeeze(0).permute(1, 2, 0)-> [P, COL, COL]
    target = torch.reshape(dataset.get("abd_map"), (cfg.TRAINING.COL, cfg.TRAINING.COL, cfg.TRAINING.P)).cpu().numpy()
    abu_est, re_result = net(x)
    abu_est = abu_est / (torch.sum(abu_est, dim=1))
    abu_est = abu_est.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


    true_endmem = dataset.get("end_mem").numpy()
    est_endmem = net.state_dict()["decoder.0.weight"].cpu().numpy()
    est_endmem = est_endmem.reshape((cfg.TRAINING.L, cfg.TRAINING.P ))

    abu_est = abu_est[:, :, cfg.TRAINING.ORDER_ABD]
    est_endmem = est_endmem[:, cfg.TRAINING.ORDER_ENDMEM]

    # sio.savemat(result_dir + f"{cfg.TRAINING.DATASET}_abd_map.mat", {"A_est": abu_est})
    # sio.savemat(result_dir + f"{cfg.TRAINING.DATASET}_endmem.mat", {"E_est": est_endmem})
    sio.savemat(result_dir + f"{cfg.TRAINING.DATASET}_abd_map{epoch}.mat", {"A_est": abu_est})
    sio.savemat(result_dir + f"{cfg.TRAINING.DATASET}_endmem{epoch}.mat", {"E_est": est_endmem})

    x = x.view(-1, cfg.TRAINING.COL, cfg.TRAINING.COL).permute(1, 2, 0).detach().cpu().numpy()
    re_result = re_result.view(-1, cfg.TRAINING.COL, cfg.TRAINING.COL).permute(1, 2, 0).detach().cpu().numpy()
    re = utils.compute_re(x, re_result)
    # target:[COL, COL, P]
    # abu_est:[]
    rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
    print("Class-wise RMSE value:")
    for i in range(cfg.TRAINING.P):
        print("Class", i + 1, ":", rmse_cls[i])

    sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)
    print("Class-wise SAD value:")
    for i in range(cfg.TRAINING.P):
        print("Class", i + 1, ":", sad_cls[i])

    print(f'Epoch:{epoch}', f'| RE:{re}', f'| Mean RMSE:{mean_rmse}',f'| Mean SAD:{mean_sad}')
    with open(exp_dir + "log1.csv", 'a') as file:
        file.write(f"Epoch: {epoch}, ")
        file.write(f"LR: {cfg.TRAINING.LR}, ")
        file.write(f"WD: {cfg.TRAINING.WEIGHT_DECAY_PARAM}, ")
        file.write(f"RE: {re:.4f}, ")
        file.write(f"SAD: {mean_sad:.4f}, ")
        file.write(f"RMSE: {mean_rmse:.4f}\n")

    # plots.plot_abundance(target, abu_est, cfg.TRAINING.P, exp_dir, epoch)
    # plots.plot_endmembers(true_endmem, est_endmem, cfg.TRAINING.P, exp_dir,epoch)
    plots.plot_abundance(target, abu_est, cfg.TRAINING.P, exp_dir)
    plots.plot_endmembers(true_endmem, est_endmem, cfg.TRAINING.P, exp_dir)

    time_end = time.time()


def test_noinit(root, dataset, trans):

    path1 = os.path.join(root, cfg.TRAINING.DATASET+'_abd_map.mat')
    path2 = os.path.join(root, cfg.TRAINING.DATASET+'_endmem.mat')

    target = torch.reshape(dataset.get("abd_map"), (cfg.TRAINING.COL, cfg.TRAINING.COL, cfg.TRAINING.P)).cpu().numpy()
    true_endmem = dataset.get("end_mem").numpy()

    abu_est = sio.loadmat(path1)['A_est']
    est_endmem = sio.loadmat(path2)['E_est']

    abu_est = abu_est[:, :, trans]
    est_endmem = est_endmem[:, trans]

    rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
    # print("Class-wise RMSE value:")
    # for i in range(cfg.TRAINING.P):
    #     print("Class", i + 1, ":", rmse_cls[i])

    sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)
    # print("Class-wise SAD value:")
    # for i in range(cfg.TRAINING.P):
    #     print("Class", i + 1, ":", sad_cls[i])

    print(f'| Mean RMSE:{mean_rmse}',f'| Mean SAD:{mean_sad}')

    plots.plot_abundance(target, abu_est, cfg.TRAINING.P, './abundance.png')
    plots.plot_endmembers(true_endmem, est_endmem, cfg.TRAINING.P, 'end_members.png')

def test_DAUE(net, dataset, exp_dir, epoch):
    result_dir = os.path.join(exp_dir, 'result/')
    os.makedirs(result_dir, exist_ok=True)
    net.eval()


    x = dataset.get("hs_img").transpose(1, 0).view(1, -1, cfg.TRAINING.COL, cfg.TRAINING.COL)
    # 丰度和端元的提取与转换
    target = torch.reshape(dataset.get("abd_map"), (cfg.TRAINING.COL, cfg.TRAINING.COL, cfg.TRAINING.P)).cpu().numpy()
    abu_est, re_result = net(dataset.get("hs_img"))    # [1, P, COL, COL]
    # abu_est = abu_est / (torch.sum(abu_est, dim=1))
    abu_est = torch.softmax(abu_est,1)
    abu_est = torch.reshape(abu_est.squeeze(0), (cfg.TRAINING.COL, cfg.TRAINING.COL, cfg.TRAINING.P)).detach().cpu().numpy()    # [P, COL, COL]

    true_endmem = dataset.get("end_mem").numpy()
    est_endmem = net.state_dict()["decoder.weight"].cpu().numpy()
    est_endmem = est_endmem.reshape((cfg.TRAINING.L, cfg.TRAINING.P ))

    abu_est = abu_est[:, :, cfg.TRAINING.ORDER_ABD]
    est_endmem = est_endmem[:, cfg.TRAINING.ORDER_ENDMEM]

    # 保存结果
    sio.savemat(result_dir + f"{cfg.TRAINING.DATASET}_abd_map.mat", {"A_est": abu_est})
    sio.savemat(result_dir + f"{cfg.TRAINING.DATASET}_endmem.mat", {"E_est": est_endmem})

    # 根据评价指标进行计算
    x = x.view(-1, cfg.TRAINING.COL, cfg.TRAINING.COL).permute(1, 2, 0).detach().cpu().numpy()
    re_result = re_result.view(-1, cfg.TRAINING.COL, cfg.TRAINING.COL).permute(1, 2, 0).detach().cpu().numpy()
    re = utils.compute_re(x, re_result)
    # target:[COL, COL, P]
    # abu_est:[]
    rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
    print("Class-wise RMSE value:")
    for i in range(cfg.TRAINING.P):
        print("Class", i + 1, ":", rmse_cls[i])

    sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)
    print("Class-wise SAD value:")
    for i in range(cfg.TRAINING.P):
        print("Class", i + 1, ":", sad_cls[i])

    print(f'Epoch:{epoch}', f'| RE:{re}', f'| Mean RMSE:{mean_rmse}',f'| Mean SAD:{mean_sad}')
    with open(exp_dir + "log1.csv", 'a') as file:
        file.write(f"Epoch: {epoch}, ")
        file.write(f"LR: {cfg.TRAINING.LR}, ")
        file.write(f"WD: {cfg.TRAINING.WEIGHT_DECAY_PARAM}, ")
        file.write(f"RE: {re:.4f}, ")
        file.write(f"SAD: {mean_sad:.4f}, ")
        file.write(f"RMSE: {mean_rmse:.4f}\n")

    # 画图
    plots.plot_abundance(target, abu_est, cfg.TRAINING.P, exp_dir)
    plots.plot_endmembers(true_endmem, est_endmem, cfg.TRAINING.P, exp_dir)
    time_end = time.time()
if __name__ == '__main__':

    dataset = TrainData('H:\Code_F\HSU-Using-Transformer\data\datasets\samson_dataset.mat')
    test_noinit(r'H:\Code_F\HSU-Using-Transformer\runs\logs\20220927-18-00-52_samson\result',
                dataset,
                (1, 2, 0))
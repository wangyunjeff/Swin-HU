from matplotlib import pyplot as plt
from utils import utils

def plot_abundance(ground_truth, estimated, em, save_dir, epoch=None):
    rmse_cls, mean_rmse = utils.compute_rmse(ground_truth, estimated)
    plt.figure(figsize=(12, 6), dpi=150)
    for i in range(em):
        plt.subplot(2, em, i + 1)
        plt.imshow(ground_truth[:, :, i], cmap='jet')

    for i in range(em):
        ax = plt.subplot(2, em, em + i + 1)
        ax.set_title(rmse_cls[i])
        plt.imshow(estimated[:, :, i], cmap='jet')
    plt.suptitle(f'mRMSE:{mean_rmse}')
    plt.tight_layout()

    plt.savefig(save_dir + f"abundance_{epoch}.png")
    plt.close()

def plot_endmembers(target, pred, em, save_dir,epoch=None):
    num_endmembers = pred.shape[1]
    sad_cls, mean_sad = utils.compute_sad(target, pred)
    for i in range(num_endmembers):
        pred[:, i] = pred[:, i] / pred[:, i].max()
        target[:, i] = target[:, i] / target[:, i].max()
    plt.figure(figsize=(12, 6), dpi=150)
    for i in range(em):
        ax = plt.subplot(2, em // 2 if em % 2 == 0 else em, i + 1)
        ax.set_title(sad_cls[i])
        plt.plot(pred[:, i], label="Extracted")
        plt.plot(target[:, i], label="GT")
        plt.legend(loc="upper left")

    plt.suptitle(f'mSAD:{mean_sad}')
    plt.tight_layout()

    plt.savefig(save_dir + f"end_members_{epoch}.png")
    plt.close()
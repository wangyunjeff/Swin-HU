from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from collections import OrderedDict
from torch.autograd import Variable, Function


class DeepAutoEncoder(nn.Module):
    def __init__(self, num_bands: int = 156, end_members: int = 3, dropout: float = 1.0,
                 activation: str = 'ReLU', threshold: int = 5.0, ae_type: str = 'deep'):
        # Constructor
        super(DeepAutoEncoder, self).__init__()

        if activation == 'ReLU':
            self.act = nn.ReLU()
        elif activation == 'LReLU':
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.Sigmoid()

        self.gauss = GaussianDropout(dropout)
        self.asc = ASC()
        if ae_type == 'deep':
            self.encoder = nn.Sequential(OrderedDict([
                ('hidden_1', nn.Linear(num_bands, 9 * end_members)),
                ('activation_1', self.act),
                ('hidden_2', nn.Linear(9 * end_members, 6 * end_members)),
                ('activation_2', self.act),
                ('hidden_3', nn.Linear(6 * end_members, 3 * end_members)),
                ('activation_3', self.act),
                ('hidden_4', nn.Linear(3 * end_members, end_members)),
                ('activation_4', self.act),
                ('batch_norm', nn.BatchNorm1d(end_members)),
                ('soft_thresholding', nn.Softplus(threshold=threshold)),
                # ('ASC', self.asc),
                ('Gaussian_Dropout', self.gauss)

            ]))
        elif ae_type == 'shallow':
            self.encoder = nn.Sequential(OrderedDict([
                ('hidden_1', nn.Linear(num_bands, end_members)),
                ('batch_norm', nn.BatchNorm1d(end_members)),
                ('soft_thresholding', nn.Softplus(threshold=threshold)),
                ('ASC', self.asc),
                ('Gaussian_Dropout', self.gauss)
            ]))

        self.decoder = nn.Linear(end_members, num_bands, bias=False)

    def forward(self, img):
        # img = img.view(9025,156)
        encoded = self.encoder(img)
        decoded = self.decoder(encoded)
        return encoded, decoded


class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        # Constructor
        super(GaussianDropout, self).__init__()

        self.alpha = torch.Tensor([alpha])

    def forward(self, x):
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x


class ASC(nn.Module):
    def __init__(self):
        super(ASC, self).__init__()

    def forward(self, input):
        """Abundances Sum-to-One Constraint"""
        constrained = input / torch.sum(input, dim=0)
        return constrained


if __name__ == '__main__':
    from torchsummary import summary
    from torchstat import stat
    from thop import clever_format, profile

    input_shape = [285, 110, 110]
    net = DeepAutoEncoder(num_bands=input_shape[0],).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(110*110, input_shape[0], ).to(device)
    # dummy_input = torch.randn(1, input_shape[0], ).to(device)
    import time
    s_time = time.clock()
    flops, params = profile(net.to(device), (dummy_input,), verbose=False)
    e_time = time.clock()
    print(f'running time:{e_time-s_time}')
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))

    # stat(net, input_size=(9025, 156))
    # stat(net, input_size=(156, 95, 95))
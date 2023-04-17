# class Encoder(tf.keras.Model):
#     def __init__(self, params):
#         super(Encoder, self).__init__()
#         self.params = params
#         self.hidden_layer_one = tf.keras.layers.Conv2D(filters=self.params['e_filters'],
#                                                        kernel_size=self.params['e_size'],
#                                                        activation=self.params['activation'], strides=1, padding='same',
#                                                        kernel_initializer=params['initializer'], use_bias=False)
#         self.hidden_layer_two = tf.keras.layers.Conv2D(filters=self.params['num_endmembers'], kernel_size=1,
#                                                        activation=self.params['activation'], strides=1, padding='same',
#                                                        kernel_initializer=self.params['initializer'], use_bias=False)
#         self.asc_layer = SumToOne(params=self.params, name='abundances')
#
#     def call(self, input_patch):
#         code = self.hidden_layer_one(input_patch)
#         code = tf.keras.layers.BatchNormalization()(code)
#         code = tf.keras.layers.SpatialDropout2D(0.2)(code)
#         code = self.hidden_layer_two(code)
#         code = tf.keras.layers.BatchNormalization()(code)
#         code = tf.keras.layers.SpatialDropout2D(0.2)(code)
#         code = self.asc_layer(code)
#         return code
#
# class Decoder(tf.keras.layers.Layer):
#     def __init__(self, params):
#         super(Decoder, self).__init__()
#         self.output_layer = tf.keras.layers.Conv2D(filters=params['d_filters'], kernel_size=params['d_size'],
#                                                    activation='linear',
#                                                    kernel_constraint=tf.keras.constraints.non_neg(),
#                                                    name='endmembers', strides=1, padding='same',
#                                                    kernel_regularizer=None,
#                                                    kernel_initializer=params['initializer'], use_bias=False)
#
#     def call(self, code):
#         recon = self.output_layer(code)
#         return recon
#
#     def getEndmembers(self):
#         return self.output_layer.get_weights()

import torch
import torch.nn as nn


class CNNAEU(nn.Module):
    def __init__(self, num_bands: int, end_members: int, ):
        super(CNNAEU, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_bands, 48, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            # nn.Conv2d(num_bands, 240, kernel_size=(4, 4), stride=(4, 4), padding=(2, 2)),
            nn.BatchNorm2d(48, momentum=0.99),
            nn.Dropout(0.2),
            nn.Conv2d(48, end_members, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(end_members, momentum=0.99),
            nn.Dropout(0.2),
        )
        # self.decoder = nn.Conv2d(end_members, num_bands, kernel_size=(13, 13), stride=(1, 1), padding=(0, 0),bias=False)

    def forward(self, x):
        abu_est = self.encoder(x)
        re_result = self.decoder(abu_est)
        return abu_est, re_result
if __name__ == '__main__':
    from torchsummary import summary
    from torchstat import stat
    from thop import clever_format, profile
    import matplotlib.pyplot as plt
    input_shape = [198, 100, 100]
    input_shape = [156, 95, 95]
    input_shape = [285, 110, 110]
    # input_shape = [300, 500, 100]
    patch_size = 40


    net = CNNAEU(num_bands=input_shape[0], end_members=4 ).cuda()
    # net = CNNAEU(num_bands=3, end_members=4 ).cuda()
    summary(net, (input_shape[0], input_shape[1], input_shape[2]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, input_shape[0], patch_size, patch_size ).to(device)
    # dummy_input = torch.randn(1, 3,256,256 ).to(device)
    size = dummy_input.shape
    import time

    s_time = time.clock()
    flops, params = profile(net.to(device), (dummy_input,), verbose=False)
    e_time = time.clock()
    print(f'running time:{e_time - s_time}')
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    # print(f'input_size:{size}')
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))

    # stat(net, input_size=(9025, 156))
    # stat(net, input_size=(156, 95, 95))

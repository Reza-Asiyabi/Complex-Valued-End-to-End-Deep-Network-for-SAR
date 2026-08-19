"""
Based on "R. M.Asiyabi, M. Datcu, A. Anghel, H. Nies, "Complex-Valued End-to-end Deep
 Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and
  Classification" IEEE Transactions on Geoscience and Remote Sensing (2023)"
"""

import torch
import torch.nn as nn
from Src.CV_Net_Modules import Complex_DoubleConv, Complex_Down_DoubleConv2d, Complex_Up, Complex_OutConv
from Src.CV_Net_Modules import Complex_Linear, Complex_Flatten
from Src.CV_Functions import ComplexReLU, ComplexConv2d, Complexavg_pool2d, ComplexConvTranspose2d


class CV_SAR_SuperRes_Net(nn.Module):
    def __init__(self, n_in_channels, n_out_channels):
        super(CV_SAR_SuperRes_Net, self).__init__()

        ############## Encoder
        self.en_conv1 = ComplexConv2d(in_channels=n_in_channels, out_channels=16, kernel_size=3, stride=(1, 2), padding=1,
                      dilation=1, groups=1, bias=True)
        self.en_conv2 = ComplexConv2d(in_channels=16, out_channels=32, kernel_size=3, stride=(1, 2), padding=1,
                      dilation=1, groups=1, bias=True)
        self.en_conv3 = ComplexConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1,
                      dilation=1, groups=1, bias=True)
        self.en_conv4 = ComplexConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
                      dilation=1, groups=1, bias=True)
        self.en_conv5 = ComplexConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1,
                      dilation=1, groups=1, bias=True)
        self.en_conv6 = ComplexConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1,
                      dilation=1, groups=1, bias=True)

        ############## Decoder

        self.de_conv1 = ComplexConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1,
                      output_padding=0, dilation=1, groups=1, bias=True, padding_mode = 'zeros')
        self.de_conv2 = ComplexConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                      output_padding=1, dilation=1, groups=1, bias=True, padding_mode = 'zeros')
        self.de_conv3 = ComplexConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                      output_padding=1, dilation=1, groups=1, bias=True, padding_mode = 'zeros')
        self.de_conv4 = ComplexConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,
                      output_padding=1, dilation=1, groups=1, bias=True, padding_mode = 'zeros')
        self.de_conv5 = ComplexConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1,
                      output_padding=1, dilation=1, groups=1, bias=True, padding_mode = 'zeros')
        self.de_conv6 = ComplexConvTranspose2d(in_channels=16, out_channels=n_out_channels, kernel_size=3, stride=1, padding=1,
                      output_padding=0, dilation=1, groups=1, bias=True, padding_mode = 'zeros')

    def forward(self, x):

        en1 = self.en_conv1(x)
        en2 = self.en_conv2(en1)
        en3 = self.en_conv3(en2)
        en4 = self.en_conv4(en3)
        en5 = self.en_conv5(en4)
        en6 = self.en_conv6(en5)

        de1 = self.de_conv1(en6)
        de2 = self.de_conv2(de1)
        de3 = self.de_conv3(de2)
        de4 = self.de_conv4(de3)
        de5 = self.de_conv5(de4)
        de6 = self.de_conv6(de5)

    ####### for debugging
        # print('x:', x.size())
        #
        # en1 = self.en_conv1(x)
        # print('en1:', en1.size())
        # en2 = self.en_conv2(en1)
        # print('en2:', en2.size())
        # en3 = self.en_conv3(en2)
        # print('en3:', en3.size())
        # en4 = self.en_conv4(en3)
        # print('en4:', en4.size())
        # en5 = self.en_conv5(en4)
        # print('en5:', en5.size())
        # en6 = self.en_conv6(en5)
        # print('en6:', en6.size())
        #
        # de1 = self.de_conv1(en6)
        # print('de1:', de1.size())
        # de2 = self.de_conv2(de1)
        # print('de2:', de2.size())
        # de3 = self.de_conv3(de2)
        # print('de3:', de3.size())
        # de4 = self.de_conv4(de3)
        # print('de4:', de4.size())
        # de5 = self.de_conv5(de4)
        # print('de5:', de5.size())
        # de6 = self.de_conv6(de5)
        # print('de6:', de6.size())

        return {'output': de6}



def get_net(model_config):

    model = model_config['name']
    in_channels = model_config['in_channels']
    out_channels = model_config['out_channels']

    if model == 'CV_SAR_SuperRes_Net':
        net = CV_SAR_SuperRes_Net(n_in_channels=in_channels, n_out_channels=out_channels)
    else:
        assert 'Network is not defined'

    return net

# ################################## Test and debug
# model_config = {'name': 'CV_SAR_SuperRes_Net',
#                 'in_channels': 2,
#                 'out_channels': 2
#                 }
# net = get_net(model_config)
# inp = torch.rand((1, 2, 32, 128)) + 1j*torch.rand((1, 2, 32, 128))
# outp = net(inp)
# print('outp size is:', outp.size())
#
#




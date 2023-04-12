"""
Based on "R. M.Asiyabi, M. Datcu, A. Anghel, H. Nies, "Complex-Valued End-to-end Deep
 Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and
  Classification" IEEE Transactions on Geoscience and Remote Sensing (2023)"
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, Flatten, AvgPool2d
from RV.RV_Net_Parts import RV_DoubleConv, RV_Down_DoubleConv2d, RV_Up, RV_OutConv, RV_Linear
from Src.complexFunctions_Torch import summary

class end_to_end_Net(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_classes, bilinear=False,
                 feature_extractor=False):
        super(end_to_end_Net, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_extractor = feature_extractor

        ############## Autoencoder

        self.inc1 = RV_DoubleConv(in_channels=n_in_channels, out_channels=16, mid_channels=None, kernel_size=3)
        self.down1 = RV_Down_DoubleConv2d(in_channels=16, out_channels=32, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        self.down2 = RV_Down_DoubleConv2d(in_channels=32, out_channels=64, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        self.down3 = RV_Down_DoubleConv2d(in_channels=64, out_channels=128, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        factor = 2 if bilinear else 1
        self.down4 = RV_Down_DoubleConv2d(in_channels=128, out_channels=256 // factor, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)

        self.feature = RV_DoubleConv(in_channels=256 // factor, out_channels=256 // factor, mid_channels=None,
                                          kernel_size=3)

        self.up1 = RV_Up(in_channels=256, out_channels=128 // factor, bilinear=self.bilinear, Conv2d_kernel_size=3,
                      ConvTrans_kernel_size=2)
        self.up2 = RV_Up(in_channels=128, out_channels=64 // factor, bilinear=self.bilinear, Conv2d_kernel_size=3,
                              ConvTrans_kernel_size=2)
        self.up3 = RV_Up(in_channels=64, out_channels=32 // factor, bilinear=self.bilinear, Conv2d_kernel_size=3,
                              ConvTrans_kernel_size=2)
        self.up4 = RV_Up(in_channels=32, out_channels=16, bilinear=self.bilinear, Conv2d_kernel_size=3,
                              ConvTrans_kernel_size=2)
        self.outc1 = RV_OutConv(in_channels=16, out_channels=n_out_channels, kernel_size=1)

        ############### Classifier

        self.conv1 = Conv2d(in_channels=128, out_channels=200, kernel_size=3, stride=1, padding='same',
                                   dilation=1, groups=1, bias=True)
        self.ReLU1 = ReLU(inplace=False)
        self.pool1 = AvgPool2d(kernel_size=2, stride=None, padding=0,ceil_mode=False,
                               count_include_pad=True, divisor_override=None)
        self.conv2 = Conv2d(in_channels=200, out_channels=256, kernel_size=3, stride=1, padding='same',
                                   dilation=1, groups=1, bias=True)
        self.ReLU2 = ReLU(inplace=False)
        self.flat1 = Flatten()
        self.lin1 = RV_Linear(in_channels=2304, out_channels=2000, activation="relu")
        self.lin2 = RV_Linear(in_channels=2000, out_channels=1000, activation="relu")
        self.lin3 = RV_Linear(in_channels=1000, out_channels=500, activation="relu")
        self.lin4 = RV_Linear(in_channels=500, out_channels=100, activation="relu")
        self.out1 = RV_Linear(in_channels=100, out_channels=n_classes, activation="None")


    def forward(self, x):
        if self.feature_extractor == True:
            i1 = self.inc1(x)
            d1 = self.down1(i1)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            f = self.feature(d4)
            u1 = self.up1(f, d3)
            u2 = self.up2(u1, d2)
            u3 = self.up3(u2, d1)
            u4 = self.up4(u3, i1)
            reconstructed = self.outc1(u4)
            return f
        else:
            i1 = self.inc1(x)
            d1 = self.down1(i1)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            f = self.feature(d4)
            u1 = self.up1(f, d3)
            u2 = self.up2(u1, d2)
            u3 = self.up3(u2, d1)
            u4 = self.up4(u3, i1)
            reconstructed = self.outc1(u4)

            c1 = self.ReLU1(self.conv1(f))
            c2 = self.pool1(c1)
            c3 = self.ReLU2(self.conv2(c2))
            c4 = self.flat1(c3)
            c5 = self.lin1(c4)
            c6 = self.lin2(c5)
            c7 = self.lin3(c6)
            c8 = self.lin4(c7)
            classified = abs(self.out1(c8))

            return reconstructed, classified
            # return reconstructed
model = end_to_end_Net(n_in_channels=8, n_out_channels=2, n_classes=7, bilinear=True)

# #### Print the summary of the model
# summary(model, input_size=(8, 100, 100), batch_size=10, device='cpu', dtypes=[torch.float])
#
# def get_n_params(model):
#     pp=0
#     for p in list(model.parameters()):
#         nn=1
#         for s in list(p.size()):
#             nn = nn*s
#         pp += nn
#     return pp

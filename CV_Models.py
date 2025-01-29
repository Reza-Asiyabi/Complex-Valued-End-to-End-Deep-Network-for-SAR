"""
Based on "R. M.Asiyabi, M. Datcu, A. Anghel, H. Nies, "Complex-Valued End-to-end Deep
 Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and
  Classification" IEEE Transactions on Geoscience and Remote Sensing (2023)"
"""

import torch
import torch.nn as nn
from Src.CV_Net_Modules import Complex_DoubleConv, Complex_Down_DoubleConv2d, Complex_Up, Complex_OutConv
from Src.CV_Net_Modules import Complex_Linear, Complex_Flatten
from Src.CV_Functions import ComplexReLU, ComplexConv2d, Complexavg_pool2d


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

        self.inc1 = Complex_DoubleConv(in_channels=n_in_channels, out_channels=16, mid_channels=None, kernel_size=3)
        self.down1 = Complex_Down_DoubleConv2d(in_channels=16, out_channels=32, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        self.down2 = Complex_Down_DoubleConv2d(in_channels=32, out_channels=64, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        self.down3 = Complex_Down_DoubleConv2d(in_channels=64, out_channels=128, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        factor = 2 if bilinear else 1
        self.down4 = Complex_Down_DoubleConv2d(in_channels=128, out_channels=256 // factor, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)

        self.feature = Complex_DoubleConv(in_channels=256 // factor, out_channels=256 // factor, mid_channels=None,
                                          kernel_size=3)

        self.up1 = Complex_Up(in_channels=256, out_channels=128 // factor, bilinear=self.bilinear, Conv2d_kernel_size=3,
                              ConvTrans_kernel_size=2)
        self.up2 = Complex_Up(in_channels=128, out_channels=64 // factor, bilinear=self.bilinear, Conv2d_kernel_size=3,
                              ConvTrans_kernel_size=2)
        self.up3 = Complex_Up(in_channels=64, out_channels=32 // factor, bilinear=self.bilinear, Conv2d_kernel_size=3,
                              ConvTrans_kernel_size=2)
        self.up4 = Complex_Up(in_channels=32, out_channels=16, bilinear=self.bilinear, Conv2d_kernel_size=3,
                              ConvTrans_kernel_size=2)
        self.outc1 = Complex_OutConv(in_channels=16, out_channels=n_out_channels, kernel_size=1)



        ############### Classifier

        # self.flatten = Complex_Flatten()
        # self.inc2 = Complex_Linear(in_channels=4608, out_channels=2000, activation="relu")
        # self.hidden1 = Complex_Linear(in_channels=2000, out_channels=1000, activation="relu")
        # self.hidden2 = Complex_Linear(in_channels=1000, out_channels=500, activation="relu")
        # self.hidden3 = Complex_Linear(in_channels=500, out_channels=250, activation="relu")
        # self.hidden4 = Complex_Linear(in_channels=250, out_channels=100, activation="relu")
        # self.outc2 = Complex_Linear(in_channels=100, out_channels=n_classes, activation="sigmoid")
        # self.outc3 = Complex_Softmax()

        self.conv1 = ComplexConv2d(in_channels=128, out_channels=200, kernel_size=3, stride=1, padding='same',
                      dilation=1, groups=1, bias=True)
        self.ReLU1 = ComplexReLU()
        self.pool1 = Complexavg_pool2d(kernel_size=2, stride=None, padding=0,
                              dilation=1, return_indices=False, ceil_mode=False,
                              count_include_pad=True, divisor_override=None)
        self.conv2 = ComplexConv2d(in_channels=200, out_channels=256, kernel_size=3, stride=1, padding='same',
                                     dilation=1, groups=1, bias=True)
        self.ReLU2 = ComplexReLU()
        self.flat1 = Complex_Flatten()
        self.lin1 = Complex_Linear(in_channels=2304, out_channels=2000, activation="relu")
        self.lin2 = Complex_Linear(in_channels=2000, out_channels=1000, activation="relu")
        self.lin3 = Complex_Linear(in_channels=1000, out_channels=500, activation="relu")
        self.lin4 = Complex_Linear(in_channels=500, out_channels=100, activation="relu")
        self.out1 = Complex_Linear(in_channels=100, out_channels=n_classes, activation="None")



    def forward(self, x):

        i1 = self.inc1(x)
        d1 = self.down1(i1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        f = self.feature(d4)
        if self.feature_extractor == True:
            return f
        else:
            u1 = self.up1(f, d3)
            u2 = self.up2(u1, d2)
            u3 = self.up3(u2, d1)
            u4 = self.up4(u3, i1)
            reconstructed = self.outc1(u4)

            # f = self.flatten(f)
            # c1 = self.inc2(f)
            # c2 = self.hidden1(c1)
            # c3 = self.hidden2(c2)
            # c4 = self.hidden3(c3)
            # c5 = self.hidden4(c4)
            # o1 = self.outc2(c5)
            # classified = self.outc3(o1)

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

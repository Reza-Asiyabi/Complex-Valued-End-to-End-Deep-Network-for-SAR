"""
Based on "R. M.Asiyabi, M. Datcu, A. Anghel, H. Nies, "Complex-Valued End-to-end Deep
 Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and
  Classification" IEEE Transactions on Geoscience and Remote Sensing (2023)"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from CV_Functions import ComplexBatchNorm1d, ComplexBatchNorm2d, ComplexConv1d, ComplexConv2d, ComplexLinear, ComplexConvTranspose1d, ComplexConvTranspose2d, ComplexReLU, ComplexLeakyReLU, ComplexSigmoid, Complexavg_pool1d, Complexavg_pool2d, ComplexUpsample

class Complex_DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, activation="Relu"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if activation == "Relu":
            self.double_conv = nn.Sequential(
                ComplexConv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True),
                ComplexBatchNorm2d(num_features=mid_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                ComplexReLU(),
                ComplexConv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True),
                ComplexBatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                ComplexReLU()
            )
        elif activation == "LeakyRelu":
            self.double_conv = nn.Sequential(
                ComplexConv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True),
                ComplexBatchNorm2d(num_features=mid_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                ComplexLeakyReLU(),
                ComplexConv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True),
                ComplexBatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                ComplexLeakyReLU()
            )
        else:
            print("Activation function not defined")

    def forward(self, x):
        return self.double_conv(x)

class Complex_DoubleConv_1d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ComplexConv1d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True),
            ComplexBatchNorm1d(num_features=mid_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            ComplexReLU(),
            ComplexConv1d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True),
            ComplexBatchNorm1d(num_features=out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            ComplexReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Complex_DoubleTransposeConv(nn.Module):
    """(Transpose convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ComplexConvTranspose2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True),
            ComplexBatchNorm2d(num_features=mid_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            ComplexReLU(),
            ComplexConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True),
            ComplexBatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            ComplexReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Complex_DoubleTransposeConv_1d(nn.Module):
    """(Transpose convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ComplexConvTranspose1d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True),
            ComplexBatchNorm1d(num_features=mid_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            ComplexReLU(),
            ComplexConvTranspose1d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True),
            ComplexBatchNorm1d(num_features=out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            ComplexReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Complex_Down_DoubleConv1d(nn.Module):
    """Downscaling with avgpool then double conv"""

    def __init__(self, in_channels, out_channels, Conv2d_kernel_size=3, Pooling_kernel_size=2):
        super().__init__()
        self.pool_conv = nn.Sequential(
            # Complexmax_pool2d(kernel_size=Pooling_kernel_size, stride=1, padding=0,
            #                   dilation=1, ceil_mode=False, return_indices=False),
            Complexavg_pool1d(kernel_size=Pooling_kernel_size, stride=None, padding=0,
                              dilation=1, return_indices=False, ceil_mode=False,
                              count_include_pad=True),
            Complex_DoubleConv_1d(in_channels, out_channels, mid_channels=None, kernel_size=Conv2d_kernel_size)
        )

    def forward(self, x):
        return self.pool_conv(x)

class Complex_Down_DoubleConv2d(nn.Module):
    """Downscaling with avgpool then double conv"""

    def __init__(self, in_channels, out_channels, Conv2d_kernel_size=3, Pooling_kernel_size=2):
        super().__init__()
        self.pool_conv = nn.Sequential(
            # Complexmax_pool2d(kernel_size=Pooling_kernel_size, stride=1, padding=0,
            #                   dilation=1, ceil_mode=False, return_indices=False),
            Complexavg_pool2d(kernel_size=Pooling_kernel_size, stride=None, padding=0,
                              dilation=1, return_indices=False, ceil_mode=False,
                              count_include_pad=True, divisor_override=None),
            Complex_DoubleConv(in_channels, out_channels, mid_channels=None, kernel_size=Conv2d_kernel_size)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Complex_Up(nn.Module):
    """Upscaling then double Transpose conv and with Skip connection"""

    def __init__(self, in_channels, out_channels, bilinear=True, Conv2d_kernel_size=3, ConvTrans_kernel_size=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = ComplexUpsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.Tconv = Complex_DoubleTransposeConv(in_channels, out_channels, mid_channels=in_channels // 2, kernel_size=Conv2d_kernel_size)
        else:
            self.up = ComplexConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ConvTrans_kernel_size, stride=2, padding=0,
                                             output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
            self.Tconv = Complex_DoubleTransposeConv(in_channels, out_channels, mid_channels=None, kernel_size=Conv2d_kernel_size)


    def forward(self, x1, x2):
        # print("x1", x1.size())
        # print("x2", x2.size())
        x1 = self.up(x1)
        # print("x1", x1.size())
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # print("diffX", diffX)
        # print("diffY", diffY)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # print("x1", x1.size())
        '''
        if you have padding issues, see
        https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        '''
        x = torch.cat([x2, x1], dim=1)
        # print("x", x.size())
        return self.Tconv(x)

class Complex_Up_1d(nn.Module):
    """Upscaling then double Transpose conv and with Skip connection"""

    def __init__(self, in_channels, out_channels, bilinear=True, Conv2d_kernel_size=3, ConvTrans_kernel_size=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = ComplexUpsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.Tconv = Complex_DoubleTransposeConv_1d(in_channels, out_channels, mid_channels=in_channels // 2, kernel_size=Conv2d_kernel_size)
        else:
            self.up = ComplexConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ConvTrans_kernel_size, stride=2, padding=0,
                                             output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
            self.Tconv = Complex_DoubleTransposeConv_1d(in_channels, out_channels, mid_channels=None, kernel_size=Conv2d_kernel_size)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        '''
        if you have padding issues, see
        https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        '''
        x = torch.cat([x2, x1], dim=1)
        return self.Tconv(x)

class Complex_Up_Noskip(nn.Module):
    """Upscaling then double Transpose conv without skip connection"""

    def __init__(self, in_channels, out_channels, bilinear=True, Conv2d_kernel_size=3, ConvTrans_kernel_size=2, pad=0):
        super().__init__()
        self.pad = pad
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = ComplexUpsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.Tconv = Complex_DoubleTransposeConv(in_channels, out_channels, mid_channels=in_channels // 2, kernel_size=Conv2d_kernel_size)
        else:
            self.up = ComplexConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ConvTrans_kernel_size, stride=2, padding=0,
                                             output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
            self.Tconv = Complex_DoubleTransposeConv(in_channels, out_channels, mid_channels=None, kernel_size=Conv2d_kernel_size)

    def forward(self, x):
        x = self.up(x)
        if self.pad > 0:
            x = F.pad(x, [self.pad // 2, self.pad - self.pad // 2,
                        self.pad // 2, self.pad - self.pad // 2])
        x = self.Tconv(x)
        return x

class Complex_Up_Noskip_1d(nn.Module):
    """Upscaling then double Transpose conv without skip connection"""

    def __init__(self, in_channels, out_channels, bilinear=True, Conv2d_kernel_size=3, ConvTrans_kernel_size=2, pad=0):
        super().__init__()
        self.pad = pad
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = ComplexUpsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.Tconv = Complex_DoubleTransposeConv_1d(in_channels, out_channels, mid_channels=in_channels // 2, kernel_size=Conv2d_kernel_size)
        else:
            # self.up = ComplexConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=ConvTrans_kernel_size, stride=2, padding=0,
            #                                  output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
            self.up = ComplexUpsample(scale_factor=2, mode='nearest', align_corners=True)
            self.Tconv = Complex_DoubleTransposeConv_1d(in_channels, out_channels, mid_channels=None, kernel_size=Conv2d_kernel_size)

    def forward(self, x):
        x = self.up(x)
        if self.pad > 0:
            x = F.pad(x, [self.pad // 2, self.pad - self.pad // 2])
        x = self.Tconv(x)
        return x

class Complex_OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Complex_OutConv, self).__init__()
        self.out = ComplexConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

    def forward(self, x):
        return self.out(x)

class Complex_OutConv_1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Complex_OutConv_1d, self).__init__()
        self.out = ComplexConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

    def forward(self, x):
        return self.out(x)

class Complex_Linear(nn.Module):
    def __init__(self, in_channels, out_channels, activation="relu"):
        super(Complex_Linear, self).__init__()
        if activation == "relu":
            self.lin = nn.Sequential(
                ComplexLinear(in_features=in_channels, out_features=out_channels),
                ComplexReLU(),
            )
        elif activation == "sigmoid":
            self.lin = nn.Sequential(
                ComplexLinear(in_features=in_channels, out_features=out_channels),
                ComplexSigmoid(),
            )
        elif activation == "None":
            self.lin = nn.Sequential(
                ComplexLinear(in_features=in_channels, out_features=out_channels)
            )

    def forward(self, x):
        return self.lin(x)

class Complex_Flatten(nn.Module):
    def __init__(self):
        super(Complex_Flatten, self).__init__()
        self.Flatten = nn.Flatten()

    def forward(self, x):
        return self.Flatten(x)

class Complex_Softmax(nn.Module):
    def __init__(self):
        super(Complex_Softmax, self).__init__()
        self.Softmax = nn.Softmax()

    def forward(self, x):
        s_r = self.Softmax(x.real)
        s_i = self.Softmax(x.imag)
        return s_r + 1j*s_i

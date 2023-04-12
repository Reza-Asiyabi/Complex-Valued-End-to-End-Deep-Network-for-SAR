"""
Based on "R. M.Asiyabi, M. Datcu, A. Anghel, H. Nies, "Complex-Valued End-to-end Deep
 Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and
  Classification" IEEE Transactions on Geoscience and Remote Sensing (2023)"
"""

import torch
import torch.nn as nn
from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, BatchNorm2d, ReLU, Sigmoid, AvgPool2d, Upsample, Linear
from torch.nn import ConvTranspose2d
import torch.nn.functional as F

class RV_DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding = 1,
                 dilation=1, groups=1, bias=True),
            BatchNorm2d(num_features=mid_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=False),
            Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1,
                   dilation=1, groups=1, bias=True),
            BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)

class RV_Down_DoubleConv2d(nn.Module):
    """Downscaling with avgpool then double conv"""

    def __init__(self, in_channels, out_channels, Conv2d_kernel_size=3, Pooling_kernel_size=2):
        super().__init__()
        self.pool_conv = nn.Sequential(
            AvgPool2d(kernel_size=Pooling_kernel_size, stride=None, padding=0,
                              ceil_mode=False, count_include_pad=True, divisor_override=None),
            RV_DoubleConv(in_channels, out_channels, mid_channels=None, kernel_size=Conv2d_kernel_size)
        )

    def forward(self, x):
        return self.pool_conv(x)

class RV_DoubleTransposeConv(nn.Module):
    """(Transpose convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ConvTranspose2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1,
                            padding=1, dilation=1, groups=1, bias=True),
            BatchNorm2d(num_features=mid_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=False),
            ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                            padding=1, dilation=1, groups=1, bias=True),
            BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)

class RV_Up(nn.Module):
    """Upscaling then double Transpose conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, Conv2d_kernel_size=3, ConvTrans_kernel_size=2):
        super().__init__()

        self.up = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Tconv = RV_DoubleTransposeConv(in_channels, out_channels, mid_channels=in_channels // 2, kernel_size=Conv2d_kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.Tconv(x)

class RV_OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(RV_OutConv, self).__init__()
        self.out = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

    def forward(self, x):
        return self.out(x)

class RV_Linear(nn.Module):
    def __init__(self, in_channels, out_channels, activation="relu"):
        super(RV_Linear, self).__init__()
        if activation == "relu":
            self.lin = nn.Sequential(
                Linear(in_features=in_channels, out_features=out_channels),
                ReLU(),
            )
        elif activation == "sigmoid":
            self.lin = nn.Sequential(
                Linear(in_features=in_channels, out_features=out_channels),
                Sigmoid(),
            )
        elif activation == "None":
            self.lin = nn.Sequential(
                Linear(in_features=in_channels, out_features=out_channels)
            )

    def forward(self, x):
        return self.lin(x)

class RV_2foldloss(Module):
    '''
    Computs the MSE loss for the reconstruction and the classification and average them with the regularization term (alpha).
    '''

    def __init__(self, image_max, alpha=0.5):
        super(RV_2foldloss, self).__init__()
        self.alpha = alpha
        self.image_max = image_max

    def forward(self, reconstroction_predicted, reconstroction_target, classification_predicted, classification_target):

        reconstroction_loss_madule = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        reconstroction_loss = reconstroction_loss_madule(reconstroction_predicted, reconstroction_target)

        classification_loss_madule = torch.nn.CrossEntropyLoss()
        classification_loss = classification_loss_madule(classification_predicted, classification_target.long())

        loss = self.alpha*reconstroction_loss + (1-self.alpha)*classification_loss

        return loss, reconstroction_loss, classification_loss
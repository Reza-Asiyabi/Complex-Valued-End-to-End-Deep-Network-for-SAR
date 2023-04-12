"""
Based on "R. M.Asiyabi, M. Datcu, A. Anghel, H. Nies, "Complex-Valued End-to-end Deep
 Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and
  Classification" IEEE Transactions on Geoscience and Remote Sensing (2023)"
"""

import torch
from torch.nn import Module, Parameter, init
from torch.nn import Conv1d, Conv2d, Linear
from torch.nn import ConvTranspose1d, ConvTranspose2d
import torch.nn.functional as F
from Src.complexFunctions_Torch import complex_relu, complex_leaky_relu, complex_sigmoid, complex_MSE_loss, complex_CrossEntropy_loss, complex_avg_pool1d, complex_avg_pool2d, complex_Upsample
from Src import Coherence

def apply_complex(fr, fi, input):

    return (fr(input.real)-fi(input.imag)) \
           + 1j*(fr(input.imag)+fi(input.real))


#
class ComplexReLU(Module):
    '''
    Perform element-wise rectified linear unit function.
    '''

    def forward(self, input):
        return complex_relu(input)

class ComplexLeakyReLU(Module):
    '''
    Perform element-wise Leaky rectified linear unit function.
    '''

    def forward(self, input):
        return complex_leaky_relu(input)

class ComplexSigmoid(Module):
    '''
    Perform element-wise Sigmoid function.
    '''

    def forward(self, input):
        return complex_sigmoid(input)

class Complexavg_pool1d(Module):
    '''
    Perform 1D complex average pooling.
    '''

    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False, count_include_pad=True):
        super(Complexavg_pool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices
        self.count_include_pad = count_include_pad

    def forward(self, input):
        return complex_avg_pool1d(input, self.kernel_size, stride=self.stride, padding=self.padding,
                                  ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad)

class Complexavg_pool2d(Module):
    '''
    Perform complex average pooling.
    '''

    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super(Complexavg_pool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input):
        return complex_avg_pool2d(input, self.kernel_size, stride=self.stride, padding=self.padding,
                                  ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad,
                                  divisor_override=self.divisor_override)

# class Complexmax_pool2d(Module):
#     '''
#     Perform complex max pooling by selecting on the absolute value on the complex values.
#     Not approved yet
#     '''
#
#     def __init__(self, kernel_size=3, stride=1, padding=0,
#                  dilation=1, return_indices=False, ceil_mode=False):
#         super(Complexmax_pool2d, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.ceil_mode = ceil_mode
#         self.return_indices = return_indices
#
#     def forward(self, input):
#         return complex_max_pool2d(input=input, kernel_size = self.kernel_size,
#                                   stride = self.stride, padding = self.padding,
#                                   dilation = self.dilation, ceil_mode = self.ceil_mode,
#                                   return_indices = self.return_indices)
#
class ComplexUpsample(Module):
    '''
    Upsamples a given multi-channel compelx data.
    '''

    def __init__(self, scale_factor=2, mode='nearest', align_corners=True):

        super(ComplexUpsample, self).__init__()

        self.complex_upsample_func = complex_Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    def forward(self, input):
        complex_upsample_r = self.complex_upsample_func(input.real)
        complex_upsample_i = self.complex_upsample_func(input.imag)
        return complex_upsample_r + 1j*complex_upsample_i

class ComplexConv1d(Module):
    '''
    Applies a 1D complex convolution over an input signal composed of several input planes
    '''

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        self.conv_r = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self,input):
        return apply_complex(self.conv_r, self.conv_i, input)

class ComplexConv2d(Module):
    '''
    Applies a 2D complex convolution over an input signal composed of several input planes
    '''

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self,input):
        return apply_complex(self.conv_r, self.conv_i, input)

class ComplexConvTranspose1d(Module):
    '''
    Perform complex 1D transposed convolution operator over an input image composed of several input planes.
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):

        super(ComplexConvTranspose1d, self).__init__()

        self.conv_tran_r = ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)


    def forward(self,input):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, input)

class ComplexConvTranspose2d(Module):
    '''
    Perform complex 2D transposed convolution operator over an input image composed of several input planes.
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):

        super(ComplexConvTranspose2d, self).__init__()

        self.conv_tran_r = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)


    def forward(self,input):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, input)

class ComplexLinear(Module):
    '''
    Applies a complex linear transformation to the data
    '''

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)

class _ComplexBatchNorm(Module):
    '''
    The complex BatchNorm implementation which requires the calculation of the inverse square root of the covariance matrix.
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,3))
            self.bias = Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype = torch.complex64))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:,:2],1.4142135623730951)
            init.zeros_(self.weight[:,2])
            init.zeros_(self.bias)

class ComplexBatchNorm2d(_ComplexBatchNorm):
    '''
    The complex BatchNorm implementation which requires the calculation of the inverse square root of the covariance matrix.
    '''

    def forward(self, input):
        exponential_average_factor = 0.0


        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            # mean_r = input.real.mean([0, 2, 3]).type(torch.complex64)
            # mean_i = input.imag.mean([0, 2, 3]).type(torch.complex64)
            mean_r = input.real.mean([0, 2, 3])
            mean_i = input.imag.mean([0, 2, 3])
            mean = mean_r + 1j*mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, :, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = 1./n*input.real.pow(2).sum(dim=[0,2,3])+self.eps
            Cii = 1./n*input.imag.pow(2).sum(dim=[0,2,3])+self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[0,2,3])
        else:
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]#+self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1) \
                                          + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1) \
                                          + (1 - exponential_average_factor) * self.running_covar[:,1]

                self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1) \
                                          + (1 - exponential_average_factor) * self.running_covar[:,2]

        # calculate the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        # input = (Rrr[None,:,None,None]*input.real+Rri[None,:,None,None]*input.imag).type(torch.complex64) \
        #         + 1j*(Rii[None,:,None,None]*input.imag+Rri[None,:,None,None]*input.real).type(torch.complex64)
        input = (Rrr[None,:,None,None]*input.real+Rri[None,:,None,None]*input.imag) \
                + 1j*(Rii[None,:,None,None]*input.imag+Rri[None,:,None,None]*input.real)

        if self.affine:
            # input = (self.weight[None,:,0,None,None]*input.real+self.weight[None,:,2,None,None]*input.imag+ \
            #          self.bias[None,:,0,None,None]).type(torch.complex64) \
            #         +1j*(self.weight[None,:,2,None,None]*input.real+self.weight[None,:,1,None,None]*input.imag+ \
            #              self.bias[None,:,1,None,None]).type(torch.complex64)
            input = (self.weight[None,:,0,None,None]*input.real+self.weight[None,:,2,None,None]*input.imag+ \
                     self.bias[None,:,0,None,None]) \
                    +1j*(self.weight[None,:,2,None,None]*input.real+self.weight[None,:,1,None,None]*input.imag+ \
                         self.bias[None,:,1,None,None])

        return input


class ComplexBatchNorm1d(_ComplexBatchNorm):
    '''
    The complex BatchNorm implementation which requires the calculation of the inverse square root of the covariance matrix.
    '''

    def forward(self, input):

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            mean_r = input.real.mean(dim=[0, 2])
            mean_i = input.imag.mean(dim=[0, 2])
            mean = mean_r + 1j*mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, :, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = input.real.var(dim=[0, 2],unbiased=False)+self.eps
            Cii = input.imag.var(dim=[0, 2],unbiased=False)+self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[0, 2])
        else:
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]

        if self.training and self.track_running_stats:
            self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1) \
                                      + (1 - exponential_average_factor) * self.running_covar[:,0]

            self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1) \
                                      + (1 - exponential_average_factor) * self.running_covar[:,1]

            self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1) \
                                      + (1 - exponential_average_factor) * self.running_covar[:,2]

        # calculate the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None, :, None]*input.real+Rri[None, :, None]*input.imag) \
                + 1j*(Rii[None, :, None]*input.imag+Rri[None, :, None]*input.real)

        if self.affine:
            input = (self.weight[None,:,0,None]*input.real+self.weight[None,:,2,None]*input.imag+ \
                     self.bias[None,:,0,None]) \
                    +1j*(self.weight[None,:,2,None]*input.real+self.weight[None,:,1,None]*input.imag+ \
                         self.bias[None,:,1,None])


        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return input

class ComplexMSEloss(Module):
    '''
    Computs the loss by subtracting real part and imaginary part.
    '''

    def __init__(self):
        super(ComplexMSEloss, self).__init__()

    def forward(self, predicted, target):

        loss = complex_MSE_loss()
        loss_r = loss(predicted.real, target.real)
        loss_i = loss(predicted.imag, target.imag)

        return (loss_r + loss_i)/2

class ComplexCrossEntropyloss(Module):
    '''
    Computs the cross entropy loss.
    '''

    def __init__(self):
        super(ComplexCrossEntropyloss, self).__init__()

    def forward(self, predicted, target):

        loss_madule = complex_CrossEntropy_loss()
        loss = loss_madule(predicted, target)

        return loss

class Complex2foldloss(Module):
    '''
    Computs the MSE loss for the reconstruction and the classification and average them with the regularization term (alpha).
    '''

    def __init__(self, alpha=0.5):
        super(Complex2foldloss, self).__init__()
        self.alpha = alpha

    def forward(self, reconstroction_predicted, reconstroction_target, classification_predicted, classification_target):

        reconstroction_loss_madule = complex_MSE_loss()
        reconstroction_loss_r = reconstroction_loss_madule(reconstroction_predicted.real, reconstroction_target.real)
        reconstroction_loss_i = reconstroction_loss_madule(reconstroction_predicted.imag, reconstroction_target.imag)

        # classification_loss_madule = complex_CrossEntropy_loss()
        # classification_loss_r = classification_loss_madule(classification_predicted.real, classification_target.real.long())
        # classification_loss_i = classification_loss_madule(classification_predicted.imag, classification_target.imag.long())

        classification_loss_madule = torch.nn.CrossEntropyLoss()
        classification_loss = classification_loss_madule(classification_predicted, classification_target.long())

        reconstroction_loss = (reconstroction_loss_r + reconstroction_loss_i)/2
        # classification_loss = (classification_loss_r + classification_loss_i)/2

        loss = self.alpha*reconstroction_loss + (1-self.alpha)*classification_loss
        # print("reconstroction_loss", reconstroction_loss)
        # print("classification_loss", classification_loss)
        # print("loss", loss)

        return loss, reconstroction_loss, classification_loss

class Complex2foldloss_Coh(Module):
    '''
    Computs the Coherency loss for the reconstruction and the classification and average them with the regularization term (alpha).
    '''

    def __init__(self, alpha=0.5):
        super(Complex2foldloss_Coh, self).__init__()
        self.alpha = alpha

    def forward(self, reconstroction_predicted, reconstroction_target, classification_predicted, classification_target):

        reconstroction_loss = 1 - torch.mean(Coherence.complex_correlation(inputdata1=reconstroction_predicted, inputdata2=reconstroction_target, k=2))

        classification_loss_madule = torch.nn.CrossEntropyLoss()
        classification_loss = classification_loss_madule(classification_predicted, classification_target.long())

        loss = self.alpha*reconstroction_loss + (1-self.alpha)*classification_loss
        # print("reconstroction_loss", reconstroction_loss)
        # print("classification_loss", classification_loss)
        # print("loss:", loss)

        return loss, reconstroction_loss, classification_loss

class ComplexCoherencyloss_1d(Module):
    '''
    Computs the loss by averaging the coherency of amplithud and phase between the input and output 1D vectors.
    '''

    def __init__(self):
        super(ComplexCoherencyloss_1d, self).__init__()

    def forward(self, predicted, target, filter_size=5):


        filter = torch.ones(predicted.size(1), predicted.size(1), filter_size).cuda()

        c = (F.conv1d(torch.real(predicted * torch.conj(target)), filter, padding="valid") + 1j *
             F.conv1d(torch.imag(predicted * torch.conj(target)), filter, padding="valid")) / \
            (torch.sqrt(F.conv1d(torch.abs(predicted) ** 2, filter, padding="valid") *
                        F.conv1d(torch.abs(target) ** 2, filter, padding="valid")))

        coh = torch.mean(abs(c))

        return (1 - coh)


class ComplexCoherencyloss(Module):
    '''
    Computs the loss by averaging the coherency of amplithud and phase between the input and output images.
    '''

    def __init__(self):
        super(ComplexCoherencyloss, self).__init__()

    def forward(self, predicted, target, filter_size=5):


        filter = torch.ones(predicted.size(1), predicted.size(1), filter_size, filter_size).cuda()

        c = (F.conv2d(torch.real(predicted * torch.conj(target)), filter, padding="valid") + 1j *
             F.conv2d(torch.imag(predicted * torch.conj(target)), filter, padding="valid")) / \
            (torch.sqrt(F.conv2d(torch.abs(predicted) ** 2, filter, padding="valid") *
                        F.conv2d(torch.abs(target) ** 2, filter, padding="valid")))

        coh = torch.mean(abs(c))
        # print("predicted", predicted.size())
        # print("target", target.size())
        # print("c", c.size())
        # print("coh", coh)
        return (1 - coh)
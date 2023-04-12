"""
Based on "R. M.Asiyabi, M. Datcu, A. Anghel, H. Nies, "Complex-Valued End-to-end Deep
 Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and
  Classification" IEEE Transactions on Geoscience and Remote Sensing (2023)"
"""

from torch.nn.functional import relu,leaky_relu, max_pool2d, avg_pool1d, avg_pool2d
from torch import sigmoid
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from torch import Tensor

def complex_matmul(A, B):
    '''
        Performs the matrix product between two complex matrices
    '''

    outp_real = torch.matmul(A.real, B.real) - torch.matmul(A.imag, B.imag)
    outp_imag = torch.matmul(A.real, B.imag) + torch.matmul(A.imag, B.real)

    return outp_real+ 1j * outp_imag

def complex_avg_pool2d(input, *args, **kwargs):
    '''
    Perform 2D complex average pooling.
    '''
    absolute_value_real = avg_pool2d(input.real, *args, **kwargs)
    absolute_value_imag = avg_pool2d(input.imag, *args, **kwargs)

    return absolute_value_real+1j*absolute_value_imag

def complex_avg_pool1d(input, *args, **kwargs):
    '''
    Perform 1D complex average pooling.
    '''
    absolute_value_real = avg_pool1d(input.real, *args, **kwargs)
    absolute_value_imag = avg_pool1d(input.imag, *args, **kwargs)

    return absolute_value_real+1j*absolute_value_imag

def complex_relu(input):
    '''
    Perform element-wise rectified linear unit function.
    '''
    return relu(input.real, inplace=False)+1j*relu(input.imag, inplace=False)

def complex_leaky_relu(input):
    '''
    Perform element-wise Leaky rectified linear unit function.
    '''
    return leaky_relu(input.real, negative_slope=0.01, inplace=False)+1j*leaky_relu(input.imag, negative_slope=0.01, inplace=False)

def complex_sigmoid(input):
    '''
    Perform element-wise sigmoid function.
    '''

    return sigmoid(input.real)+1j*sigmoid(input.imag)

# def _retrieve_elements_from_indices(tensor, indices):
#     flattened_tensor = tensor.flatten(start_dim=-2)
#     print("shape flatten_tensor:", flattened_tensor.shape)
#     print("shape indices:", indices.shape)
#     output = flattened_tensor.gather(dim=-1, index=indices.flatten(start_dim=-2)).view_as(indices)
#     return output
#
# def complex_max_pool2d(input, kernel_size, stride=None, padding=0,
#                        dilation=1, ceil_mode=False, return_indices=True):
#     '''
#     Perform complex max pooling by selecting on the absolute value on the complex values.
#     Not approved yet
#     '''
#     absolute_value, indices =  max_pool2d(
#                                input.abs(),
#                                kernel_size = kernel_size,
#                                stride = stride,
#                                padding = padding,
#                                dilation = dilation,
#                                ceil_mode = ceil_mode,
#                                return_indices = return_indices,
#                             )
#     # performs the selection on the absolute values
#     print(absolute_value)
#     print(indices.shape)
#     absolute_value = absolute_value
#     # retrieve the corresonding phase value using the indices
#     # unfortunately, the derivative for 'angle' is not implemented
#     angle = torch.atan2(input.imag,input.real)
#     # get only the phase values selected by max pool
#     angle = _retrieve_elements_from_indices(angle, indices.type(torch.int64))
#     return absolute_value \
#            * (torch.cos(angle)+1j*torch.sin(angle))

def complex_Upsample(scale_factor=2, mode='nearest', align_corners=True):
    '''
    Upsamples a given multi-channel compelx data.
    '''

    if mode == "linear" or mode == "bilinear" or mode == "bicubic" or mode == "trilinear":
        func_upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    else:
        func_upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    return func_upsample

# def complex_dropout(input, p=0.5, training=True):
#     '''
#     Randomly sets input units to 0 with a frequency of p at each step to prevent overfitting
#     '''
#
#     # need to have the same dropout mask for real and imaginary part,
#     # this not a clean solution!
#     mask = torch.ones_like(input).type(torch.float32)
#     mask = dropout(mask, p, training)*1/(1-p)
#     return mask*input
#
#
# def complex_dropout2d(input, p=0.5, training=True):
#     '''
#     Randomly sets input units to 0 with a frequency of p at each step to prevent overfitting
#     '''
#     # need to have the same dropout mask for real and imaginary part,
#     # this not a clean solution!
#     mask = torch.ones_like(input).type(torch.float32)
#     mask = dropout2d(mask, p, training)*1/(1-p)
#     return mask*input
#
def complex_MSE_loss():

    loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    return loss

def complex_CrossEntropy_loss():

    loss = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean')
    return loss
###############################################################################################################

'''
Ref: https://github.com/sksq96/pytorch-summary
How to call:
s = summary(model, input_size=(1, 128, 128), batch_size=-1, device='cpu', dtypes=len(input data)*[torch.complex64])
'''

def summary(model, input_size, batch_size=-1, device='cpu', dtypes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device='cpu', dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)

###############################################################################################################

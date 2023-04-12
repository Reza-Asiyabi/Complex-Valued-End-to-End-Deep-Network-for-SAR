"""
Based on "R. M.Asiyabi, M. Datcu, A. Anghel, H. Nies, "Complex-Valued End-to-end Deep
 Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and
  Classification" IEEE Transactions on Geoscience and Remote Sensing (2023)"
"""

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as f
from scipy import signal


def complex_correlation(inputdata1, inputdata2, k=1):

    ####### In the model
    # filter = torch.ones((1, inputdata1.size(1), 2*k+1, 2*k+1)).cuda()

    #Out of the model
    filter = torch.ones((1, 1, 2*k+1, 2*k+1)).cuda()
    # filter_c = torch.ones((1, 1, 2*k+1, 2*k+1)) + 0j*torch.ones((1, 1, 2*k+1, 2*k+1))
    # filter_c = filter_c.cuda()
    # inputdata1 = torch.tensor(inputdata1)
    inputdata1 = torch.unsqueeze(inputdata1, dim=0)
    inputdata1 = torch.unsqueeze(inputdata1, dim=0)
    # inputdata2 = torch.tensor(inputdata2)
    inputdata2 = torch.unsqueeze(inputdata2, dim=0)
    inputdata2 = torch.unsqueeze(inputdata2, dim=0)

    # print("input1 size:", inputdata1.size())
    # print("input2 size:", inputdata2.size())
    # print("filter size is:", filter.size())

    c = (f.conv2d(torch.real(inputdata1*torch.conj(inputdata2)), filter, padding="same") + 1j * f.conv2d(torch.imag(inputdata1*torch.conj(inputdata2)), filter, padding="same"))/ \
        (torch.sqrt(f.conv2d(torch.abs(inputdata1)**2, filter, padding="same")*
                    f.conv2d(torch.abs(inputdata2)**2, filter, padding="same")))

    return c



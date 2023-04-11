import torch
import numpy as np
import matplotlib.pyplot as plt
import torchac
from torch import nn
import torch.nn.functional as F
import os
import h5py

f = h5py.File("E:/Reza/DataSets/My Annotated dataset/Sao Paulo/Raw/s1a-s3-raw-s-hh-20210516t213550-20210516t213616-037918-0479a0_decoded_raw_data.mat")
#
for k, v in f.items():
    img = np.array(v)

HH = img['real'] + 1j*img['imag']

f = h5py.File("E:/Reza/DataSets/My Annotated dataset/Sao Paulo/Raw/s1a-s3-raw-s-hv-20210516t213550-20210516t213616-037918-0479a0_decoded_raw_data.mat")
#
for k, v in f.items():
    img = np.array(v)

HV = img['real'] + 1j*img['imag']


print(np.mean(np.real(HH)))
print(np.std(np.real(HH)))
print(np.mean(np.imag(HH)))
print(np.std(np.imag(HH)))

print(np.mean(np.real(HV)))
print(np.std(np.real(HV)))
print(np.mean(np.imag(HV)))
print(np.std(np.imag(HV)))

# plt.figure()
# n_HH_real, x, _ = plt.hist(np.real(HH).flatten(), bins=500)
# bin_centers_HH_real = 0.5*(x[1:]+x[:-1])
# n_HH_imag, x, _ = plt.hist(np.imag(HH).flatten(), bins=500)
# bin_centers_HH_imag = 0.5*(x[1:]+x[:-1])
# n_HV_real, x, _ = plt.hist(np.real(HV).flatten(), bins=500)
# bin_centers_HV_real = 0.5*(x[1:]+x[:-1])
# n_HV_imag, x, _ = plt.hist(np.imag(HV).flatten(), bins=500)
# bin_centers_HV_imag = 0.5*(x[1:]+x[:-1])
# plt.figure()
# plt.plot(bin_centers_HH_real, n_HH_real, label='HH I', color='red', alpha=0.5)
# plt.plot(bin_centers_HH_imag, n_HH_imag, label='HH Q', color='Blue', alpha=0.5)
# plt.plot(bin_centers_HV_real, n_HV_real, label='HV I', color='orange', alpha=0.5)
# plt.plot(bin_centers_HV_imag, n_HV_imag, label='HV Q', color='green', alpha=0.5)
# plt.legend()
# plt.show()

plt.figure()
plt.imshow(abs(HH), cmap="gray")
plt.show()

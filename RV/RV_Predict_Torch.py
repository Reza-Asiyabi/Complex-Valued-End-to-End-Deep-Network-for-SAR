"""
Based on "R. M.Asiyabi, M. Datcu, A. Anghel, H. Nies, "Complex-Valued End-to-end Deep
 Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and
  Classification" IEEE Transactions on Geoscience and Remote Sensing (2023)"
"""

import numpy as np
import torch
# import torch.nn.functional as F
# from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from RV.RV_Network_Torch import end_to_end_Net
from Src import Coherence

# import torchvision
# import torchvision.transforms as transforms

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

###################################################################################################################
PATH = 'Trained Models/Epoches/RV_mytrainednet_10E_10BS_lr0.001_mydataset_Features_30000Sample_Normal_7Class_epoch10.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = end_to_end_Net(n_in_channels=8, n_out_channels=2, n_classes=7, bilinear=True).cuda()

net.load_state_dict(torch.load(PATH))
net.eval()

dataset = abs(np.load("Path to the data"))

labels = np.load("patch to the labels")
######### 3-class label classification
# for i in range(len(labels_temp)):
#     if labels_temp[i] == 1 or labels_temp[i] == 2:
#         labels[i] = 0
#     elif labels_temp[i] == 3 or labels_temp[i] == 4 or labels_temp[i] == 5 or labels_temp[i] == 6:
#         labels[i] = 1
#     elif labels_temp[i] == 7:
#         labels[i] = 2

testset = torch.tensor(dataset)
testset = np.expand_dims(testset, axis=1)

outp_rect = []
outp_classified = []
inp = []

with torch.no_grad():
    for i in testset:
        i = torch.tensor(i)
        i = i.to(device=device, dtype=torch.float)
        temp_rect, temp_classified = net(i)
        outp_rect.append(temp_rect.cpu().detach().numpy())
        outp_classified.append(temp_classified.cpu().detach().numpy())
        inp.append(i.cpu().detach().numpy())


###############################presenting the results
channel = 1
img_inx = 19

Coh = Coherence.correlation(inputdata1=outp_rect[img_inx][0][channel], inputdata2=inp[img_inx][0][channel], k=2, mode="valid")
subtract = inp[img_inx][0][channel] - outp_rect[img_inx][0][channel]

plt.figure()
plt.subplot(4,2,1); plt.imshow(inp[img_inx][0][channel], cmap='gray'); plt.ylabel("Input Image"); plt.colorbar()
plt.subplot(4,2,2); temp = inp[img_inx][0][channel].flatten(); plt.hist(temp, bins=1000, log=False, range=(0, 255))
plt.subplot(4,2,3); plt.imshow(outp_rect[img_inx][0][channel], cmap='gray'); plt.ylabel("Output Image"); plt.colorbar()
plt.subplot(4,2,4); temp = outp_rect[img_inx][0][channel].flatten(); plt.hist(temp, bins=1000, log=False, range=(0, 255))
plt.subplot(4,2,5); plt.imshow(Coh, cmap='gray', vmin=0, vmax=1); plt.ylabel("Coherence Image"); plt.colorbar()
plt.subplot(4,2,6); temp = Coh.flatten(); plt.hist(temp, bins=1000, log=False)
plt.subplot(4,2,7); plt.imshow(subtract, cmap='gray'); plt.ylabel("Subtraction Image - abs"); plt.colorbar()
plt.subplot(4,2,8); temp = subtract.flatten(); plt.hist(temp, bins=1000, log=False)

plt.suptitle('Absolute Value - Image channel is:' + str(channel))
plt.show()


# true_labels = np.argmax(abs(np.array(labels)))
true_labels = np.array(labels)
predicted_labels = np.argmax(np.array(outp_classified), axis=2)

OA = accuracy_score(y_true=true_labels, y_pred=predicted_labels, normalize=True)
Confusion_matrix = confusion_matrix(y_true=true_labels, y_pred=predicted_labels, normalize='true')
print("Classification Confusion matrix:\n", Confusion_matrix)
print("Classification Overall Accuracy:", OA)

# #######################################Save image
# fig = plt.figure(figsize=(1.25, 1.25))
# ax = fig.add_axes([0, 0, 1, 1])
# ax.axis("off")
#
# ax.imshow(Coh, cmap='gray', vmin=0.9, vmax=1)
# fig.savefig('Path to save the image.png' %img_inx, dpi=80)

# ##########################################Save Confusion matrix
# import pandas as pd
#
# df = pd.DataFrame(Confusion_matrix)
# df.to_csv("Path to save the confusion matrix.csv")


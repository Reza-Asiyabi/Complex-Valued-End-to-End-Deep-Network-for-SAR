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

from ComplexValuedAutoencoderMain_Torch import end_to_end_Net
from Src import Coherence

# import torchvision
# import torchvision.transforms as transforms

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

###################################################################################################################
PATH = 'path to save the trained model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = end_to_end_Net(n_in_channels=8, n_out_channels=2, n_classes=7, bilinear=True).cuda()

net.load_state_dict(torch.load(PATH))
net.eval()

dataset = np.load("Path to the data")

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
# testset = dataset
testset = np.expand_dims(testset, axis=1)

outp_rect = []
outp_classified = []
inp = []

with torch.no_grad():
    for i in testset:
        i = torch.tensor(i)
        i = i.to(device=device, dtype=torch.complex64)
        temp_rect, temp_classified = net(i)
        # outp_rect.append(temp_rect.cpu().detach().numpy())
        outp_classified.append(temp_classified.cpu().detach().numpy())
        # inp.append(i.cpu().detach().numpy())
        try:
            outp_rect = torch.cat((outp_rect, temp_rect))
            # outp_classified = torch.cat((outp_classified, temp_classified))
            inp = torch.cat((inp, i))
        except:
            outp_rect = temp_rect
            # outp_classified = temp_classified
            inp = i

# inp = np.array(inp)
# outp_rect = np.array(outp_rect)
# outp_classified = np.array(outp_classified)

###############################Coherence
channel = 0
Coh = []
Coh_mean = []
for img_inx in range(len(inp)):
    Coh_temp = Coherence.complex_correlation(inputdata1=outp_rect[img_inx][1], inputdata2=inp[img_inx][1], k=2)
    Coh_mean_temp = torch.mean(Coh_temp)

    Coh.append(Coh_temp)
    Coh_mean.append(Coh_mean_temp)
Coh_mean_mean = torch.mean(torch.tensor(Coh_mean))
print("Coh mean:", torch.abs(Coh_mean_mean))



true_labels = np.array(labels)
predicted_labels = np.argmax(abs(np.array(outp_classified)), axis=2)

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

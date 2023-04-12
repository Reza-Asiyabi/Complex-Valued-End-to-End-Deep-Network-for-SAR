"""
Based on "R. M.Asiyabi, M. Datcu, A. Anghel, H. Nies, "Complex-Valued End-to-end Deep
 Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and
  Classification" IEEE Transactions on Geoscience and Remote Sensing (2023)"
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import logging



class BasicDataset(Dataset):
    def __init__(self, imgs, labels=None, Bottleneck_features=None, scale=1, normal=False):
        self.imgs = imgs
        self.labels = labels
        self.Bottleneck_features = Bottleneck_features
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.normal = normal

        self.ids = range(np.size(imgs, 0))
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        if self.labels is not None: label = np.array(self.labels[idx])
        img = np.array(self.imgs[idx])
        if self.Bottleneck_features is not None: Bottleneck_feature = np.array(self.Bottleneck_features[idx])


        if self.normal == True:
            img = self.preprocess(img, self.scale)
            if self.labels is not None: label = self.preprocess(label, self.scale)
            if self.Bottleneck_features is not None: Bottleneck_feature = self.preprocess(Bottleneck_feature, self.scale)

        if self.Bottleneck_features is not None and self.labels is not None:
            return {
                'image': torch.from_numpy(img).type(torch.complex64),
                'label': torch.from_numpy(label).type(torch.long),
                'Bottleneck_feature': torch.from_numpy(Bottleneck_feature).type(torch.complex64)
            }
        elif self.Bottleneck_features is not None:
            return {
                'image': torch.from_numpy(img).type(torch.complex64),
                'Bottleneck_feature': torch.from_numpy(Bottleneck_feature).type(torch.complex64)
            }
        elif self.labels is not None:
            return {
                'image': torch.from_numpy(img).type(torch.complex64),
                'label': torch.from_numpy(label).type(torch.long)
            }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs, labels=None, Bottleneck_features=None, scale=1, normal=False):
        super().__init__(imgs, labels, Bottleneck_features, scale, normal)

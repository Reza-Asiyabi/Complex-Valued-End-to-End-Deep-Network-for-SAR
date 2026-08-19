"""
Based on "R. M.Asiyabi, M. Datcu, A. Anghel, H. Nies, "Complex-Valued End-to-end Deep
 Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and
  Classification" IEEE Transactions on Geoscience and Remote Sensing (2023)"
"""

import numpy as np
import torch
from bokeh.colors.named import azure
from torch.utils.data import Dataset
import logging
from pathlib import Path
import pandas as pd

class custom_dataset(Dataset):
    def __init__(self, root, dataset_path, data_channels, range_downsample_factor, azimuth_downsample_factor, logger=None, data_normalization=None, data_stats_dir=None,
                 clipping_threshold=None, transform= None):
        self.dataset = np.load(Path(root) / dataset_path)
        self.data_channels = data_channels
        self.range_downsample_factor = range_downsample_factor
        self.azimuth_downsample_factor = azimuth_downsample_factor
        self.logger = logger
        self.data_stats = pd.read_csv(Path(root) / data_stats_dir)
        self.data_normalization = data_normalization
        self.clipping_threshold = clipping_threshold
        self.transform = transform
        self.num_bands = self.dataset.shape[1]

        self.ids = range(self.dataset.shape[0])
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def _down_sample(self, img, range_factor=1, azimuth_factor=1):
        h, w = img.shape[:2]  # Get image dimensions (256x256 expected)
        new_h, new_w = h // azimuth_factor, w // range_factor  # Compute new size
        downsampled = img.reshape(new_h, azimuth_factor, new_w, range_factor, -1).mean(axis=(1, 3))  # Reshape and take mean to downsample
        return np.squeeze(downsampled)

    def _Normalize(self, img):
        processed_bands = []
        for b in range(self.num_bands):
            # Separate real & imaginary parts
            real_part = np.clip(np.real(img[b]), self.clipping_threshold[0], self.clipping_threshold[1]).astype(np.float32)
            imag_part = np.clip(np.imag(img[b]), self.clipping_threshold[0], self.clipping_threshold[1]).astype(np.float32)

            # Apply normalization
            if self.data_normalization == 'Mean_Std':
                real_part = (real_part - self.data_stats['mean']) / np.sqrt(self.data_stats['std'])
                imag_part = (imag_part - self.data_stats['mean']) / np.sqrt(self.data_stats['std'])

            elif self.data_normalization == 'CV_MinMax':
                real_part /= self.data_stats['abs_max']
                imag_part /= self.data_stats['abs_max']

            elif self.data_normalization == 'MinMax':
                real_part = (real_part - self.data_stats['real_min']) / (self.data_stats['real_max'] - self.data_stats['real_min'] + 1e-6)
                imag_part = (imag_part - self.data_stats['imag_min']) / (self.data_stats['imag_max'] - self.data_stats['imag_min'] + 1e-6)

            # Convert back to complex format
            processed_bands.append(real_part + 1j * imag_part)
        return np.stack(processed_bands, axis=0)  # Shape: (B, H, W)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_orig = np.array(self.dataset[idx][self.data_channels])
        img_ds = self._down_sample(img=img_orig, range_factor=self.range_downsample_factor, azimuth_factor=self.azimuth_downsample_factor)

        if self.data_normalization:
            img_ds = self._Normalize(img_ds)

        return {
            'img_orig': torch.from_numpy(img_orig).type(torch.complex64),
            'img_ds': torch.from_numpy(img_ds).type(torch.complex64)
        }

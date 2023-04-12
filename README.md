# Complex-Valued-End-to-end-Deep-Network-for-SAR
This repository provides the implementation of "Complex-Valued End-to-end Deep Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and Classification" (IEEE TGRS 2023).

![image](https://user-images.githubusercontent.com/49744743/231399143-7800ada8-4d80-41ff-9237-b58733716daa.png)


## Abstract
Deep learning models have achieved remarkable success in many different fields and attracted many interests. Several researchers attempted to apply deep learning models to Synthetic Aperture Radar (SAR) data processing, but it did not have the same breakthrough as the other fields, including optical remote sensing. SAR data are in complex domain by nature and processing them with Real-Valued (RV) networks neglects the phase component which conveys important and distinctive information. A Complex-Valued (CV) end-to-end deep network is developed in this study for the reconstruction and classification of CV-SAR data. Azimuth subaperture decomposition is utilized to incorporate physics-aware attributes of the CV-SAR into the deep model. Moreover, the correlation coefficient amplitude (Coherence) of the CV-SAR images depends on the SAR system characteristics and physical properties of the target. This coherency should be considered and preserved in the processing chain of the CV-SAR data. The coherency preservation of the CV deep networks for CV-SAR images, which is mostly neglected in the literature, is evaluated in this study. Furthermore, a large-scale CV-SAR annotated dataset for the evaluation of the CV deep networks is lacking. A semantically annotated CV-SAR dataset from Sentinel-1 Single Look Complex StripMap mode data (S1SLC_CVDL dataset) is developed and introduced in this study. The experimental analysis demonstrated the better performance of the developed CV deep network for CV-SAR data classification and reconstruction in comparison to the equivalent RV model and more complicated RV architectures, as well as its coherency preservation and physics-aware capability.

## Dataset
The "S1SLC_CVDL: A COMPLEX-VALUED ANNOTATED SINGLE LOOK COMPLEX SENTINEL-1 SAR DATASET FOR COMPLEX-VALUED DEEP NETWORKS" dataset developed and used for training and evaluation in this paper. The S1SLC_CVDL dataset is available at the IEEE DataPort (http://ieee-dataport.org/11016)

![image](https://user-images.githubusercontent.com/49744743/231400139-2ca9c022-89ad-4b13-b4c1-1078151dca13.png)

30,000 patches are randomely selected from the S1SLC_CVDL dataset for the experiments in this study. 66% of the selected patches are used for training the network in each case study and the rest are used for evaluation.

## Usage

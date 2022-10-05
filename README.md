# DFENet: A dual-branch feature enhanced network integrating transformers and convolutional feature leaning for multimodal medical image fusion


[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/wdhudiekou/UMF-CMGR/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.6.0-%237732a8)](https://pytorch.org/)


## Updates
[2022-10-05] The whole code repository had been uploaded.


## Requirements
- CUDA 11.3
- Python 3.7 (or later)
- Pytorch 1.10.1
- Torchvision 0.11.2
- OpenCV 4.5.5


This code has been tested with `Pytorch` and NVIDIA RTX A6000 GPU.


## Data preparation
1.You can obtain the training MS-COCO 2014 Datasets from the hyperlink[[link](https://cocodataset.org/#download)].

2.You can obtain the test medical image Datasets from the hyeprlink[[link](www.med.harvard.edu/AANIIB/home.html)], and we have also prepared some test cases in the file directory of Data.


## Inference
If you want to test the pretrained model using your own well-categorized dataset, please define your own data generator for `test.py` and perform the following script.
```
python test.py --model_path ./ckpt_model.pth --mri_file ./your_mri_dataset_root --ct_file ./your_ct_dataset_root
python test.py --model_path ./ckpt_model.pth --mri_file ./your_mri_dataset_root --pet_file ./your_pet_dataset_root
python test.py --model_path ./ckpt_model.pth --mri_file ./your_mri_dataset_root --spect_file ./your_spect_dataset_root
```


## Training on the large-scale scene images dataset for image-reconstructing task
If you want to train the encoder-decoder based model on the large-scale scene datasets, please define your own data generator for `model_train.py` and perform the following script.
```
python model_train.py --train_path ./your_dataset_path
```


## Acknowledgement
Some codes in this repository are modified from [VIT](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer) and [RFN-NEST](https://github.com/hli1221/imagefusion-rfn-nest).


#### Keywords
* Keywords: multimodal medical image fusion, convolutional neural network, vision transformer, feature fuser, local energy and gradient






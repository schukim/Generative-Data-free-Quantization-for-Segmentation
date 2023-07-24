# Generative Data-Free Quantization for Segmentation

<br/>

We provide PyTorch implementation for "Generative Data-Free Quantization for Segmentation".

This project was conducted by modifying the previous research ["Generative Low-bitwidth Data Free Quantization (GDFQ)"](https://arxiv.org/abs/2003.03603).

We applied GDFQ to U-Net, the representative image segmentation model.

<br/>

## Paper
* Generative Data-free Quantization for Segmentation
* Sang-woo Kim, Jin-ho Lee 
* Seoul National University Bachelor Thesis

<br/>

## Dependencies

* Python 3.6
* PyTorch 1.2.0
* dependencies in requirements.txt

<br/>

## Getting Started

### Installation

1. Clone this repository.

2. Install pytorch and other dependencies.

        pip install -r requirements.txt

### Set the paths of datasets for testing

1. Set the "dir_img" and "dir_mask" in "carvana_unet.hocon" as the path root of the image and mask data of your Carvana dataset. For example:

        dir_img = "/home/datasets/carvana/imgs"
        dir_mask = "/home/datasets/carvana/masks"


2. Set the "dir_img" and "dir_mask" in "nih_unet.hocon" as the path root of the image and mask data of your NIH dataset. For example:

        dir_img = "/home/datasets/nih_pancras/img"
        dir_mask = "/home/datasets/nih_pancras/mask"

### Load your own pre-trained model

This repository contains our own pre-trained U-Nets on Carvana and NIH datasets.

You can use your own pre-trained model by modifying .hocon files.

1. Set the "model_path" in "carvana_unet.hocon" as the path root of your own pre-trained model on Carvana datset. For example:

        model_path = "carvana_pre.pth"

2. Set the "model_path" in "nih_unet.hocon" as the path root of your own pre-trained model on NIH datset. For example:
        
        model_path = "nih_pre.pth"

### Training

To quantize the pretrained U-Net on Carvana to 4-bit:

    python main.py --conf_path=./carvana_unet.hocon --id=01
To quantize the pretrained U-Net on NIH to 4-bit:

    python main.py --conf_path=./nih_unet.hocon --id=01

<br/>

## Results

|  Dataset | Model | Pretrain Top1 Acc(%) | W4A4(Ours) Top1 Acc(%) |
   | :-: | :-: | :-: | :-: |
  | Carvana | U-Net | 98.89 | 95.63 |
  | NIH | U-Net | 82.67 | 81.53 |

<br/>

## Acknowledgments

All the source codes in this project is the fusion of two github projects below:
* [Generative-Low-bitwidth-Data-Free-Quantization](https://github.com/xushoukai/GDFQ)
* [U-Net: Semantic Segmentation with PyTorch](https://github.com/milesial/Pytorch-UNet)

Also, this README script is written by modifying that of [GDFQ](https://github.com/xushoukai/GDFQ).

Special thanks to the researchers of the two projects above, which this project is based on.

<br/>

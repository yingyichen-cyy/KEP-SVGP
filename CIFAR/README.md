# CIFAR

## Environment
Our model can be learnt on a **single NVIDIA GeForce RTX 2070 SUPER GPU** 
```
conda create -n uncertainty python=3.8
conda activate uncertainty
pip install -r requirements.txt
```
The code was tested on Python 3.8 and PyTorch 1.10.1.

## Dataset
* We are using **CIFAR10, CIFAR100**. 
* We keep **10%** of training samples as a validation dataset for failure prediction.
* Download datasets to `./data/` and split into `train/`, `val/` and `test/`. Take `CIFAR10` for an example:
```
cd data
bash download_cifar.sh
```
The structure of the file should be:
```
./data/CIFAR10/
  ├── train
  ├── val
  └── test
```
* Using **CIFAR10-C** to test robustness under data corrputions.
* To download CIFAR10-C dataset [[Hendrycks et al., 2019]](https://arxiv.org/pdf/1903.12261.pdf), please refer to [here](https://github.com/hendrycks/robustness?tab=readme-ov-file). The structure of the file should be:
```
./data/CIFAR-10-C/
  ├── brightness.npy
  ├── contrast.npy
  ├── defocus_blur.npy
...
```

## Run Tasks
Please train our model according to `run/vit_cifar10.sh` and `run/vit_cifar100.sh`.
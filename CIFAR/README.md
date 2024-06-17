# sgp_primal

## Environment
Our model can be learnt in a **single GPU RTX3090-24G** 

```
conda create -n uncertainty python=3.8
conda activate uncertainty
pip install -r requirements.txt
```

The code was tested on Python 3.8 and PyTorch 1.10.1.

* We are using **CIFAR10, CIFAR100**. 
* Download CIFAR10 and CIFAR100 to ./data/ and split into train/val/test.
```
cd data
bash download_cifar.sh
```

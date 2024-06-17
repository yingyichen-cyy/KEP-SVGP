# IMDB

## Environment
Our model can be learnt on a **single NVIDIA GeForce RTX 2070 SUPER GPU** 
```
conda create -n imdb python=3.7
conda activate imdb
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install torchtext==0.9.0
pip install scikit-learn==1.0.2
pip install tensorboard
pip install transformers
pip install warmup_scheduler
```
## Run Tasks
Please train our model according to `run_imdb.sh`. Note that we devide the whole dataset into 70% training, 10% validation and 20% test sets according to [`main.py: 59-63`](https://github.com/yingyichen-cyy/KEP-SVGP/blob/f32422db893128b5228e48a83942fccd447bdad0/IMDB/main.py#L59-L63).

## Acknowledgement
This code is based on [pytorch-sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/tree/master), [text](https://github.com/pytorch/text).

# sgp_primal on IMDB

This repository is based on [pytorch-sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/tree/master), [text](https://github.com/pytorch/text)

## Environment
Our model can be learnt in a **single P100 GPU** 

```
conda create -n imdb python=3.7
conda activate imdb
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install torchtext==0.9.0
pip install scikit-learn==1.0.2 
```


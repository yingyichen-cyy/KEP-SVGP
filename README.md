# KEP-SVGP
(ICML 2024) PyTorch implementation of **KEP-SVGP** attention mechanism available on [OpenReview](https://openreview.net/forum?id=4RqG4K5UwL).

**Self-Attention through Kernel-Eigen Pair Sparse Variational Gaussian Processes**

by *[Yingyi Chen*](https://yingyichen-cyy.github.io/),
[Qinghua Tao*](https://qinghua-tao.github.io/), 
[Francesco Tonin](https://taralloc.github.io/), 
[Johan A.K. Suykens](https://www.esat.kuleuven.be/sista/members/suykens.html)*

[[arXiv](https://arxiv.org/abs/2402.01476)]
[[PDF](https://openreview.net/forum?id=4RqG4K5UwL)]
<!-- [[Video](https://nips.cc/virtual/2023/poster/71144)] -->
<!-- [[Poster](https://yingyichen-cyy.github.io/Primal-Attention/resrc/poster.pdf)] -->
<!-- [[Project Page](https://yingyichen-cyy.github.io/Primal-Attention/)] -->

<p align="center">
<img src="./img/workflow.jpeg" height = "320" alt="" align=center />
<br><br>
<b>Figure 1.</b> An illustration of canonical self-attention and our KEP-SVGP attention in one layer.
</p>

If our project is helpful for your research, please consider citing:
``` 
@inproceedings{chen2024self,
  title={Self-Attention through Kernel-Eigen Pair Sparse Variational Gaussian Processes},
  author={Chen, Yingyi and Tao, Qinghua and Tonin, Francesco and Suykens, Johan A.K.},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## Table of Content
Please refer to different folders for detailed experiment instructions. 
Note that we specified **different environments for different tasks**.

Please feel free to contact chenyingyi076@gmail.com for any discussion.

* [1. CIFAR](https://github.com/yingyichen-cyy/KEP-SVGP/tree/master/CIFAR)
* [2. CoLA](https://github.com/yingyichen-cyy/KEP-SVGP/tree/master/CoLA)
* [3. IMDB](https://github.com/yingyichen-cyy/KEP-SVGP/tree/master/IMDB)

## Acknowledgement
This repository is based on the official codes of 
**CIFAR**:
[SGPA](https://github.com/chenw20/SGPA/), 
[OpenMix](https://github.com/Impression2805/OpenMix),
[ViT-CIFAR](https://github.com/omihub777/ViT-CIFAR/tree/main),
**CoLA**:
[SGPA](https://github.com/chenw20/SGPA/),
[huggingface](https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/bert/modeling_bert.py),
**IMDB**:
[pytorch-sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/tree/master), 
[text](https://github.com/pytorch/text).

# CoLA

## Environment
Since we use [``allennlp``](https://github.com/allenai/allennlp) package, we need to install
```
conda create -n allennlp python=3.7
conda activate allennlp
conda install -c conda-forge jsonnet
pip install allennlp==2.9.3
# and we need to downgrade to avoid 
# "AttributeError: module 'cached_path' has no attribute 'file_friendly_logging'"
pip install cached-path==1.1.2
```

## Dataset
Please download the dataset via
```
wget https://nyu-mll.github.io/CoLA/cola_public_1.1.zip
```
and use `in_domain_train.tsv`, `in_domain_dev.tsv`, `out_of_domain_dev.tsv` from the `raw/` folder. The structure of the file should be:
```
./data/
  ├── in_domain_train.tsv
  ├── in_domain_dev.tsv
  └── out_of_domain_dev.tsv
```

## Run Tasks
Please train our model according to `run_cola.sh`.

## Acknowledgement
This code is based on the official codes of [huggingface](https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/bert/modeling_bert.py), [SGPA](https://github.com/chenw20/SGPA/).
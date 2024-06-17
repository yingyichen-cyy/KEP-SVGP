################ Baseline ViT ################
python3 main.py \
--attn-type softmax \
--batch-size 128 \
--gpu 0 \
--nb-epochs 300 \
--nb-run 1 \
--model vit_cifar \
--lr 1e-3 \
--weight-decay 5e-5 \
--save-dir ./CIFAR10_out/vit_out \
Cifar10

python3 test.py \
--attn-type softmax \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./CIFAR10_out/vit_out \
Cifar10

################ KEP-SVGP-Attention ################ 
########## [e(x),r(x)] ##########
# one-layer of KSVD
python3 main.py \
--attn-type kep_svgp \
--concate \
--ksvd-layers 1 \
--eta-ksvd 10 \
--batch-size 128 \
--gpu 0 \
--nb-epochs 300 \
--nb-run 1 \
--model vit_cifar \
--lr 1e-3 \
--weight-decay 5e-5 \
--save-dir ./CIFAR10_out/vit_out_cat \
Cifar10

python3 test.py \
--attn-type kep_svgp \
--concate \
--ksvd-layers 1 \
--eta-ksvd 10 \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./CIFAR10_out/vit_out_cat \
Cifar10

########## e(x)+r(x) ##########
# one-layer of KSVD
python3 main.py \
--attn-type kep_svgp \
--ksvd-layers 1 \
--eta-ksvd 1 \
--batch-size 128 \
--gpu 0 \
--nb-epochs 300 \
--nb-run 1 \
--model vit_cifar \
--lr 1e-3 \
--weight-decay 5e-5 \
--save-dir ./CIFAR10_out/vit_out_sum \
Cifar10

python3 test.py \
--attn-type kep_svgp \
--ksvd-layers 1 \
--eta-ksvd 1 \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./CIFAR10_out/vit_out_sum \
Cifar10
################ Baseline ViT ################
python3 main.py \
--attn-type softmax \
--batch-size 32 \
--gpu 0 \
--nb-epochs 50 \
--nb-run 1 \
--model vit_cola \
--lr 5e-4 \
--save-dir ./cola_out/vit_out

python3 test.py \
--attn-type softmax \
--batch-size 32 \
--gpu 0 \
--nb-run 1 \
--model vit_cola \
--save-dir ./cola_out/vit_out

################ KEP-SVGP-Attention ################ 
########## e(x)+r(x) ##########
python3 main.py \
--attn-type kep_svgp \
--ksvd-layers 1 \
--eta-ksvd 1 \
--batch-size 32 \
--gpu 0 \
--nb-epochs 50 \
--nb-run 1 \
--model vit_cola \
--lr 5e-4 \
--weight-decay 5e-5 \
--save-dir ./cola_out/vit_out_sum

python3 test.py \
--attn-type kep_svgp \
--ksvd-layers 1 \
--eta-ksvd 1 \
--batch-size 32 \
--gpu 0 \
--nb-run 1 \
--model vit_cola \
--save-dir ./cola_out/vit_out_sum
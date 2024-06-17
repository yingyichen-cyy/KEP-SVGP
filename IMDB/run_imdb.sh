################ Baseline ViT ################
python3 main.py \
--attn-type softmax \
--batch-size 32 \
--gpu 0 \
--nb-epochs 20 \
--nb-run 1 \
--model transformer_imdb \
--lr 1e-3 \
--save-dir ./imdb_out/vit_out

################ KEP-SVGP ################
python3 main.py \
--attn-type kep_svgp \
--ksvd-layers 1 \
--eta-ksvd 10 \
--batch-size 32 \
--gpu 0 \
--nb-epochs 20 \
--nb-run 1 \
--model transformer_imdb \
--lr 1e-3 \
--weight-decay 5e-5 \
--save-dir ./imdb_out/vit_out_sum
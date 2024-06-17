import argparse

def get_args_parser():
    
    parser = argparse.ArgumentParser(description='Kernel-Eigen Pair Sparse Variational Gaussian Processes',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--nb-epochs', default=20, type=int, help='Total number of training epochs ')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--num-classes',type=int,default=2)
    parser.add_argument('--max-len',type=int,default=512)
    parser.add_argument('--dataset', default='imdb', type=str, choices = ['imdb'], help='dataset')
    parser.add_argument('--seed', type=int,default=0)

    # KEP-SVGP-attention
    parser.add_argument('--ksvd-layers', type=int, default=1, help='Number of ksvd layers applied to the transformer')
    parser.add_argument('--attn-type', default='kep_svgp', type=str, choices = ['kep_svgp', 'softmax'], help='Type of attention') 
    parser.add_argument('--eta-ksvd', type=float, default=0.1, help='coefficient of the KSVD regularization')
    parser.add_argument('--eta-kl', type=float, default=1.0, help='coefficient of the KL divergence regularization')
    parser.add_argument('--low_rank', type=int, default=10, help='Number of dimension the low rank method projected to')
    parser.add_argument('--rank_multi', type=int, default=10, help='low rank dimension * rank_multi')

    ## optimizer 
    parser.add_argument('--lr', default=1e-3, type=float, help='Max learning rate for cosine learning rate scheduler')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='Weight decay')
    parser.add_argument("--min-lr", default=1e-4, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--warmup-epoch", default=5, type=int)

    ## nb of run + print freq
    parser.add_argument('--nb-run', default=1, type=int, help='Run n times, in order to compute std')

    ## dataset setting
    parser.add_argument('--nb-worker', default=4, type=int, help='Nb of workers')
    
    ## Model
    parser.add_argument('--model', default='transformer_imdb', type=str, choices = ['transformer_imdb'], help='Models name to use')
    parser.add_argument('--emb_dim',type=int,default=128)
    parser.add_argument('--depth',type=int,default=1)
    parser.add_argument('--hdim',type=int,default=128)
    parser.add_argument('--num_heads',type=int,default=8)
    
    parser.add_argument('--save-dir', default='./output', type=str, help='Output directory')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')

    return parser.parse_args()

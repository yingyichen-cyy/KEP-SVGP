import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='Test results with different post-hoc methods',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
    parser.add_argument('--nb-run', default=1, type=int, help='Run n times, in order to compute std')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
    parser.add_argument('--save-dir', default='./output', type=str, help='Output directory')
    
    parser.add_argument('--num-classes',type=int,default=2)
    parser.add_argument('--max-len',type=int,default=100)
    parser.add_argument('--dataset', default='cola', type=str, choices = ['cola'], help='dataset')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--min_word_count',type=int,default=0)
    
    # SGP-primal-attention
    parser.add_argument('--ksvd-layers', type=int, default=1, help='Number of ksvd layers applied to the transformer')
    parser.add_argument('--attn-type', default='kep_svgp', type=str, choices = ['kep_svgp', 'softmax'], help='Type of attention')
    parser.add_argument('--eta-ksvd', type=float, default=0.1, help='coefficient of the KSVD regularization')
    parser.add_argument('--eta-kl', type=float, default=1.0, help='coefficient of the KL divergence regularization')
    parser.add_argument('--low-rank', type=int, default=5, help='Number of dimension the low rank method projected to')
    parser.add_argument('--rank-multi', type=int, default=2, help='low rank dimension * rank_multi')

    ## Model + optim method + data aug + loss + post-hoc
    parser.add_argument('--model', default='vit_cola', type=str, choices = ['vit_cola'], help='Models name to use')
    parser.add_argument('--emb_dim',type=int,default=128)
    parser.add_argument('--depth',type=int,default=2)
    parser.add_argument('--hdim',type=int,default=256)
    parser.add_argument('--num_heads',type=int,default=4)

    return parser.parse_args()

import torch
from torch import nn, Tensor
import math
from model.layers import TransformerEncoder
import argparse

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout= 0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, args, vocab_size, attn_type, ksvd_layers=1, low_rank=5, rank_multi=2, num_classes=2, \
                dropout=0., num_layers=7, hidden=384, mlp_hidden=384, head=8):
        super().__init__()
        self.attn_type = attn_type
        self.num_layers = num_layers
        self.ksvd_layers = ksvd_layers

        self.vocab_size = vocab_size
        self.max_len = args.max_len
        self.emb_dim = args.emb_dim
        self.hidden = hidden
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_encoder = PositionalEncoding(self.emb_dim, self.dropout, self.max_len)
        
        enc_list = [TransformerEncoder(args=args, attn_type="softmax", low_rank=low_rank, rank_multi=rank_multi, \
                    feats=hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        if self.attn_type == "kep_svgp":
            for i in range(self.ksvd_layers):
                enc_list[-(i+1)] = TransformerEncoder(args=args, attn_type="kep_svgp", low_rank=low_rank, rank_multi=rank_multi, embed_len=self.max_len, \
                    feats=hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head)
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )

    def forward(self, text):
        score_list = []
        Lambda_inv_list = []
        kl_list = []

        out = self.embedding(text) 
        out = self.pos_encoder(out)

        for enc in self.enc:
            if enc.attn_type == "softmax":
                out = enc(out)
            elif enc.attn_type == "kep_svgp":
                out, scores, Lambda_inv, kl = enc(out)
                score_list.append(scores)
                Lambda_inv_list.append(Lambda_inv)
                kl_list.append(kl)

        out = out.mean(1)
        out = self.fc(out)

        if self.attn_type == "softmax":
            return out
        elif self.attn_type == "kep_svgp":
            return out, score_list, Lambda_inv_list, kl_list

def transformer_imdb(args, vocab_size, attn_type, ksvd_layers, low_rank, rank_multi):
    return Transformer(args=args, vocab_size=vocab_size, attn_type=attn_type, ksvd_layers=ksvd_layers, num_classes=args.num_classes, low_rank=low_rank, rank_multi=rank_multi, \
                dropout=0.1, num_layers=args.depth, hidden=args.hdim, head=args.num_heads, mlp_hidden=args.hdim) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--emb_dim',type=int,default=128)
    parser.add_argument('--max-len',type=int,default=512)
    parser.add_argument('--concate', action='store_true', help='whether to use [e(x),r(x)] instead of (e(x)+r(x))')  
    parser.add_argument('--eta-ksvd', type=float, default=0.1, help='coefficient of the KSVD regularization')
    parser.add_argument('--eta-kl', type=float, default=1.0, help='coefficient of the KL divergence regularization')
    args = parser.parse_args()

    b,c,d = 32, 512, 128
    x = torch.randn(b, c, d).cuda()
    # x = torch.tensor(x, dtype=torch.long).cuda()
    net = Transformer(args=args, vocab_size=3000, attn_type="kep_svgp", ksvd_layers=1, num_classes=2, low_rank=10, rank_multi=10, \
                dropout=0.1, num_layers=1, hidden=128, head=8, mlp_hidden=128)
    net.cuda()
    outs = net(x)    
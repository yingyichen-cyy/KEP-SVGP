import torch
import torch.nn as nn
import torch.nn.functional as F
from model.kep_svgp import KEP_SVGPAttention

class TransformerEncoder(nn.Module):
    def __init__(self, args, attn_type, feats, mlp_hidden=128, head=8, dropout=0., embed_len=64, \
                low_rank=10, rank_multi=10, attn_drop=0.):
        super(TransformerEncoder, self).__init__()
        self.attn_type = attn_type
        self.la1 = nn.LayerNorm(feats)
        if self.attn_type == "softmax":
            self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        elif self.attn_type == "kep_svgp":
            self.msa = KEP_SVGPAttention(feats, head, embed_len=embed_len, low_rank=low_rank, rank_multi=rank_multi, proj_drop=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.la1(x)
        if self.attn_type == "softmax":
            out = self.msa(out)
        elif self.attn_type == "kep_svgp":
            out, scores, Lambda_inv, kl = self.msa(out)

        out = out + x
        out = self.mlp(self.la2(out)) + out

        if self.attn_type == "softmax":
            return out
        elif self.attn_type == "kep_svgp":
            return out, scores, Lambda_inv, kl


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats, head, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o


if __name__=="__main__":
    b,n,f = 100, 64, 128
    x = torch.randn(b,n,f).cuda()
    # net = TransformerEncoder("softmax", f).cuda()
    net = TransformerEncoder("kep_svgp", f).cuda()
    out = net(x)[0]
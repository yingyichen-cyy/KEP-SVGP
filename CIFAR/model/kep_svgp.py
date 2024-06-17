import torch
import torch.nn as nn
import torch.nn.functional as F


class KEP_SVGPAttention(nn.Module):
    def __init__(self, dim, num_heads=8, embed_len=64, low_rank=10, rank_multi=10, concate=False, \
                qk_bias=False, attn_drop=0., proj_drop=0.):
        super(KEP_SVGPAttention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qk = nn.Linear(dim, dim * 2, bias=qk_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        ## projection weights we, wr in kep_svgp attention
        self.low_rank = low_rank
        self.rank_multi = rank_multi
        self.embed_len = embed_len
        self.we = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.num_heads, min(self.embed_len, self.low_rank * self.rank_multi), self.low_rank)))
        self.wr = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.num_heads, min(self.embed_len, self.low_rank * self.rank_multi), self.low_rank)))
        self.log_lambda_sqrt_inv_diag = nn.Parameter(nn.init.uniform_(torch.Tensor(self.num_heads, self.low_rank)))

        ## sparse GP
        self.m_u = nn.Parameter(nn.init.normal_(torch.Tensor(1, self.num_heads, self.low_rank, self.low_rank)))
        self.s_sqrt_low_triangle = nn.Parameter(nn.init.normal_(torch.Tensor(1, self.num_heads, self.low_rank, self.low_rank, self.low_rank)))
        self.log_ssqrt = nn.Parameter(nn.init.normal_(torch.Tensor(1, self.num_heads, self.low_rank, self.low_rank)))
        self.final_weight = nn.Linear(self.low_rank, self.head_dim)

        self.concate = concate
        if self.concate:
            self.embed_len_weight = nn.Linear(self.embed_len * 2, self.embed_len)

    def gen_weights(self, x):
        ## evenly sample
        if self.embed_len > self.low_rank * self.rank_multi:
            indices = torch.linspace(0, x.shape[1]-1, self.low_rank * self.rank_multi, dtype=int)
            x = x.transpose(-2,-1).reshape(x.size(0), self.num_heads, self.head_dim, x.size(1))
            x = x[:, :, :, indices].transpose(1, 2)
        else:
            x = x.transpose(-2,-1).reshape(x.size(0), self.num_heads, self.head_dim, x.size(1))
            x = x.transpose(1, 2)
        we = torch.einsum('bahd,hde->bahe', x, self.we.type_as(x)).transpose(1,2)
        wr = torch.einsum('bahd,hde->bahe', x, self.wr.type_as(x)).transpose(1,2)
        return we, wr 

    def feature_map(self, x):
        ## normalization should be on dim=-1
        return F.normalize(x, p=2, dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk.unbind(0)

        we, wr = self.gen_weights(x)
        q = self.feature_map(q) 
        k = self.feature_map(k) 
        escore = torch.einsum('...nd,...de->...ne', q, we)
        rscore = torch.einsum('...nd,...de->...ne', k, wr)
        if self.concate:
            score = torch.cat((escore, rscore), dim=2)

        ## compute mean and covariance for the SGP
        # mean
        lambda_sqrt_inv_diag = torch.diag_embed(torch.exp(self.log_lambda_sqrt_inv_diag))
        if self.concate:
            v1 = score @ (lambda_sqrt_inv_diag.unsqueeze(0) ** 2)
        else:
            v1 = (escore + rscore) @ (lambda_sqrt_inv_diag.unsqueeze(0) ** 2)
        mean = v1 @ self.m_u
        # covariance 
        s_sqrt = torch.exp(self.log_ssqrt) 
        s_sqrt_diag = torch.diag_embed(s_sqrt) 
        s_sqrt_local = s_sqrt_diag + torch.tril(self.s_sqrt_low_triangle, diagonal=-1) 

        # choleskey factor of the covariance matrix
        # the last dimension should be the [d] dimension
        v2 = v1.unsqueeze(2) @ s_sqrt_local.permute(0,1,4,2,3) 
        
        ## samples from the approximate posterior
        if self.concate:
            samples = mean + (v2.permute(0,1,3,2,4) @ torch.randn(B, self.num_heads, 2*N, mean.shape[3], 1).to(x.device)).squeeze()
        else:
            samples = mean + (v2.permute(0,1,3,2,4) @ torch.randn(B, self.num_heads, N, mean.shape[3], 1).to(x.device)).squeeze()
        attn_out = self.final_weight(samples)
        if self.concate:
            attn_out = self.embed_len_weight(attn_out.permute(0,1,3,2)).permute(0,1,3,2)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        # attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)

        ## compute the KL divergence 
        # Tr(\Lambda^{-2}S_{uu}) term 
        # where Tr(AA^\top) = ||A||_F^2
        v3 = (lambda_sqrt_inv_diag[None,None,...] ** 2) @ s_sqrt_local.permute(0,4,1,2,3)
        kl = 0.5 * torch.sum(v3.pow(2)) 
        # m_u^\top\Lambda^{-2}m_u term:
        mu_d = self.m_u.permute(0,1,3,2).unsqueeze(-1)
        kl += 0.5 * (mu_d.permute(0,1,2,4,3) @ (lambda_sqrt_inv_diag.unsqueeze(0).unsqueeze(2) ** 4) @ mu_d).sum()
        # log(|\Lambda^2|/|S_uu|) term:
        kl -= torch.sum(self.log_ssqrt)
        kl -= 0.5 * 4 * torch.sum(self.log_lambda_sqrt_inv_diag) * self.low_rank
        # s term, which is a constant
        kl -= 0.5 * self.low_rank * self.low_rank * self.num_heads

        return attn_out, [escore, rscore, self.we, self.wr], lambda_sqrt_inv_diag, kl


if __name__ == '__main__': 
    net = KEP_SVGPAttention(dim=128, num_heads=4, embed_len=64)
    net = net.cuda()
    inputs = torch.cuda.FloatTensor(100, 64, 128)
    with torch.no_grad():
        samples = net(inputs)
import torch
import torch.nn as nn
from model.layers import TransformerEncoder
from allennlp.modules.elmo import batch_to_ids, Elmo

class Embeddings(torch.nn.Module):
    def __init__(self, vocab_size, max_len, emb_size, h_size, drop_rate):
        super(Embeddings,self).__init__()
        self.token_embeds=nn.Embedding(vocab_size,emb_size,padding_idx=0)
        self.pos_embeds=nn.Embedding(max_len,emb_size+1024)
        self.layer_norm=nn.LayerNorm(h_size)
        self.project=nn.Linear(emb_size+1024,h_size)
        self.dropout = nn.Dropout(drop_rate)
        self.emb_size=emb_size
        self.h_size = h_size
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" 
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        num_rep=1
        self.elmo=Elmo(options_file,weight_file,num_rep,dropout=0.)

    def forward(self,input_data,pos,data=None):
        pos=self.pos_embeds(pos)
        character_ids=batch_to_ids(data).cuda()
        rep=self.elmo(character_ids)['elmo_representations'][0]
        rep2=self.token_embeds(input_data)
        rep=torch.cat([rep,rep2],dim=-1)
        output=rep+pos 
        shape_o = output.shape
        output = output.reshape(-1,self.emb_size+1024)
        res=self.project(output)
        res = self.dropout(res)
        output=res.reshape((shape_o[0],shape_o[1],self.h_size))
        return output
    

class ViT(nn.Module):
    def __init__(self, args, vocab_size, attn_type, ksvd_layers=1, low_rank=5, rank_multi=2, num_classes=10, \
                dropout=0., num_layers=7, hidden=384, mlp_hidden=384, head=8):
        super(ViT, self).__init__()
        self.attn_type = attn_type
        self.num_layers = num_layers
        self.ksvd_layers = ksvd_layers

        self.vocab_size = vocab_size
        self.max_len = args.max_len
        self.emb_dim = args.emb_dim
        self.hidden = hidden
        self.dropout = dropout
        self.embedding = Embeddings(vocab_size=self.vocab_size, max_len=self.max_len, emb_size=self.emb_dim, \
                                    h_size=self.hidden, drop_rate=self.dropout)
        
        enc_list = [TransformerEncoder(args=args, attn_type="softmax", low_rank=low_rank, rank_multi=rank_multi, \
                    feats=hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        if self.attn_type == "kep_svgp":
            for i in range(self.ksvd_layers):
                enc_list[-(i+1)] = TransformerEncoder(args=args, attn_type="kep_svgp", low_rank=low_rank, rank_multi=rank_multi, \
                    feats=hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head)
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )

    def forward(self, input_data, positional, inputs_mask, data):
        score_list = []
        Lambda_inv_list = []
        kl_list = []

        out = self.embedding.forward(input_data, positional, data) 
        for enc in self.enc:
            if enc.attn_type == "softmax":
                out = enc(out, inputs_mask)
            elif enc.attn_type == "kep_svgp":
                out, scores, Lambda_inv, kl = enc(out, inputs_mask)
                score_list.append(scores)
                Lambda_inv_list.append(Lambda_inv)
                kl_list.append(kl)

        out = out.mean(1)
        out = self.fc(out)

        if self.attn_type == "softmax":
            return out
        elif self.attn_type == "kep_svgp":
            return out, score_list, Lambda_inv_list, kl_list

def vit_cola(args, vocab_size, attn_type, ksvd_layers, low_rank, rank_multi):
    return ViT(args=args, vocab_size=vocab_size, attn_type=attn_type, ksvd_layers=ksvd_layers, num_classes=args.num_classes, low_rank=low_rank, rank_multi=rank_multi, \
                dropout=0.1, num_layers=args.depth, hidden=args.hdim, head=args.num_heads, mlp_hidden=args.hdim) 
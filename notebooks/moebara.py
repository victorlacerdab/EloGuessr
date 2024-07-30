import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEbara(nn.Module):
    def __init__(self, embdim):
        super(MoEbara, self).__init__()
        self.emb_layer = nn.Embedding(num_embedding=None, embedding_dim=None, padding_idx=None,
                                      freeze=None, device=None)
        self.pos_enc = PositionalEncoding(emb_dim=None, max_len=None)
        enc_layer = nn.TransformerEncoderLayer(d_model=None, nhead=None, dim_feedforward=None, dropout=None,
                                             batch_first=True, bias = False)
        
        self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer)
        self.fc_newbie = nn.Linear(in_features=embdim, out_features=1)
        self.fc_intermediate = nn.Linear(in_features=embdim, out_features=1)
        self.fc_expert = nn.Linear(in_features=embdim, out_features=1)

        self.gating = nn.Linear(in_features=embdim, out_features=3)

    def forward(self, x):
        out = self.emb_layer(x)
        out = self.pos_enc(out)
        out = self.encoder(out)
        out = out[:, 0, :]

        out_gating = self.gating(out)
        out_experts = (self.fc_newbie(out), self.fc_intermediate(out), self.fc_expert(out))

        return out_gating, out_experts

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, max_len: int):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

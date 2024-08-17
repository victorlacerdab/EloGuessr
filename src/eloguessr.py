import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class EloGuessr(nn.Module):
    def __init__(self, vocab_len: int, num_heads: int, num_encoder_layers: int, embdim: int, dim_ff: int,
                 padding_idx: int, max_match_len: int, pretrained_embeddings, device):
        super(EloGuessr, self).__init__()

        if isinstance(pretrained_embeddings, OrderedDict):
            pretrained_embeddings = torch.stack(list(pretrained_embeddings.values())).squeeze(0)

        if pretrained_embeddings is not None:
            self.emb_layer = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=padding_idx)
        else:
            self.emb_layer = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embdim, padding_idx=padding_idx)
        self.posenc = PositionalEncoding(emb_dim=embdim, max_len=max_match_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embdim, nhead=num_heads,
                                                    dim_feedforward=dim_ff, batch_first=True,
                                                    bias=True, dropout=0.1, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(embdim, 1)

    def forward(self, x):
        pad_mask = self.get_pad_mask(x, self.emb_layer.padding_idx)
        out = self.emb_layer(x)
        out = self.posenc(out)
        out = self.transformer_encoder(out, src_key_padding_mask=pad_mask)
        out = out[:, 0, :] # Gets only the embedding for the special '[ELO]' token.
        out = self.fc_out(out)
        out = out.squeeze(1)
        return out
    
    def get_pad_mask(self, batch_sequence: torch.tensor, pad_token: int) -> torch.tensor:
        return (batch_sequence == pad_token)
    
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
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class ChessPybara(nn.Module):
    def __init__(self, vocab_len: int, num_heads: int, num_decoder_layers: int, embdim: int, dim_ff: int, padding_idx: int, max_match_len: int, device):
        super(ChessPybara, self).__init__()
        self.emb_layer = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embdim, padding_idx=padding_idx)
        self.posenc = PositionalEncoding(emb_dim=embdim, max_len=max_match_len)
        decoder_layers = nn.TransformerDecoderLayer(d_model=embdim, nhead=num_heads, dim_feedforward=dim_ff, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layers, num_layers=num_decoder_layers)
        self.fc = nn.Linear(embdim, vocab_len)
        self.device = device

    def forward(self, x):
        out = self.emb_layer(x)
        out = self.posenc(out)
        mask = self.generate_square_subsequent_mask(out.size(1))
        out = self.decoder(tgt=out, memory=out, tgt_mask = mask)
        out = self.fc(out)
        return out
    
    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size), diagonal=1)).bool().to(self.device)
        return mask

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
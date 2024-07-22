import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class EloGuessr(nn.Module):
    def __init__(self, vocab_len: int, num_heads: int, embdim: int, dim_ff: int, padding_idx: int, max_match_len: int, device):
        super(EloGuessr, self).__init__()
        self.emb_layer = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embdim,
                                      padding_idx = padding_idx)
        self.posenc = PositionalEncoding(emb_dim=embdim, max_len=max_match_len)
        self.enc_block = EncoderBlock(embdim=embdim, num_heads=num_heads, dim_ff=dim_ff, device=device)
        self.dec_block = LinearDecoder(embdim=embdim, device=device)
        #self.dec_block = TransfDec(embdim=embdim, num_heads=num_heads, dim_ff=dim_ff, device=device)
        self.conv1d = nn.Conv1d(in_channels=max_match_len, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.fc_out = nn.Linear(embdim, 1)

    def forward(self, x):
        out = self.emb_layer(x)
        out = self.posenc(out)
        out = self.enc_block(out)
        # out = self.dec_block(out, out)[:, :, :] # We pick the '[ELO]' token appended at the end as the feature vector
        out = F.relu(self.conv1d(out).squeeze())
        out = self.dec_block(out)
        out = self.fc_out(out)
        return out
    
class EncoderBlock(nn.Module):
    def __init__(self, embdim: int, num_heads: int, dim_ff: int, device):
        super(EncoderBlock, self).__init__()
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=embdim, nhead=num_heads,
                                                        dim_feedforward=dim_ff, dropout=0.1,
                                                        activation='relu', batch_first=True,
                                                        device=device)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=embdim, nhead=num_heads,
                                                        dim_feedforward=dim_ff, dropout=0.1,
                                                        activation='relu', batch_first=True,
                                                        device=device)
        self.encoder_layer3 = nn.TransformerEncoderLayer(d_model=embdim, nhead=num_heads,
                                                        dim_feedforward=dim_ff, dropout=0.1,
                                                        activation='relu', batch_first=True,
                                                        device=device)
        self.encoder_layer4 = nn.TransformerEncoderLayer(d_model=embdim, nhead=num_heads,
                                                        dim_feedforward=dim_ff, dropout=0.1,
                                                        activation='relu', batch_first=True,
                                                        device=device)
        
    def forward(self, x):
        out = self.encoder_layer1(x)
        out = self.encoder_layer2(out)
        out = self.encoder_layer3(out)
        out = self.encoder_layer4(out)
        return out

class TransfDec(nn.Module):
    def __init__(self, embdim: int, num_heads: int, dim_ff: int,  device):
        super(TransfDec, self).__init__()
        self.decode_layer1 = nn.TransformerDecoderLayer(d_model=embdim, nhead=num_heads,
                                                        dim_feedforward=dim_ff, batch_first=True,
                                                        device=device)

        self.decode_layer2 = nn.TransformerDecoderLayer(d_model=embdim, nhead=num_heads,
                                                        dim_feedforward=dim_ff, batch_first=True,
                                                        device=device)
        
    def forward(self, x, memory):
        out = self.decode_layer1(x, memory)
        out = self.decode_layer2(out, out)
        return out

class LinearDecoder(nn.Module):
    def __init__(self, embdim: int, device):
        super(LinearDecoder, self).__init__()
        self.linblock1 = LinearBlock(embdim, device=device)
        self.linblock2 = LinearBlock(embdim, device=device)
        self.linblock3 = LinearBlock(embdim, device=device)
    
    def forward(self, x):
        out = self.linblock1(x)
        out = self.linblock2(out)
        out = self.linblock3(out)
        return out
    
class LinearBlock(nn.Module):
    def __init__(self, emb_dim, device):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(in_features=emb_dim, out_features=emb_dim, device=device)
        self.lnorm = nn.LayerNorm(emb_dim, device=device)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = F.relu(self.fc(x))
        out = self.lnorm(out)
        out = self.dropout(out)
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, max_len: int = 5000):
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

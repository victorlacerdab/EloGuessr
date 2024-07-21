import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class EloGuessr(nn.Module):
    def __init__(self, vocab_len: int, num_heads: int, embdim: int, dim_ff: int, padding_idx: int, device):
        super(EloGuessr, self).__init__()
        self.emb_layer = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embdim,
                                      padding_idx = padding_idx)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embdim, nhead=num_heads,
                                                        dim_feedforward=dim_ff, dropout=0.1,
                                                        activation='relu', batch_first=True,
                                                        device=device)
        self.linblock1 = LinearBlock(embdim)
        self.linblock2 = LinearBlock(embdim)
        self.linblock3 = LinearBlock(embdim)
        self.fc_out = nn.Linear(embdim, 1)

    def forward(self, x):
        out = self.emb_layer(x)
        out = self.encoder_layer(out)
        out = out[:, -1, :]

        out = self.linblock1(out)
        out = self.linblock2(out)
        out = self.linblock3(out)

        out = self.fc_out(out)
        return out
    
class LinearBlock(nn.Module):
    def __init__(self, emb_dim):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.lnorm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = F.relu(self.fc(x))
        out = self.lnorm(x)
        out = self.dropout(x)
        return out


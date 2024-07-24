import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from eloguessr import EloGuessr
from utils import load_data, plot_losses, load_json_dict

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_dir = '/Home/siv33/vbo084/EloGuessr/data/processed/'
fnames = ['chess_train_both_full.pt', 'chess_val_both_full.pt', 'chess_test_both_full.pt']
specnames = ['chess_both_full.json']

dset_specs = load_json_dict(os.path.join(data_dir, specnames[0]))
print(dset_specs)

VOCAB_LEN = dset_specs['vocab_len']
PAD_IDX = dset_specs['pad_idx']
MAX_LEN = dset_specs['match_len']
BATCH_SIZE = 256

train_dloader, val_dloader, test_dloader = load_data(data_dir, fnames, batch_size=BATCH_SIZE)
del test_dloader

config_dict = {'emb_dim': 1024,
               'vocab_len': VOCAB_LEN,
               'max_match_len': MAX_LEN,
               'num_heads': 16,
               'padding_idx': PAD_IDX,
               'dim_ff': 1024,
               'epochs': 10,
               'lr': 0.001}

def train_model(traindloader, valdloader, config_dict, device):
    epochs = config_dict['epochs']
    model = EloGuessr(vocab_len=config_dict['vocab_len'], num_heads=config_dict['num_heads'],
                      embdim=config_dict['emb_dim'], dim_ff=config_dict['dim_ff'],
                      padding_idx=config_dict['padding_idx'], max_match_len = config_dict['max_match_len'],
                      device = device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['lr'])
    loss_fn = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for batch in traindloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(traindloader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in valdloader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, targets.unsqueeze(1))
                
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(valdloader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model, train_losses, val_losses

print('Starting to train.')
model, tls, vls = train_model(train_dloader, val_dloader, config_dict, device)
plot_losses(tls, vls)

torch.save(model, os.path.join(data_dir, 'model_fullvar.pt'))
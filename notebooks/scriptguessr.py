import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from eloguessr import EloGuessr
from utils import load_data, plot_losses, load_json_dict

torch.manual_seed(22)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = '/Home/siv33/vbo084/EloGuessr/data/processed/'
emb_data_dir = '/Home/siv33/vbo084/EloGuessr/models/embeddings'
model_data_dir = '/Home/siv33/vbo084/EloGuessr/models/final_models/'

fnames = ['chess_train_both_medium.pt', 'chess_val_both_medium.pt', 'chess_test_both_medium.pt']
specnames = ['chess_both_medium.json']

dset_specs = load_json_dict(os.path.join(data_dir, specnames[0]))
print(dset_specs)
VOCAB_LEN = dset_specs['vocab_len']
PAD_IDX = dset_specs['pad_idx']
MAX_LEN = dset_specs['match_len']

BATCH_SIZE = 2056
train_dloader, val_dloader, test_dloader = load_data(data_dir, fnames, batch_size=BATCH_SIZE)
del test_dloader

config_dict = {'emb_dim': 512,
               'vocab_len': VOCAB_LEN,
               'max_match_len': MAX_LEN,
               'num_heads': 4,
               'num_enc_layers': 4,
               'padding_idx': PAD_IDX,
               'dim_ff': 1024,
               'epochs': 100,
               'lr': 0.0001,
               'embeddings': torch.load(os.path.join(emb_data_dir, 'dcdr_emb512_elite_medium_45_epcs.pt'))}

def train_model(traindloader, valdloader, config_dict, device):
    epochs = config_dict['epochs']
    model = EloGuessr(vocab_len=config_dict['vocab_len'], num_heads=config_dict['num_heads'],
                      num_encoder_layers=config_dict['num_enc_layers'], embdim=config_dict['emb_dim'], dim_ff=config_dict['dim_ff'],
                      padding_idx=config_dict['padding_idx'], max_match_len = config_dict['max_match_len'],
                      pretrained_embeddings = config_dict['embeddings'], device = device)
    
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
            loss = loss_fn(outputs, targets)
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
                loss = loss_fn(outputs, targets)
                
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(valdloader)
        val_losses.append(avg_val_loss)

        if (epoch+1) % 5 == 0:
            torch.save(model, os.path.join(model_data_dir, f'model_both_medium_{epoch}epcs.pt'))
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    print('Saving final model.')
    torch.save(model, os.path.join(model_data_dir, f'model_both_medium_final.pt'))
    
    return model, train_losses, val_losses

print(f'Starting to train EloGuessr.')
model, tls, vls = train_model(train_dloader, val_dloader, config_dict, device)
plot_losses(tls, vls)
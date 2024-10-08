import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from eloguessr import EloGuessr
from utils import path_dict, load_data, plot_losses, load_json_dict

torch.manual_seed(22)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = path_dict['data_dir'] # Data will be loaded from here
emb_data_dir = path_dict['emb_dir'] # Embeddings will be loaded from here
model_data_dir = path_dict['model_dir'] # Models will be saved here

fnames = ['chess_train_both_medium.pt', 'chess_val_both_medium.pt', 'chess_test_both_medium.pt'] # Names of your preprocessed files
specnames = ['chess_both_medium.json']

dset_specs = load_json_dict(os.path.join(data_dir, specnames[0]))
VOCAB_LEN = dset_specs['vocab_len']
PAD_IDX = dset_specs['pad_idx']
MAX_LEN = dset_specs['match_len']

BATCH_SIZE = 1024
train_dloader, val_dloader, test_dloader = load_data(data_dir, fnames, batch_size=BATCH_SIZE) # Creates dataloaders
del test_dloader

config_dict = {'emb_dim': 512,
               'vocab_len': VOCAB_LEN,
               'max_match_len': MAX_LEN,
               'num_heads': 4,
               'num_enc_layers': 4,
               'padding_idx': PAD_IDX,
               'dim_ff': 1024,
               'epochs': 10,
               'lr': 0.0001,
               'embeddings': torch.load(os.path.join(emb_data_dir, 'PRETRAINED_EMB_NAME.pt'))
               }

def train_model(traindloader, valdloader, pretrained_model, config_dict, device):

    '''
    If you want to load a pretrained_model, pass it to the function.
    Otherwise, pass None.
    '''

    epochs = config_dict['epochs']
    if pretrained_model is None:
        model = EloGuessr(vocab_len=config_dict['vocab_len'], num_heads=config_dict['num_heads'],
                        num_encoder_layers=config_dict['num_enc_layers'], embdim=config_dict['emb_dim'], dim_ff=config_dict['dim_ff'],
                        padding_idx=config_dict['padding_idx'], max_match_len = config_dict['max_match_len'],
                        pretrained_embeddings = config_dict['embeddings'], device = device)
        model = model.to(device)
    else:
        print('Loading pretrained model.')
        model = pretrained_model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['lr'])
    loss_fn = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 3
    
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter == patience:
            print(f"Validation loss increased for {patience} consecutive epochs. Stopping training.")
            torch.save(model, os.path.join(model_data_dir, f'eloguessr_earlystop_{epoch+1}epcs.pt'))
            break

        if (epoch+1) % 10 == 0:
            torch.save(model, os.path.join(model_data_dir, f'model_both_medium_{epoch}epcs.pt'))
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    print('Saving final model.')
    torch.save(model, os.path.join(model_data_dir, f'model_both_medium_fffinal.pt'))
    
    return model, train_losses, val_losses

print(f'Starting to train EloGuessr.')
model, tls, vls = train_model(train_dloader, val_dloader, None, config_dict, device)
plot_losses(tls, vls)
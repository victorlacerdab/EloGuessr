import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

class EloGuessrDset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.tensor(samples, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
    
def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_gameslendist(encoded_dataset: list, num_buckets: int):

    length_distribution = {}
    for game in encoded_dataset:  
        lengame = len(game)
        if lengame not in length_distribution.keys():
            length_distribution[lengame] = 1
        else:
            length_distribution[lengame] += 1

    num_moves = list(length_distribution.keys())

    max_moves = max(num_moves)
    min_moves = min(num_moves)
    bucket_size = (max_moves - min_moves) // num_buckets

    buckets = [(min_moves + i * bucket_size, min_moves + (i + 1) * bucket_size) for i in range(num_buckets)]
    bucketed_counts = [0] * num_buckets

    # Assign the counts to the buckets
    for move, count in length_distribution.items():
        for i, (start, end) in enumerate(buckets):
            if start <= move < end:
                bucketed_counts[i] += count
                break

    # Include the last bucket to capture the upper edge case
    bucketed_counts[-1] += sum(count for move, count in length_distribution.items() if move >= buckets[-1][1])

    # Labels
    bucket_labels = [f"{start}-{end-1}" for start, end in buckets]
    bucket_labels[-1] = f"{buckets[-1][0]}+"

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(bucket_labels, bucketed_counts, color='blue', alpha=0.7)
    plt.xlabel('Number of Moves (Buckets)')
    plt.ylabel('Number of Games')
    plt.title('Histogram of Chess Game Lengths')
    plt.show()


def load_data(path: str, fnames: list[str], batch_size: int) -> DataLoader:
    train_dloader = DataLoader(torch.load(os.path.join(path, fnames[0])), shuffle=True, batch_size=batch_size)
    val_dloader = DataLoader(torch.load(os.path.join(path, fnames[1])), shuffle=False, batch_size=batch_size)
    test_dloader = DataLoader(torch.load(os.path.join(path, fnames[2])), shuffle=False, batch_size=batch_size)
    
    return train_dloader, val_dloader, test_dloader
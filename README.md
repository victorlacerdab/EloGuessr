# EloGuessr

![Chess Image](https://i.ytimg.com/vi/Y1xzL8I9sAE/maxresdefault.jpg)

Based on the popular YouTube series by [GothamChess](https://www.youtube.com/@GothamChess), EloGuessr is a Transformers-based model for, you guessed it, guessing t he ELO.

Details about dataset creation are provided below.

# Data

We use data collected from Lichess' open database ([Lichess Open Database](https://database.lichess.org/)) and, for quick access to high rated ELO games, we used the [Lichess Elite Database](https://database.nikonoel.fr/) compiled by Niko Noel.

In order to train a more robust model, bullet games are discarded. Other than by a player resigning or being disconnected, the fastest way that a match can end is through a Fool's mate, where black can win against white by making only two moves.

The ELO in the dataset ranges from 1000 to 3000, with a roughly equal distribution of samples in each range.

# Model architecture

Inspired by the idea of training word embeddings introduced by word2vec, a two-step strategy is put in place to produce the best possible ELO predictions.

First, a model is trained on the task of next-token prediction using a decoder-only architecture. The idea is that we first "teach" the model how to play chess and employ transfer learning by using the learned token embeddings from ChessPybara to the final EloGuessr model. 

At the second step, we train a encoder-only transformer model that uses four stacked, standard encoder layers to produce a dense representation of each match. This final representation is passed to a linear layer, whose output is the predicted ELO for the match.

One important observation is that the model is predicting the average ELO of the match instead of each player's individual ELO.

Although this might seem counterintuitive at first, it is easily justifiable. Since the dataset is composed of public matches on the Lichess.org platform, players are overwhelmingly pitted against opponents of roughly the same skill level, resulting in samples where both players are evenly matched. Moreover, if there is a high ELO differential between players, it becomes increasingly hard to estimate the ELO of the higher ranked player, as players ranked, for example, in the 1600-2500 band can easily toy with beginners.

# Results

The results for +- 25, +- 50, +- 100, +- 250, and +-500 are shown in the table below.

| Top-N  | Accuracy (%) |
|--------|--------------|
| @25    | 14.99        |
| @50    | 29.41        |
| @100   | 53.60        |
| @250   | 83.75        |
| @500   | 94.54        |

# Improvements

One possible experiment to ensure better performance would be to use a Misture of Experts framework.
A model could be trained on the output of the Transformer's encoder to classify matches into 'Beginner' (ELO 0-999), 'Intermediate' (ELO 1000-1999), and 'Advanced' (2000+), and one of three specialized models for regression could be called to guess the ELO based on that information.

Another source of improvement could come from better tokenization, such as using a BPE tokenizer or off-the-shelf chess tokenizers. Since this is just a proof-of-concept model, I decided to keep it simple, but better tokenization, with a smaller vocabulary would likely lead to better results.
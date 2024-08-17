# EloGuessr

Inspired by the popular YouTube series by [GothamChess](https://www.youtube.com/@GothamChess), EloGuessr is a Transformers-based model for, you guessed it, guessing the ELO of chess matches.

# Data

We use data collected from Lichess' open database ([Lichess Open Database](https://database.lichess.org/)) and, for quick access to high rated ELO games, we used the [Lichess Elite Database](https://database.nikonoel.fr/) compiled by Niko Noel.

From the Open Database, we used games from [June 2014](https://database.lichess.org/standard/lichess_db_standard_rated_2014-06.pgn.zst). From the Elite Database, games from [May 2020](https://database.nikonoel.fr/lichess_elite_2021-05.zip).

In order to train a more robust model, bullet games are discarded. Other than by a player resigning or being disconnected, the fastest way that a match can end is through a Fool's mate, where black can win against white by making only two moves.

The ELO in the dataset ranges from 1000 to 3000, with a roughly equal distribution of samples in each range, except for the 2500-3000 ranges, which has about half the amount of samples from the other ranges (it being more difficult to find enough games in this range).

**Disclaimer**: Since the data was extracted from Lichess, the results are incomparable with matches from Chess.com.


# Model architecture

Inspired by the idea of training word embeddings introduced by word2vec, a two-step strategy is put in place to produce the best possible ELO predictions.

First, a model is trained on the task of next-token prediction using a decoder-only architecture. The idea is that we first "teach" the model how to play chess and employ transfer learning by using the learned token embeddings from ChessPybara to the final EloGuessr model. 

At the second step, we train a encoder-only transformer model that uses four stacked, standard encoder layers to produce a dense representation of each match. This final representation is passed to a linear layer, whose output is the predicted ELO for the match.

One important observation is that the model is predicting the average ELO of the match instead of each player's individual ELO.

Although this might seem counterintuitive at first, it is easily justifiable. Since the dataset is composed of public matches on the Lichess.org platform, players are overwhelmingly pitted against opponents of roughly the same skill level, resulting in samples where both players are evenly matched. Moreover, if there is a high ELO differential between players, it becomes increasingly hard to estimate the ELO of the higher ranked player, as players ranked, for example, in the 1600-2500 band can easily toy with beginners.

# Results

The table below shows the performance of the trained models on the tolerance-accuracy metric. Models 0 and 1 share the same hyperparameters (num_heads=4, enc_layers=8), but model 0 uses pretrained embeddings for the vocabulary while model 1 does not. Model 2 has double the amount of parameters of models 0 and 1, and was trained for much longer.

| Model | TAcc@25 | TAcc@50 | TAcc@100 | TAcc@250 | TAcc@500 |
|-------|---------|---------|----------|----------|----------|
| 0 | 15.24% | 29.56% | 51.89% | 82.18% | 94.48% |
| 1 | 14.71% | 28.78% | 52.21% | 82.45% | 94.33% |
| 2 | 16.07% | 31.16% | 53.79% | 82.51% | 93.92% |

The larger model performs better accross all dimensions except (surprisingly) for tolerance-accuracy at 500, while model 0, trained with pre-trained embeddings, performs better than the one without embeddings, although the difference is marginal.

One question that arises from this table is whether certain ELO ranges are easier to be correctly predicted than others. For example, one would expect beginner matches (mean ELO 1000-1499) to have much more move variance during the opening phase since new players have typically not memorized openings as well as more advanced players, and therefore make more "random" moves.

The table below breaks down the TAcc@k for the following ELO ranges: 1000-1499, 1500-1999, 2000-2499, 2500-3000.

| Model | Range | TAcc@25 | TAcc@50 | TAcc@100 | TAcc@250 | TAcc@500 |
|-------|-------|---------|---------|----------|----------|----------|
| 0 | 1000-1499 | 7.54% | 15.15% | 31.42% | 75.15% | 95.60% |
| 0 | 1500-1999 | 12.91% | 25.43% | 47.47% | 81.00% | 93.37% |
| 0 | 2000-2499 | 26.92% | 51.45% | 75.99% | 87.96% | 95.11% |
| 0 | 2500+ | 0.00% | 0.00% | 27.90% | 81.53% | 94.57% |
| 1 | 1000-1499 | 7.79% | 15.79% | 32.22% | 75.63% | 95.62% |
| 1 | 1500-1999 | 13.46% | 25.89% | 47.48% | 80.94% | 93.02% |
| 1 | 2000-2499 | 24.39% | 45.25% | 73.48% | 87.95% | 95.09% |
| 1 | 2500+ | 0.33% | 9.51% | 38.40% | 83.79% | 94.44% |
| 2 | 1000-1499 | 11.75% | 22.31% | 42.43% | 79.85% | 94.56% |
| 2 | 1500-1999 | 11.09% | 21.84% | 41.38% | 75.80% | 90.72% |
| 2 | 2000-2499 | 29.09% | 56.31% | 81.63% | 90.95% | 96.54% |
| 2 | 2500+ | 0.00% | 0.00% | 30.64% | 86.38% | 96.69% |

This table provides a comprehensive comparison of the three models' performance across different accuracy thresholds and target value ranges.

It is interesting to notice that all models struggle on the +-25 and +-50 range, but Model 1 still manages to achieve some degree of success on TAcc@50.

According to the results presented, the effects of using pre-trained embeddings to perform regression is somewhat diminished, but could still be useful with better tokenization.

# Improvements

One possible experiment to ensure better performance would be to use a Misture of Experts framework. From the per-ELO range evaluation table, we see that the models perform quite well on TAcc@250. It should be possible, therefore, to train a model on the output of the Transformer's encoder to classify matches into 'Beginner' (ELO 0-999), 'Intermediate' (ELO 1000-1999), and 'Advanced' (2000+), and one of three specialized models for regression could be called to guess the ELO based on that information. This would very likely improve performance.

Another source of improvement could come from better tokenization, such as using a BPE tokenizer or off-the-shelf chess tokenizers. Since this is just a proof-of-concept model, I decided to pass moves that meet a certain threshold (must appear at least 5 times in the dataset), but better tokenization, with a smaller vocabulary would likely lead to better results. This also makes sense since algebraic notation for describing chess games follows certain definite rules, and a small vocab size would help the learning process.

More data! Chess data is abundant, and the Lichess database provides an enormous amount of data to work with. For a production-grade application, used for educational, entertainment or fraud detection purposes, a model could be trained on a much, much larger dataset and achieve even better results. Due to the compute constraints in this project, only around a million matches were used.
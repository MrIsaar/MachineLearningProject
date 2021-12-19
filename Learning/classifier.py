import math
import random
import numpy as np
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from torch import nn, Tensor

from GamesData import GamesData
from proccessFiles import genericfiles
from proccessFiles import processCSV2

from torch.utils.data import dataset
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
print("proccessing file")



#gamedata = GamesData(genericfiles("steam","steam.csv"),["appid","name","release_date","english","developer","publisher","platforms","required_age","categories","genres","steamspy_tags","achievements","positive_ratings","negative_ratings","average_playtime","median_playtime","owners","price"])

likedGames = processCSV2(genericfiles("steam","steamgenresAction1.csv"))
dislikedGames = processCSV2(genericfiles("steam","steamgenresIndie0.csv"))






attributes, lgames = likedGames
a ,dgames = dislikedGames

normalizer = np.ones((1,len(lgames[0][0])))
games = (np.concatenate((lgames[0],dgames[0])),np.matmul(np.concatenate((lgames[1],dgames[1])),normalizer))
train_dataloader = DataLoader(games, batch_size=64, shuffle=True)

# Display image and label.
train_labels, train_features = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")





class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits





model = NeuralNetwork().to(device)
print(model)





# #classifer neural network
# cnn = NeuralNetwork(len(attributes),len(attributes)*2,1)

# input = torch.from_numpy(games[:1])


# output = cnn(input)

# labels = torch.ones((len(games),1))
# output, loss = cnn.train(labels, games,1)
# current_loss = loss


print("done")
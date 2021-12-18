import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor

from GamesData import GamesData
from proccessFiles import genericfiles
from proccessFiles import processCSV2

from torch.utils.data import dataset
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

print("proccessing file")
"""attributes,games = processCSV2(genericfiles("steam","steam.csv"))
print(attributes)
games = torch.from_numpy(games)
if torch.cuda.is_available():
    games = games.to('cuda')
    print(f"Device tensor is stored on: {games.device}")
print("done")"""


gamedata = GamesData(genericfiles("steam","steam.csv"),["appid","name","release_date","english","developer","publisher","platforms","required_age","categories","genres","steamspy_tags","achievements","positive_ratings","negative_ratings","average_playtime","median_playtime","owners","price"])

print(len(gamedata))
print(gamedata.__getitem__(1))

train_dataloader = DataLoader(gamedata, batch_size=64, shuffle=True)

#iterat = iter(train_dataloader)
#train_features, train_labels = next(iterat)
#print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")


train_iter = iter(gamedata)
#train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)


print("done")
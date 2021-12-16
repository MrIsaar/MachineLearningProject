import math
import numpy as np
import torch


from GamesData import GamesData
from proccessFiles import genericfiles
from proccessFiles import processCSV2

print("proccessing file")
"""attributes,games = processCSV2(genericfiles("steam","steam.csv"))
print(attributes)
games = torch.from_numpy(games)
if torch.cuda.is_available():
    games = games.to('cuda')
    print(f"Device tensor is stored on: {games.device}")
print("done")"""


gamedata = GamesData(genericfiles("steam","steam.csv"),['appid', 'name', 'release_date', 'english', 'developer', 'publisher','platforms', 'required_age', 'categories', 'genres', 'steamspy_tags', 'median_playtime', 'owners','price'])

print(len(gamedata))
print(gamedata.__getitem__(1))
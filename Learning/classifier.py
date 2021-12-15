import math
import numpy as np
import torch
from fileprocessing.proccessFiles import genericfiles
from fileprocessing.proccessFiles import processCSV


games = processCSV(genericfiles("steam","steam.csv"))
print(games[0])
import math
import numpy as np
import torch
import sys
sys.path.insert(1, 'fileprocessing')
from proccessFiles import genericfiles
from proccessFiles import processCSV2


games = processCSV2(genericfiles("steam","steam.csv"))
print(games[0])
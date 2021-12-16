import torch
from torch.utils.data import Dataset

import sys
import os
import pandas as pd

class GamesData(Dataset):
    def __init__(self,gamefile,attributes):
        #super().__init__()
        self.games = pd.read_csv(gamefile,names=attributes,quotechar='"',encoding='utf-8')
        
    def __len__(self):
        return len(self.games)
    
    def __getitem__(self, index):
        return self.games.__getitem__(index)
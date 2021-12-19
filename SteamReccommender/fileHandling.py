import os
import sys
from sys import platform
import pandas as pd
import numpy as np
from datetime import datetime

def genericfiles(folder,filenameEnd):
    dirname =  os.getcwd()
    carTrainFile = ""
    carDescriptFile = ""
    if dirname.endswith("DecisionTree"):
        dirname = os.path.dirname(dirname)
    if platform == "linux" or platform == "linux2":
       # carTrainFile = dirname + "/car/train.csv"
       # carDescriptFile = dirname + "/car/data-desc.txt"
       return dirname + "/"+folder+"/" + filenameEnd
    else:
      # carTrainFile = dirname + "\\car\\train.csv"
      # carDescriptFile = dirname + "\\car\\data-desc.txt"
      return dirname + "\\"+folder+"\\" + filenameEnd
  
def valuesfromFile(file):
    
    trainSamples = pd.read_csv(
        file,encoding="utf-8",delimiter=',',quotechar='"',skipinitialspace=True,
        names=["appid","name","release_date","english","developer","publisher","platforms","required_age","categories","genres","steamspy_tags","achievements","positive_ratings","negative_ratings","average_playtime","median_playtime","owners","price","label"]
        )
    train_features = trainSamples.copy()
    train_labels = train_features.pop('label')
    
    train_features_dict = {name: np.array(value) for name, value in train_features.items()}
    
    return train_features_dict
    
def getValuesToPredict(train_features_dict,start,size=1):
    features_dict = {name:values[start:size+start] for name, values in train_features_dict.items()}
    return features_dict
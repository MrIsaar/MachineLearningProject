import os
from os.path import exists
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
  
def valuesfromFile(file,hasLabel=True):
    
    if hasLabel:
        trainSamples = pd.read_csv(
            file,encoding="utf-8",delimiter=',',quotechar='"',skipinitialspace=True,
            names=["appid","name","release_date","english","developer","publisher","platforms","required_age","categories","genres","steamspy_tags","achievements","positive_ratings","negative_ratings","average_playtime","median_playtime","owners","price","label"]
            )
        train_features = trainSamples.copy()
        train_labels = train_features.pop('label')
        train_features_dict = {name: np.array(value) for name, value in train_features.items()}
    else:
        trainSamples = pd.read_csv(
            file,encoding="utf-8",delimiter=',',quotechar='"',skipinitialspace=True,
            names=["appid","name","release_date","english","developer","publisher","platforms","required_age","categories","genres","steamspy_tags","achievements","positive_ratings","negative_ratings","average_playtime","median_playtime","owners","price"]
            )
        train_features = trainSamples.copy()
        train_features_dict = {name: np.array(value)[1:] for name, value in train_features.items()}
    
  
    
    return train_features_dict


def writeList(line,file,label):
    item_n = 0
    for item in line:
        file.write('"'+str(item)+'"')
        if(item_n < len(line)-1):
            file.write(',')
        else:
            file.write(",\""+str(label)+"\"\n")
        item_n+=1

    
def getValuesToPredict(train_features_dict,start=0,size=1):
    features_dict = {name:values[start:size+start] for name, values in train_features_dict.items()}
    return features_dict

def appendFile(filename, appendString,label):
    file_exists = exists(filename)
    if file_exists:
        with open(filename,"a",encoding='utf-8') as file:
            writeList(appendString,file,label)
    else:
        with open(filename,"w",encoding='utf-8') as file:
            writeList(appendString,file,label)
    

#appendFile(genericfiles("steam","test.csv"),["10","Counter-Strike","2000-11-01","1","Valve","Valve","windows;mac;linux","0","Multi-player;Online Multi-Player;Local Multi-Player;Valve Anti-Cheat enabled","Action","Action;FPS;Multiplayer","0","124534","3339","17612","317","10000000-20000000","7.19"],"1")
    
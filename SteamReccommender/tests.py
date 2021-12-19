import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from fileHandling import *


def resultsSorted(nn,databatch):
    results = nn(databatch)
    output = {}
    for i in range(len(results)):
        output[i] = results.numpy()[i]
    return sorted(output.items(), key = lambda kv:(kv[1], kv[0]) , reverse=True)

def predictResults(nn,databatch):
    results = resultsSorted(nn,databatch)
    output = ""
    correct = 0
    total = 0
    indexList = np.array(results).T[0]
    #for i in range(len(results)):
    for i in indexList: # iterate in order of largest satisfaction to least
        if databatch['genres'][i].__contains__('Action') and results[i][1] > 0:
            correct+=1
            total+=1
        elif databatch['genres'][i].__contains__('Indie') and results[i][1] > 0:
            
            total+=1
        elif databatch['genres'][i].__contains__('Action'):
            total+=1
            

    return correct,total
    
try:
    PreviousLearned = tf.keras.models.load_model('ActionGoodIndieBad')
except:
    """ This would be learn model"""
    print("not learned model good bye")
    exit()

features_dict = valuesfromFile(genericfiles("steam","testAction.csv"))

count,total = predictResults(PreviousLearned,features_dict)
print("test Action vs Indie only: ",count,"/",total)

large_features_dict = valuesfromFile(genericfiles("steam","trainPopular.csv"))

count,total = predictResults(PreviousLearned,large_features_dict)
print("largeDataset: ",count,"/",total)


print("done")
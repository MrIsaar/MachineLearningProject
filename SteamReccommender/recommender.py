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
   
    
        
def recommended(nn,databatch,maxcount=None,NameOnly=False):
    results = resultsSorted(nn,databatch)
    output = ""
    approved = 0
    ignored = 0
    indexList = np.array(results).T[0]
    #for i in range(len(results)):
    for i in indexList: # iterate in order of largest satisfaction to least
         
        if np.random.randint(100) < 20+approved*2-ignored:
            ignored +=1
            continue
        approved+=1
        output += str(databatch['name'][i]) 
        if not NameOnly:
            output += " by "+ str(databatch['developer'][i]) +" for "+ str(databatch['price'][i]) + " Genres "+str(databatch['genres'][i])+" appid: " + str(databatch['appid'][i]) + "\n"
        else:
            output += ","
        if not maxcount is None and approved >= maxcount:
            break
    if NameOnly:
        output += "\n"
    return approved,output[:-1]
    
try:
    PreviousLearned = tf.keras.models.load_model('ActionGoodIndieBad')
except:
    """ This would be learn model"""
    print("not learned model good bye")
    exit()

features_dict = valuesfromFile(genericfiles("steam","testAction.csv"))
subset_features_dict = getValuesToPredict(features_dict,start=450,size=100)

count,output = recommended(PreviousLearned,subset_features_dict,maxcount=10)
print(output)
print(f"that is {count} games")


subset_features_dict = getValuesToPredict(features_dict,start=350,size=300)
for i in range(10):
    count,output = recommended(PreviousLearned,subset_features_dict,maxcount=20,NameOnly=True)
    print(output)


print("done")
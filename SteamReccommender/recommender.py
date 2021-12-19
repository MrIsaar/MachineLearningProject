import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from fileHandling import *

class Recommender():
    def resultsSorted(self,nn,databatch):
        results = nn(databatch)
        output = {}
        for i in range(len(results)):
            output[i] = results.numpy()[i]
        return sorted(output.items(), key = lambda kv:(kv[1], kv[0]) , reverse=True)
    


    def recommended(self,nn=None,databatch=None,maxcount=None,NameOnly=False):
        if databatch is None:
            databatch = self.features_dict
        if nn is None:
            nn = self.PreviousLearned
        results = self.resultsSorted(nn,databatch)
        output = np.array([])
        approved = 0
        ignored = 0
        indexList = np.array(results).T[0]
        #for i in range(len(results)):
        for i in indexList: # iterate in order of largest satisfaction to least
            tout = ""
            if np.random.randint(100) < 40+approved*2-ignored*2:
                ignored +=1
                continue
            approved+=1
            tout += str(databatch['name'][i]) + " by "+ str(databatch['developer'][i]) +" for $"+ str(databatch['price'][i]) +" appid: " + str(databatch['appid'][i]) + "\n"
            output = np.append(output,tout)
            
            if not maxcount is None and approved >= maxcount:
                break
        
        return output


    def __init__(self):
        try:
            self.PreviousLearned = tf.keras.models.load_model('Popular')
        except:
            """ This would be learn model"""
            print("not learned model good bye")


        self.features_dict = valuesfromFile(genericfiles("steam","trainPopular.csv"))
        #subset_features_dict = getValuesToPredict(self.features_dict,start=450,size=100)

        # count,output = self.recommended(self.PreviousLearned ,subset_features_dict,maxcount=10)
        # print(output)
        # print(f"that is {count} games")
        # subset_features_dict = getValuesToPredict(self.features_dict,start=350,size=300)
        # for i in range(10):
        #     count,output = self.recommended(self.PreviousLearned ,subset_features_dict,maxcount=20,NameOnly=True)
        #     print(output)


if __name__ == "__main__":
    rr = Recommender()
    subset_features_dict = getValuesToPredict(rr.features_dict,start=450,size=100)
    
    output = rr.recommended(rr.PreviousLearned ,subset_features_dict,maxcount=10)
    print(output)
    
    subset_features_dict = getValuesToPredict(rr.features_dict,start=350,size=300)
    for i in range(10):
        output = rr.recommended(rr.PreviousLearned ,subset_features_dict,maxcount=20,NameOnly=True)
        print(output)
    
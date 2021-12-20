import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from fileHandling import *
from learnBehaviour import *

class Recommender():
    def resultsSorted(self,nn,databatch):
        results = nn(databatch)
        output = {}
        for i in range(len(results)):
            output[i] = results.numpy()[i]
        return sorted(output.items(), key = lambda kv:(kv[1], kv[0]) , reverse=True)
    
    def rate(self, index ,label):
        if self.recent is None:
            return
        out = self.recent[index]
        appendFile(genericfiles("steam","user.csv"),out,label)
        
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
        self.recent = None
        #for i in range(len(results)):
        for i in indexList: # iterate in order of largest satisfaction to least
            tout = ""
            if np.random.randint(100) < 40+approved*2-ignored*2:
                ignored +=1
                continue
            approved+=1
            dev = str(databatch['developer'][i])
            if len(dev) > 30:
                dev = dev[:29]
            tout += str(databatch['name'][i]) + " by "+ dev +" for $"+ str(databatch['price'][i]) +" appid: " + str(databatch['appid'][i]) + "\n"
            output = np.append(output,tout)
            if self.recent is None:
                self.recent = np.array([[values[i] for name, values in databatch.items()]])
            else:
                temp = np.array([[values[i] for name, values in databatch.items()]])
                self.recent = np.concatenate((self.recent,temp))
            
            
            if not maxcount is None and approved >= maxcount:
                break
        
        return output

    def updatemodel(self):
        self.PreviousLearned = learnfromFile(genericfiles("steam","user.csv"))

    def __init__(self):
        try:
            self.PreviousLearned = tf.keras.models.load_model('user')
        except:
            
            print("Model not learned, Starting learing")
            self.PreviousLearned = learnfromFile(genericfiles("steam","user.csv"))
            
            
            

        self.recent = None
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
    
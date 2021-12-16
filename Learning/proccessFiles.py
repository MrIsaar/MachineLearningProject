import os
from sys import platform
import numpy as np
from numpy.core.arrayprint import DatetimeFormat
import torch
import csv

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

"""
    used for processing a CSV file where each item is seperated with a newline
    and each attribute is comma seperated (',')
    i.e. 
    low,vhigh,4,4,big,med,acc
    low,high,5more,4,med,high,vgood

    debugprint will print the first debugprint number of elements in file. default is 0
"""



def processCSV(CSVfile, debugprint=0):
    
    index = 0
    items = []
    with open(CSVfile) as f:
        for line in f:
            terms = line.strip().split(',')

            items.append(terms)
            if debugprint > index:
                print(items[index])
            index += 1
    return items

def parseString(temp):
    if type(temp) is str:
        try:
            temp=float(temp)
        except:
            try:
                numbers = temp.split('-')
                if(len(numbers) == 3):
                    temp=datetime.strptime(temp, '%Y-%m-%d')
                    temp=temp.day+temp.month*100+temp.year*10000
                if(len(numbers) == 2):
                    return float(numbers[1]) - (float(numbers[0])/2)
            except:     
                temp=len(temp)
    return temp

def processCSV2(CSVfile):
    
    items = None
    attributes = None
    with open(CSVfile,newline='',encoding='utf-8') as csvFile:
        read = csv.reader(csvFile,delimiter=',',quotechar='"',skipinitialspace=True)
        for row in read:
            if attributes is None:
                
                attributes = np.array(row)
            else:
                
                item = np.ones((1,len(row)))
                for i in range(len(row)):
                    
                    temp = row[i]
                    
                    item[0][i] = parseString(temp)
                if items is None:
                    items = np.ones((1,len(row)))
                    items = items * item
                else:
                    items = np.concatenate((items,item))
            
    return (attributes,items)
    

def getCSVSubSet(CSVfile,SubSetsize=100,filter=None,outputname="CSVfile<filter>.csv"):
    """
    creates a subsetfile of a CSV of designated size 

    filter takes a tuple ("attribute", "value") to filter by. 
    subset will include only elements with that value

    returns filepath to CSVSubSet
    returns None if an error occured
    """
    index = 0
    items = []
    attributes = []
    currline = 0
    if outputname == "CSVfile<filter>.csv":
        sufix = "CSVfilesub.csv"
        if filter != None:
            sufix = "CSVfile"+filter[1] + ".csv"
        outputname = genericfiles("results",sufix)
    else:
        if outputname.endswith(".csv"):
            outputname = genericfiles("results",outputname)
        else:
            outputname = genericfiles("results",outputname) + ".csv"


    file = open(outputname,"w")
    try:
        with open(CSVfile) as f:
            for line in f:
                if currline == 0:
                    attributes = line.strip().split(',')
                    file.write(line)
                    currline+=1
                    continue
                if currline < SubSetsize:
                    if filter != None and attributes.__contains__(filter[0]):
                        values = line.strip().split(',')
                        if values[attributes.index(filter[0])].__contains__(filter[1]):
                            file.write(line)
                    else:
                        file.write(line)
                    currline+=1
                else:
                    break
    except:
        file.close()
        
        return None
    finally:
        file.close()     
    return outputname



if __name__ == "__main__":
    steamcsv = genericfiles("steam","steam.csv")
    filter = ["genres","Action"]
    print(getCSVSubSet(steamcsv,filter=filter,outputname="steamAction.csv"))

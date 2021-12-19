import os
import sys
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
            temp=np.float32(temp)
           
        except:
            try:
                numbers = temp.split('-')
                if(len(numbers) == 3):
                    temp=datetime.strptime(temp, '%Y-%m-%d')
                    temp=temp.day+temp.month*100+temp.year*10000
                elif(len(numbers) == 2):
                    return np.float32(numbers[1]) - (np.float32(numbers[0])/2)
                else:
                    temp=np.float32(len(temp))
            except:     
                temp=np.float32(len(temp))
    return temp

def processCSV2(CSVfile):
    
    items = None
    labels = None
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
                    labels = np.ones((1,1)) * item[0][-1]
                    items = items * item
                    
                else:
                    items = np.concatenate((items,item))
                    labels = np.concatenate((labels,item.T[-1:]))
    output = (items,labels)
    return (attributes,output)
    
    
def writeList(line,file,label):
    item_n = 0
    for item in line:
        file.write('"'+str(item)+'"')
        if(item_n < len(line)-1):
            file.write(',')
        else:
            file.write(",\""+str(label)+"\"\n")
        item_n+=1

def getCSVSubSet(CSVfile,label,SubSetsize=None,SubSetOffset=0,filter=None,outputname="CSVfile<filter>.csv"):
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


    file = open(outputname,"w",encoding='utf-8')
    try:
        with open(CSVfile,newline='',encoding='utf-8') as f:
            read = csv.reader(f,delimiter=',',quotechar='"',skipinitialspace=True)
            for line in read:
                if currline == 0:
                    attributes = line
                    
                    writeList(line,file,"label")
                    currline+=1
                    continue
                if SubSetsize == None or currline < SubSetOffset+SubSetsize and currline > SubSetOffset:
                    if filter != None and attributes.__contains__(filter[0]):
                        values = line
                        if values[attributes.index(filter[0])].__contains__(filter[1]):
                            item_n = 0
                            writeList(line,file,label)
                            currline+=1
                    else:
                        writeList(line,file,label)
                        currline+=1
                elif currline < SubSetOffset+SubSetsize:
                    currline+=1
                    continue
                else:
                    break
    except Exception as err:
        file.close()
        
        return None
    finally:
        file.close()     
    return outputname


def combineLists(file0,file1):
    
    list1 = processCSV2(file0)
    list2 = processCSV2(file1)

    attributes, lgames = list1
    a ,dgames = list2   
    normalizer = np.ones((1,len(lgames[0][0])))
    games = (np.concatenate((lgames[0],dgames[0])),np.matmul(np.concatenate((lgames[1],dgames[1])),normalizer))
    return attributes,games


if __name__ == "__main__":
    steamcsv = genericfiles("steam","steam.csv")
    filter = ["genres","Indie"]
    note = "test"
    label = 0
    subset = 500
    offset = 300
    print(getCSVSubSet(steamcsv,label,SubSetsize=subset,SubSetOffset=offset,filter=filter,outputname="steam"+note+str(filter[0])+str(filter[1])+str(label)+".csv"))

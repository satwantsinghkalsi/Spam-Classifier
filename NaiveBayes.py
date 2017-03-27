import os
import math
def train(path1,path2,stopwords):
    trainerHam={}
    trainerSpam={}
    totalDocs=0.0
    spam=0.0
    ham=0.0
    spamWords=0.0
    hamWords=0.0
    for filename in os.listdir(path1):
        ham+=1
        file = open(path1+filename,"r")
        for line in file:
            words=line.split()
            for word in words:
              if word not in stopwords:
                if word not in trainerHam:
                   trainerHam[word]=1.0
                else:
                   count=trainerHam[word]
                   trainerHam[word]=count+1 
        file.close()
    for filename in os.listdir(path2):
        spam+=1
        file = open(path2+filename,"r")
        for line in file:
            words=line.split()
            for word in words:
              if word not in stopwords:
                if word not in trainerSpam:
                   trainerSpam[word]=1.0
                else:
                   count=trainerSpam[word]
                   trainerSpam[word]=count+1
        file.close()
    totalDocs=ham+spam
    for key,value in trainerHam.items():
        hamWords+=value+1
    for key,value in trainerSpam.items():
        spamWords+=value+1
    for key,value in trainerHam.items():
        #print value,hamWords
        trainerHam[key]=math.log((value+1)/hamWords,2)
    for key,value in trainerSpam.items():
        trainerSpam[key]=math.log((value+1)/spamWords,2)
    trainer={"spam":{"prior":math.log((spam/totalDocs),2),"trainer":trainerSpam},
             "ham":{"prior":math.log((ham/totalDocs),2),"trainer":trainerHam}}
    return trainer
def classify(path1,path2,trainer,stopwords):
    totalDocs=0.0
    correct=0.0
    for filename in os.listdir(path1):
        totalDocs+=1
        file = open(path1+filename,"r")
        score={"spam":trainer["spam"]["prior"],"ham":trainer["ham"]["prior"]}
        for line in file:
            words =line.split()
            for word in words:
              if word not in stopwords:
                for key in score:
                    if word in trainer[key]["trainer"]:
                       score[key]+=trainer[key]["trainer"][word]
        if score["ham"]<score["spam"]:
           correct+=1
    for filename in os.listdir(path2):
        totalDocs+=1
        file = open(path2+filename,"r")
        score={"spam":trainer["spam"]["prior"],"ham":trainer["ham"]["prior"]}
        for line in file:
            words =line.split()
            for word in words:
              if word not in stopwords:
                for key in score:
                    if word in trainer[key]["trainer"]:
                       score[key]+=trainer[key]["trainer"][word]
        if score["spam"]<score["ham"]:
           correct+=1
    #print correct,totalDocs
    accuracy=correct/totalDocs*100
    return accuracy
             
'''def main():
    path1='train/ham/'
    path2='train/spam/'
    stopwords=[]
    file=open("stopwords.txt","r")
    for line in file:
        words=line.split()
        for word in words:
            if word not in stopwords:
               stopwords.append(word)
    file.close()
    trainer=train(path1,path2,[])  
    accuracy=classify("test/ham/","test/spam/",trainer,[])
    print "Accuracy with stopwords:"+str(accuracy)+"%"
    trainer_stop=train(path1,path2,stopwords)
    accuracy_stop=classify("test/ham/","test/spam/",trainer_stop,stopwords)
    print "Accuracy without stopwords:"+str(accuracy_stop)+"%"
if __name__=="__main__":
    main() 
'''

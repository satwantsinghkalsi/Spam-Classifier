import os
import math
from sys import argv
def sigmoid(weights,features,className,vocab):
    i=1
    x=weights[0]
    for word in vocab:
        x+=weights[i]*features[word]
        i+=1
    value=0.0
    if x<0:
       value=1-(1/(1+math.exp(x)))
    else:
       value=1/(1+math.exp(-x))
    if className=='1':
       return value
    else:
       return 1-value 
def train(path1,path2,vocab,e,l,itr):
    weights=[]
    lmbda=l
    eta=e
    iterations=itr
    weights.append(1.0)
    for i in range(1,len(vocab)+1):
        weights.append(0.0)
    X=[]
    Y=[]
    for filename in os.listdir(path1):
        file=open(path1+filename,'r')
        tmp={}
        for word in vocab:
            tmp[word]=0.0
        for line in file:
            words=line.split()
            for word in words:
                if word in vocab:
                   tmp[word]+=1  
        X.append(tmp)
        Y.append('1')
    for filename in os.listdir(path2):
        file=open(path2+filename,'r')
        tmp={}
        for word in vocab:
            tmp[word]=0.0
        for line in file:
            words=line.split()
            for word in words:
                if word in vocab:
                   tmp[word]+=1
        X.append(tmp)
        Y.append('0')
    for j in range(0,iterations):
        for i in range(0,len(X)):
            prob=sigmoid(weights,X[i],Y[i],vocab)
            k=1
            for word in vocab:
                classValue=0
                if Y[i]=='1':
                   classValue=1 
                weights[k]+=(eta*X[i][word]*(classValue-prob))-(eta*lmbda*weights[k]*weights[k])
                k+=1  
    return weights
def classify(path1,path2,vocab,trainer):
    correct=0.0
    totalcount=0.0
    for filename in os.listdir(path1):
        totalcount+=1
        x={}
        for word in vocab:
            x[word]=0.0
        file=open(path1+filename,'r')
        for line in file:
            words=line.split()
            for word in words:
                if word in vocab:
                  x[word]+=x[word]+1
        prob=[]
        prob.append(sigmoid(trainer,x,'1',vocab))
        prob.append(sigmoid(trainer,x,'0',vocab))
        if prob[0]>prob[1]:
           correct+=1
    for filename in os.listdir(path2):
        totalcount+=1
        x={}
        for word in vocab:
            x[word]=0.0
        file=open(path2+filename,'r')
        for line in file:
            words=line.split()
            for word in words:
                if word in vocab:
                  x[word]+=x[word]+1
        prob=[]
        prob.append(sigmoid(trainer,x,'1',vocab))
        prob.append(sigmoid(trainer,x,'0',vocab))
        if prob[0]<prob[1]:
           correct+=1
    return (correct/totalcount)*100         
def main():
    path1='train/ham/'
    path2='train/spam/'
    stopwords=[]
    eta=0.001
    lmbda=0.02
    file=open("stopwords.txt","r")
    for line in file:
        words=line.split()
        for word in words:
            if word not in stopwords:
               stopwords.append(word)
    file.close()
    vocab=[]
    paths=[]
    paths.append(path1)
    paths.append(path2)
    for path in paths: 
      for filename in os.listdir(path):
        file = open(path+filename,"r")
        for line in file:
            words=line.split()
            for word in words:
               if word not in vocab:
                  vocab.append(word)
        file.close()
    trainer=train(path1,path2,vocab,eta,lmbda,13) 
    #print trainer
    #print
    accuracy=classify("test/ham/","test/spam/",vocab,trainer)
    print "Accuracy with stopwords:"+str(accuracy)+"%"
    vocab_stop=[]
    for word in vocab:
        if word not in stopwords:
           vocab_stop.append(word)
    trainer_stop=train(path1,path2,vocab_stop,eta,lmbda,15)
    #print trainer_stop
    accuracy_stop=classify("test/ham/","test/spam/",vocab_stop,trainer_stop)
    print "Accuracy without stopwords:"+str(accuracy_stop)+"%"
    #print len(vocab),len(vocab_stop),stopwords
if __name__=="__main__":
    main() 


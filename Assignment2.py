import os
from sys import argv
import NaiveBayes as NB
import LogisticRegression as LR
def main():
    train_ham_folder=argv[1]
    train_spam_folder=argv[2]
    test_ham_folder=argv[3]
    test_spam_folder=argv[4]
    stopwords=[]
    file=open("stopwords.txt","r")
    for line in file:
        words=line.split()
        for word in words:
            if word not in stopwords:
               stopwords.append(word)
    file.close()
    vocab=[]
    paths=[]
    paths.append(train_ham_folder)
    paths.append(train_spam_folder)
    for path in paths:
      for filename in os.listdir(path):
        file = open(path+filename,"r")
        for line in file:
            words=line.split()
            for word in words:
               if word not in vocab:
                  vocab.append(word)
        file.close()
    vocab_stop=[]
    for word in vocab:
        if word not in stopwords:
           vocab_stop.append(word) 
    choice=''
    while choice!='5':
          print "1. Naive Bayes with stopwords."
          print "2. Naive Bayes without stopwords."
          print "3. Logistic Regression with stopwords."
          print "4. Logistic Regression without stopwords"
	  print "5. Exit"
          print "Enter choice(1-5)",
          choice=str(raw_input())
          if choice=='1':
             trainer=NB.train(train_ham_folder,train_spam_folder,[])
             accuracy=NB.classify(test_ham_folder,test_spam_folder,trainer,[])
             print "Accuracy with stopwords:"+str(accuracy)+"%"
          elif choice=='2':
             trainer=NB.train(train_ham_folder,train_spam_folder,stopwords)
             accuracy=NB.classify(test_ham_folder,test_spam_folder,trainer,stopwords)
             print "Accuracy without stopwords:"+str(accuracy)+"%"
          elif choice=='3':
             print "Do you want to enter lmbda,eta and iterations values yes/no"
             yesno=str(raw_input())
             if yesno=="yes":
                print "eta:"
                eta=float(raw_input())
                print "lmbda:"
                lmbda=float(raw_input())
                print "Iterations:"
                iterations=int(raw_input())
                trainer=LR.train(train_ham_folder,train_spam_folder,vocab,eta,lmbda,iterations)
                accuracy=LR.classify(test_ham_folder,test_spam_folder,vocab,trainer)
                print "Accuracy with stopwords:"+str(accuracy)+"%"
             else:
                eta=0.001
                lmbda=0.075
                iterations=12
                trainer=LR.train(train_ham_folder,train_spam_folder,vocab,eta,lmbda,iterations)
                accuracy=LR.classify(test_ham_folder,test_spam_folder,vocab,trainer)
                print "Accuracy with stopwords:"+str(accuracy)+"%"
          elif choice=='4':
                print "Do you want to enter lmbda,eta and iterations values yes/no"
                yesno=str(raw_input())
                if yesno=="yes":
                   print "eta:"
                   eta=float(raw_input())
                   print "lmbda:"
                   lmbda=float(raw_input())
                   print "Iterations:"
                   iterations=int(raw_input())
                   trainer=LR.train(train_ham_folder,train_spam_folder,vocab_stop,eta,lmbda,iterations)
                   accuracy=LR.classify(test_ham_folder,test_spam_folder,vocab_stop,trainer)
                   print "Accuracy without stopwords:"+str(accuracy)+"%"
                else:
                   eta=0.001
                   lmbda=0.075
                   iterations=12
                   trainer=LR.train(train_ham_folder,train_spam_folder,vocab_stop,eta,lmbda,iterations)
                   accuracy=LR.classify(test_ham_folder,test_spam_folder,vocab_stop,trainer)
                   print "Accuracy without stopwords:"+str(accuracy)+"%"
if __name__=='__main__':
     main()  

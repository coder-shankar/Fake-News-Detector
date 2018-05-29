import os
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
import numpy as np
import math
import pickle
from pathlib import Path
 
 


class NaiveBayes(object):
    
    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0 #0.0 is defalult value in no is available
        return word_counts
    
    
    
    def fit(self,X,Y):
        
#         to store the prior probability 
        self.log_class_priors = {}
        self.words_count = {}
        voc = set()
        self.vocabulary = set()
        file = Path('./voc.pickle')
#         if(file.is_file() and (os.path.getsize(file)>0)):
#             with open(file,'rb') as fp:
#                 print("file opening")
#                 voc = pickle.load(fp)
            
#             self.vocabulary = voc
        
        
        
        n = len(X)
        #class prior probability
        
        
        self.log_class_priors['true'] = math.log(sum(1 for label in Y if label == 1) / n)
        self.log_class_priors['fake'] =math.log(sum(1 for label in Y if label == 0)/n)
        
        self.words_count['true'] = {}
        self.words_count['fake'] = {}
        
        
        for x,y in zip(X,Y):
            
            c = 'true' if y == 1 else 'fake'
            
            counts = self.get_word_counts(word_tokenize(str(x)))
            
            for word,count in counts.items():
                if word not in self.vocabulary:
                    self.vocabulary.add(word)
                
                if word not in self.words_count[c]:
                    self.words_count[c][word]=0.0
                    
                self.words_count[c][word]+=count
                
            
        print(self.words_count)
            
        with open ('./voc.pickle','wb') as fp:
            print("writing to voc.pickle")
            pickle.dump(self.vocabulary,fp,protocol = pickle.HIGHEST_PROTOCOL)
            
        with open ('./prior.pickle','wb') as fp:
            print("writing to prior.pickle")
            pickle.dump(self.log_class_priors,fp,protocol = pickle.HIGHEST_PROTOCOL)
        
        
        with open ('./count.pickle','wb') as fp:
            print("writing to count.pickle")
            pickle.dump(self.words_count,fp,protocol = pickle.HIGHEST_PROTOCOL)
        
        
            
    
    
    
    def predict(self,X):
        result = []
        for x in X:
            counts = self.get_word_counts(word_tokenize(str(x)))
        
            print(counts)
            fake_prob = 0
            true_prob = 0
        
            for word,count in counts.items():
                if word not in self.vocabulary:
                    continue
                
                log_word_given_true = math.log((self.words_count['true'].get(word,0.0)+1)/((sum(self.words_count['true'].values())) +len(self.vocabulary)))
                log_word_given_false = math.log((self.words_count['fake'].get(word,0.0)+1)/((sum(self.words_count['fake'].values())) +len(self.vocabulary)))
                
                
                fake_prob +=log_word_given_false
                true_prob +=log_word_given_true
                
                
            
            print(self.log_class_priors['fake'])
            fake_prob += self.log_class_priors['fake']
            true_prob += self.log_class_priors['true']
            
            print("false ")
            print(fake_prob)
            print("true ")
            print(true_prob)
                
            
            print(fake_prob)
            print(true_prob)
            if true_prob > fake_prob:
                result.append(1)
            else:
                result.append(0)
            
            
        print (result)
        return result
                
                    
                    
                
if __name__ == '__main__':

    df = pd.read_csv("../input/train.csv")
    print(df.shape)
    
    df = df.head(200)
    
    
    
    # Make training and test sets
    X_train, x_test, Y_train, y_test = train_test_split(df['text'],df['label'],test_size=0.33)

    
    print(X_train.head())
    print(Y_train.head())

    
    
    nb = NaiveBayes()
    nb.fit(X_train,Y_train)
    pred = nb.predict(x_test)
    type(pred)
    type(y_test)
    
    y_test = y_test.values.tolist()
    accuracy = sum(1 for i in range(len(pred)) if pred[i] == y_test[i]) / float(len(pred))
    print("{0:.4f}".format(accuracy))
    y_pred = pred
    print(pred)
    print(y_test)
    if(y_test== pred):
        print("both are equal")
    
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    plt.clf()
    plt.imshow(cm,interpolation = 'nearest',  cmap=plt.cm.Blues)
    
    ClassNames = ['fake','true']
    plt.ylabel('expected')
    plt.xlabel('predicted')
    
    tick_marks = np.arange(len(ClassNames))
    plt.xticks(tick_marks,ClassNames,rotation = 45)
    plt.yticks(tick_marks,ClassNames)
    plt.imshow(cm,interpolation = 'nearest',  cmap=plt.cm.Blues)
    s = [['TN','FP'],['FN','TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i,str(s[i][j]) +" = " +str(cm[i][j]))

    plt.show()

    
    
    

import numpy as np
from tfidf import TfidfVectorizer
import pandas as  pd
from sklearn.model_selection import train_test_split

class Passive:

    def fit(self,X_train,Y_train):
        C = 0.001
        self.w = np.zeros((features,1))

        for i in range(len(X_train)):

         xi = np.asmatrix(X_train.iloc[i]).reshape(features,1)

         val  = 1 - (Y_train[i] * (np.dot(self.w.T, xi)))



         loss = max(0, val[0])



         denom = sum(x*x for x in xi) +(1/(2*C))
         tau = loss/denom



         coefficient = float(tau * Y_train[i])
         self.w += coefficient *xi
         print(self.w.shape)






    def predict(self, x_test,y_test):

        count =0
        for i in range(len(x_test)):
            xi = np.asmatrix(X_train.iloc[i]).reshape(features,1)
            pred = np.sign(np.dot(self.w.T,xi))
            print(y_test)
            if(pred - y_test[i]==0):
                count  += 1



        acc = count/len(y_test)
        print(acc)



if __name__ == '__main__':

        df = pd.read_csv("./data/Train.csv")
        df = df.head(2)

        tf = TfidfVectorizer(df['text'])
        tfMatrix = tf.fit()
        df['text'] = tfMatrix

        features = len(tfMatrix[0])


        X_train, x_test, Y_train, y_test = train_test_split(df['text'],df['label'],test_size=0.5)



        Y_train[Y_train==0] = -1
        y_test[y_test==0] = -1

        Y_train = Y_train.values.tolist()

        pa =  Passive()
        pa.fit(X_train,Y_train)
        pa.predict(x_test,y_test)


        print(" program  completed ")

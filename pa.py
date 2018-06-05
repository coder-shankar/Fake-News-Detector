#this is implementation of passive aggressive classsifier
import numpy as np
from tfidf import TfidfVectorizer
import pandas as  pd;
from sklearn.model_selection import train_test_split






def main():

    df = pd.read_csv("./data/Train.csv")
    print(df.shape)

    df = df.head(3)
    //tfidf of total training and testing data set



    dataFrame = pd.DataFrame(index= , columns= )





    # Make training and test sets
    X_train, x_test, Y_train, y_test = train_test_split(df['text'],df['label'],test_size=0.5)




    Y_train[Y_train==0] = -1
    y_test[y_test==0] = -1

    Y_train = Y_train.values.tolist()



    tf = TfidfVectorizer(X_train)
    tfMatrix = tf.fit()
    tfMatrix = np.asmatrix(tfMatrix)
    print(tfMatrix)
    features = tfMatrix.shape[1]




    print (tfMatrix.shape)

    C = 0.001
    w = np.zeros((features,1))
    print (w)


    for i in range(tfMatrix.shape[0]):
         print( "value of i is "+ str(i))
         xi = tfMatrix[i].reshape((features,1))



         label = Y_train[i]
         val  = 1 - (label * (np.dot(w.T,xi)))
         print(type(val))



         loss = max(0, val)
         print(loss)


         denom = sum(x*x for x in xi) +(1/(2*C))
         tau = loss/denom

         print (tau)

         coefficient = tau * label
         print(coefficient)
         w += int(coefficient) *xi





    tf = TfidfVectorizer(x_test)
    tfMatrix = tf.fit()

    # tfMatrix = np.asmatrix(tfMatrix)
    pred = np.sign(np.dot(w,x_test.T))
    c = np.count_nonzero(pred - y_test)
    print('PA accuracy: {}'.format(1 - float(c) / x_test.shape[0]))







if __name__ == '__main__':
    main()

import math
import numpy as np;
from nltk import word_tokenize,sent_tokenize


class TfidfVectorizer:

    #constructor
    def __init__(self,documents):
        self.documents = documents
        self.vocabulary = set()
        self.doc_term_matrix = []
        self.doc_term_tfidf_matrix = []




    def bulid_vocabulary(self):
        # set has unique element so we should use set here to store our vocabulary
        voc = set()
        # tokenizing each document using nltk word_tokenize() function and adding to vocabulary
        for doc in self.documents:
            voc.update([word for word in word_tokenize(doc)])

        self.vocabulary = voc



    # tf means term frequency
    # that means count of how many times term is repeated in a perticular document
    @staticmethod
    def tf(term, document):
     return TfidfVectorizer.freq(term, document)


    # frequency function actually count the number of term in a document
    @staticmethod
    def freq(term, document):
        return word_tokenize(document).count(term)


    def build_doc_term_matrix(self):

        for doc in self.documents:
            tf_vector = [TfidfVectorizer.tf(word, doc) for word in self.vocabulary]
            self.doc_term_matrix.append(tf_vector)


        return self.doc_term_matrix



    def doc_term_matrix_norm(self,doc_term_matrix):
        # normalize document term matrix
        doc_term_matrix_norm = []

        for vec in doc_term_matrix:
            doc_term_matrix_norm.append(TfidfVectorizer.normalization(vec))


        return doc_term_matrix_norm


    # normalization is done
    # that means making unit vector
    # we know the formula for unit vector V =(a,b,c)
    # v/square root of (a^2 +b^2 + c^2)
    @staticmethod
    def normalization(vec):
        denom = np.sum([e1 ** 2 for e1 in vec])
        return [e1 / math.sqrt(denom) for e1 in vec]



    # df document frequecy
    @staticmethod
    def doc_freq(word, documents):
        doccount = 0
        # iterating through all the documents
        for doc in documents:
            # if the word is in  document the we increment the counter as the doucment contain the word
            if (TfidfVectorizer.freq(word, doc) > 0):
                doccount += 1
        return doccount

 # now defining the inverse document frequency
    # we know the formula
    # idf = log(1+nd/df)  --here 1 is added because if nd and df are same then the result will be o so
    @staticmethod
    def idf(word, documents):

        nd = len(documents)

        df = TfidfVectorizer.doc_freq(word, documents)

        return np.log((nd / df) + 1)



    # in order to multiply the tf and idf we have to make hot vecor into spare matrix
    @staticmethod
    def build_idf_matrix(idf_vector):
        # making matrix with all zeros
        idf_matrix = np.zeros((len(idf_vector), len(idf_vector)))
        # now filling the diagonal with hot vector
        # the perfect matrix is build
        np.fill_diagonal(idf_matrix, idf_vector)
        return idf_matrix



    def build_tfidf_matrix(self):
        # for each word in the vocabulary
        # calculating the idf value and putting in the vecor
        # this is hot vecor that means all the words idf is placed in it

        idf_vector = [TfidfVectorizer.idf(word, documents) for word in self.vocabulary]

        doc_term_matrix_tfidf = []
        idf_matrix = TfidfVectorizer.build_idf_matrix(idf_vector)
        # performing the matrix multiplicaiton
        for tf_vector in self.doc_term_matrix:
            doc_term_matrix_tfidf.append(np.dot(tf_vector, idf_matrix))

        self.doc_term_tfidf_matrix = doc_term_matrix_tfidf

        return self.doc_term_tfidf_matrix

    def fit(self):
        self.bulid_vocabulary()
        doc_term = self.build_doc_term_matrix()
        doc_term_norm = self.doc_term_matrix_norm(doc_term)
        print(doc_term_norm)
        doc_tfidf = self.build_tfidf_matrix()
        doc_tfidf_norm = self.doc_term_matrix_norm(doc_tfidf)
        print(np.asmatrix(doc_tfidf_norm).shape)

        return doc_tfidf_norm








if __name__ == '__main__':
    document_0 = "The sun is shining"
    document_1 = "The weather is sweet"
    document_2 = "The sun is not shining and the weather is  not sweet"
    document_3 = "The sun is shining"

    # making the list of the documents
    documents = [document_0, document_1, document_2, document_3]
    # making all the documents in lower case bcz basically 'the' and 'The' means same in tfidf
    documents = np.array([document.lower() for document in documents])

    # text document
    document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
    document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
    document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the aleconomic turmoil in his own country for his view on the future of his people."
    document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
    document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
    document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
    document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"


    documents = [document_0,document_1,document_2,document_3,document_4,document_5,document_6]

    # import pandas as pd
    # from sklearn.model_selection import train_test_split
    # from nltk import sent_tokenize
    #
    # df = pd.read_csv("data.csv")
    # print(df.shape)
    #
    # y = df.label
    #
    # # Drop the `label` column
    # df.drop("label", axis=1)  # where numbering of news article is done that column is dropped in dataset
    #
    # # Make training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.99, random_state=53)
    #
    # documents = [data for data in X_train]
    #
    #
    #
    documents = np.array([document.lower() for document in documents])

    tfidf = TfidfVectorizer(documents)

    vector = tfidf.fit()
    print(np.asmatrix(vector))




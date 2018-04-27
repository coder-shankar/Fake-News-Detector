import math

import numpy as np;
from nltk import word_tokenize,sent_tokenize



#things need to add
# visualizaton
#data in more usable format
#more structured program
#remove stop words





#for testing purpose
# text document
# document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
# document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
# document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the aleconomic turmoil in his own country for his view on the future of his people."
# document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
# document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
# document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
# document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"
#
#
# documents = [document_0,document_1,document_2,document_3,document_4,document_5,document_6]
# documents = np.array([document.lower() for document in documents])


#small document for testing purpose

document_0 = "The sun is shining"
document_1 = "The weather is sweet"
document_2 = "The sun is shining and the weather is sweet"

documents = [document_0,document_1,document_2]
documents = np.array([document.lower() for document in documents])







from collections import Counter
for doc in documents:
    tf = Counter()
    for word in word_tokenize(doc):
        tf[word]+=1
    print(tf.items())





def bulid_lexicon(corpus):
    lexicon = set()

    for doc in corpus:
        lexicon.update([word for word in word_tokenize(doc)])

    return lexicon


vocabulary = bulid_lexicon(documents)

print(vocabulary)

def tf(term, document):
  return freq(term, document)

def freq(term, document):
  return word_tokenize(document).count(term)






doc_term_matrix = []

for doc in documents:
    tf_vector = [tf(word,doc ) for word in vocabulary]
    doc_term_matrix.append(tf_vector)

print(doc_term_matrix)




def normalization (vec):

    denom = np.sum([  e1**2  for e1 in vec])
    return [e1/math.sqrt(denom) for e1 in vec]



doc_term_matrix_norm =  []

for vec in doc_term_matrix:
    doc_term_matrix_norm.append(normalization(vec))

print(np.matrix(doc_term_matrix))
print(np.matrix(doc_term_matrix_norm))



# idf frequency weighting
#

def numDocsCount(word,documents):
    doccount = 0

    for doc in documents:
        if(freq(word,doc)>0):
            doccount +=1
    return doccount


def idf (word,documents):

    nd = len(documents)

    df = numDocsCount(word,documents)

    return np.log((nd/df)+1)


my_idf_vector = [idf(word,documents) for word in vocabulary]

def build_idf_matrix(idf_vector):
    idf_matrix = np.zeros((len(idf_vector),len(idf_vector)))

    np.fill_diagonal(idf_matrix,idf_vector)
    return idf_matrix



my_idf_matrix = build_idf_matrix(my_idf_vector)


print(np.matrix(my_idf_matrix))



doc_term_matrix_tfidf = []

#performing the matrix multiplicaiton

for tf_vector in doc_term_matrix:
    doc_term_matrix_tfidf.append(np.dot(tf_vector,my_idf_matrix))

#normalizing

doc_term_matrix_tfidf_l2 = []

for tf_vector in doc_term_matrix_tfidf:
    doc_term_matrix_tfidf_l2.append(normalization(tf_vector))


print(vocabulary)
print(np.matrix(doc_term_matrix_tfidf_l2))

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import pickle #a module for saving anything like any function,class, model,trained model any thing in python
import random
from nltk.corpus import movie_reviews


# In[20]:


from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC


# In[21]:



#making own voting classifier
from nltk.classify import ClassifierI
from statistics import mode
 
class VoteClassifier(ClassifierI):  #inheritence from cclassifier1 class
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# In[3]:


movie_reviews.categories()


# In[4]:


movie_reviews.fileids()


# In[5]:


movie_reviews.words(movie_reviews.fileids("pos")[0])


# In[6]:


len(movie_reviews.words())


# In[7]:


documents=[]
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)),category))


# In[8]:


documents


# In[9]:


random.shuffle(documents)

all_words = []
for i in movie_reviews.words():
    all_words.append(i.lower())
    
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(10))
print(all_words["good"])


# In[25]:


word_features = list(all_words.keys())[:30000]


# In[26]:


def find_features(document):    #for checking wheather each word in document is in word features or not
    words = set(document)
    features = {}
    for w in words:
        features[w] = (w in word_features)
    return features
        


# In[27]:


find_features(movie_reviews.words('neg/cv000_29416.txt'))


# In[28]:


feature_sets = []
for doc in documents:
    appending_tuple = (find_features(doc[0]),doc[1]) 
    feature_sets.append(appending_tuple)


# In[29]:


#training set
train_set = feature_sets[:1900]
#test set
test_set = feature_sets[1900:]


# In[30]:


#training classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[31]:


print("accuracy percentage : ",nltk.classify.accuracy(classifier,test_set)*100)


# In[32]:


classifier.show_most_informative_features(1000)


# In[33]:


# #using pickle
# #saving the classifier
# save_classifier = open("nltk_naviebayes_lassifier.pickle","wb")  #wb means write bytes for saving
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

#for using the classifier
#classifier_f = open("nltk_naviebayes_lassifier.pickle","rb")  #rb for reading classifier
#classifier = pickle.load(classifier_f)
#classifier_f.close()
#now the classsifier is the trained one


# In[34]:


MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_classifier.train(train_set)
print("MultinomialNB_classifier accuracy percentage : ",nltk.classify.accuracy(MultinomialNB_classifier,test_set)*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(train_set)
print("BernoulliNB_classifier accuracy percentage : ",nltk.classify.accuracy(BernoulliNB_classifier,test_set)*100)


# In[35]:


LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(train_set)
print("LogisticRegression_classifier accuracy percentage : ",nltk.classify.accuracy(LogisticRegression_classifier,test_set)*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(train_set)
print("SGDClassifier_classifier accuracy percentage : ",nltk.classify.accuracy(SGDClassifier_classifier,test_set)*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(train_set)
print("SVC_classifier accuracy percentage : ",nltk.classify.accuracy(SVC_classifier,test_set)*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(train_set)
print("NuSVC_classifier accuracy percentage : ",nltk.classify.accuracy(NuSVC_classifier,test_set)*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_set)
print("LinearSVC_classifier accuracy percentage : ",nltk.classify.accuracy(LinearSVC_classifier,test_set)*100)


# In[36]:


voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MultinomialNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, test_set))*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





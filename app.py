from flask import Flask, request, render_template
import numpy as np
from gensim.models import word2vec
import nltk.data
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
import pickle
from bs4 import BeautifulSoup 
import re

app = Flask(__name__)

def review_wordlist(review, remove_stopwords=False):# preprocessing
    
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))     
        words = [w for w in words if not w in stops]
    return(words)

def featureVecMethod(words, model):
    featureVec = np.zeros(300,dtype="float32")
    nwords = 0   
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word]) # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    
    return featureVec

def getAvgFeatureVecs(reviews, model):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),300),dtype="float32")
    for review in reviews:    
        reviewFeatureVecs[counter] = featureVecMethod(review, model,300)
        counter = counter+1
        
    return reviewFeatureVecs
    
def Predict(text):
    test_reviews = []
    model = Word2Vec.load('TP/Model')  #load word2Vec的model   
    test_reviews.append(text)
    testDataVecs = getAvgFeatureVecs(test_reviews, model,300)  #to Vec       
    with open('TP/model.pickle','rb') as f:  #load random forest model
        forest = pickle.load(f)
        result = forest.predict(testDataVecs)
        if result == 1:
            return render_template('index.html', result = 'Positive')
        elif result == 0
            return render_template('index.html', result = 'Neutral')
        elif result == -1:
            return render_template('index.html', result = 'Negative')
            

    
@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		details = request.form
                                if details['form_type'] == 'get_sentiment':
		    return Predict(details['sentence'])

if __name__ == '__main__':
	app.run(host = '0.0.0.0')

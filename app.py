from flask import Flask, request, render_template
import numpy as np
from gensim.models import word2vec
import nltk.data
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
import pickle
from gevent import pywsgi

app = Flask(__name__)


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
    model = Word2Vec.load('写模型路径/Model') #加载word2Vec的   
    test_reviews.append(review_wordlist(text,remove_stopwords=True))
    testDataVecs = getAvgFeatureVecs(test_reviews, model,300)  #to Vec       
    with open('写模型路径/model.pickle','rb') as f:  #加载随机森林的
        forest = pickle.load(f)
        result = forest.predict(testDataVecs)
    return result
    
@app.route('/', methods=['GET', 'POST'])

def index():
	if request.method == 'POST':
		details = request.form
		return Predict(details['data'])

	return render_template('index.html', result = '')    
	
if __name__ == '__main__':
    #app.run(host='0.0.0.0')
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()
    app.run()
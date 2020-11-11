from flask import Flask, request, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

def get_sentiment(text):
	analyser = SentimentIntensityAnalyzer()

	pos = analyser.polarity_scores(text)['pos']
	neg = analyser.polarity_scores(text)['neg']
	neu = analyser.polarity_scores(text)['neu']

	if max(pos,neg,neu) == pos:
		return render_template('index.html', result = 'Positive')
	elif max(pos,neg,neu) == neg:
		return render_template('index.html', result = 'Negative')
	elif max(pos,neg,neu) == neu:
		return render_template('index.html', result = 'Neutral')
	
	
	
@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		details = request.form
		if details['form_type'] == 'get_sentiment':
			return get_sentiment(details['sentence'])
			
	return render_template("index.html", result = '')pos

if __name__ == '__main__':
	app.run(host = '0.0.0.0')

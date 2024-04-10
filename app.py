from flask import Flask, request
from flask_restful import Resource, Api
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)
api = Api(app)
sia = SentimentIntensityAnalyzer()

class SentimentAnalysis(Resource):
    def post(self):
        reviews_data = request.get_json()

        grouped_reviews = {}
        for review in reviews_data:
            text = review.get('text', '')
            sentiment_scores = sia.polarity_scores(text)
            length = len(text.split())

            if sentiment_scores['compound'] > 0.1 and length >= 5:
                group, buy = 'Positive', True
            elif sentiment_scores['compound'] < -0.1 and length >= 5:
                group, buy = 'Negative', False
            elif length < 5:
                group, buy = 'Artificial', False
            else:
                group, buy = 'Neutral', False

            if group not in grouped_reviews:
                grouped_reviews[group] = []
            grouped_reviews[group].append({'text': text, 'sentiment': group, 'buy': buy})

        positive_reviews = [review for review in grouped_reviews.get('Positive', []) if review['buy']]
        positive_percentage = len(positive_reviews) / len(reviews_data) * 100

        overall_recommendation = (
            'Based on the overwhelmingly positive reviews, I highly recommend purchasing this product.'
            if positive_percentage >= 50
            else 'While there are some negative reviews, I\'d suggest taking a closer look at the overall feedback. If the product meets your specific needs, it could still be a worthwhile purchase despite a few negative reviews.'
        )

        return {
            'grouped_reviews': grouped_reviews,
            'positive_percentage': positive_percentage,
            'overall_recommendation': overall_recommendation
        }

@app.route('/')
def index():
    return 'Hello, World!'

api.add_resource(SentimentAnalysis, '/sentimentalAnalysis')

if __name__ == '__main__':
    app.run(debug=True)

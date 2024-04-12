from flask import Flask, request, jsonify
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

        grouped_reviews = {
            "Positive": [],
            "Negative": [],
            "Neutral": [],
            "Artificial": []
        }
        total_rating = 0
        rating_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for review in reviews_data:
            text = review.get('comment', '')
            rating = review.get('rating', 0)
            date = review.get('date', '')
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

            grouped_reviews[group].append({
                'user': review.get('user', ''),
                'text': text,
                'sentiment': group,
                'buy': buy,
                'rating': rating,
                'date': date
            })

            total_rating += rating
            rating_counts[rating] += 1

        positive_reviews = [review for review in grouped_reviews.get('Positive', []) if review['buy']]
        positive_percentage = len(positive_reviews) / len(reviews_data) * 100

        average_rating = total_rating / len(reviews_data) if len(reviews_data) > 0 else 0
        rating_percentages = {str(rating): count / sum(rating_counts.values()) * 100 for rating, count in rating_counts.items()}

        overall_recommendation = (
            'Based on the overwhelmingly positive reviews, I highly recommend purchasing this product.'
            if positive_percentage >= 50
            else 'While there are some negative reviews, I\'d suggest taking a closer look at the overall feedback. If the product meets your specific needs, it could still be a worthwhile purchase despite a few negative reviews.'
        )

        output = {
            "grouped_reviews": grouped_reviews,
            "positive_percentage": positive_percentage,
            "average_rating": average_rating,
            "rating_percentages": rating_percentages,
            "overall_recommendation": overall_recommendation
        }

        return jsonify(output)

@app.route('/')
def index():
    return 'Hello, World!'

api.add_resource(SentimentAnalysis, '/sentimentalAnalysis')

if __name__ == '__main__':
    app.run(debug=True)

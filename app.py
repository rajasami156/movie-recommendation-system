from flask import Flask, request, jsonify
from ML_model import recommend_movies
from simple_model import get_recommendations

app = Flask(__name__)


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_id = data.get('movie_id')
    user_id = data.get('user_id')

    if movie_id is None or user_id is None:
        return jsonify({"error": "Movie ID or User ID is missing."}), 400

    recommended_movies = recommend_movies(int(movie_id))

    return jsonify({"user_id": user_id, "movie_id": movie_id, "recommended_movies": recommended_movies})


@app.route('/recommend_by_genre', methods=['POST'])
def recommend_genres():
    try:
        # Extract genres from the request
        data = request.json
        user_genres = data.get('genres', [])
        recommendations = get_recommendations(user_genres)
        return jsonify(recommendations)
    except Exception as e:
        print(f'Error in API endpoint: {e}')
        return jsonify([]), 500


if __name__ == '__main__':
    app.run(debug=True)
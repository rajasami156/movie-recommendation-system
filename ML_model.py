import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
movies = pd.read_csv('new_dataset.csv')  # Replace with your actual file name

# Fill missing values and combine relevant columns for similarity
movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('')
movies['keywords'] = movies['keywords'].fillna('')
movies['combined_features'] = movies['overview'] + ' ' + movies['genres'] + ' ' + movies['keywords']

# Vectorize the combined text features
vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(movies['combined_features'])

# Compute cosine similarity between all movie pairs
movie_similarity = cosine_similarity(feature_matrix)

# Convert to DataFrame for easier manipulation
movie_similarity_df = pd.DataFrame(movie_similarity, index=movies['id'], columns=movies['id'])

def recommend_movies(movie_id, num_recommendations=7):
    if movie_id not in movie_similarity_df.index:
        return "Movie ID not found."
    
    # Get movies sorted by similarity scores
    similar_movies = movie_similarity_df[movie_id].sort_values(ascending=False)
    
    # Exclude the movie itself and get the top N recommendations
    top_recommendations = similar_movies.iloc[1:num_recommendations+1]
    
    # Map indices to movie titles
    recommended_movie_ids = top_recommendations.index
    recommended_movies = movies[movies['id'].isin(recommended_movie_ids)]['title']
    
    return recommended_movies.tolist()
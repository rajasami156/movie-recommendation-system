import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()  # This loads environment variables from .env file

# Establish connection to MongoDB
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
database = client.Movie_Recommendation
collection = database.CSVData

# Fetch data and convert to DataFrame
data = list(collection.find({}))
movies = pd.DataFrame(data)
movies.drop(columns=['_id'], inplace=True)  # Optional: remove MongoDB's auto-generated ID

# Close the MongoDB connection after fetching data
client.close()

# Ensure non-null values for text processing
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
movie_similarity_df = pd.DataFrame(movie_similarity, index=movies['id'].astype(str), columns=movies['id'].astype(str))

def recommend_movies(movie_id, num_recommendations=7):
    # Convert movie_id to string if it's numeric
    movie_id = str(movie_id)
    if movie_id not in movie_similarity_df.index:
        return "Movie ID not found."
    
    # Get movies sorted by similarity scores
    similar_movies = movie_similarity_df.loc[movie_id].sort_values(ascending=False)
    
    # Exclude the movie itself and get the top N recommendations
    top_recommendations = similar_movies.iloc[1:num_recommendations+1]
    
    # Map indices to movie titles
    recommended_movie_ids = top_recommendations.index
    recommended_movies = movies[movies['id'].astype(str).isin(recommended_movie_ids)]['title']
    
    return recommended_movies.tolist()

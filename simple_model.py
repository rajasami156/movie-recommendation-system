import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()  # This loads environment variables from .env file

# Function for weighted rating calculation
def weighted_rating(x, m, c):
    v = x['vote_count']
    r = x['vote_average']
    return (v / (v + m) * r) + (m / (m + v) * c)

# Load movie dataset function from MongoDB
def load_movies():
    try:
        # Establish connection to MongoDB
        mongo_uri = os.getenv("MONGO_URI")
        client = MongoClient(mongo_uri)
        database = client.Movie_Recommendation
        collection = database.CSVData

        # Fetch data from MongoDB and convert to DataFrame
        data = list(collection.find({}))
        if not data:
            raise ValueError("No data found in the MongoDB collection")

        movies = pd.DataFrame(data)
        movies.drop(columns=['_id'], inplace=True)  # Remove MongoDB's auto-generated ID
        
        # Close the MongoDB connection after fetching data
        client.close()

        # Process genres column into lists
        if 'genres' not in movies.columns:
            raise ValueError("Expected 'genres' field is missing from the MongoDB collection")

        movies['genre_names'] = movies['genres'].str.split(", ")
        return movies
    except Exception as e:
        print(f"Error loading dataset from MongoDB: {e}")
        return pd.DataFrame()

# Function to get recommendations based on genres
def get_recommendations(user_genres):
    try:
        movies = load_movies()
        if movies.empty:
            return []

        # Calculate mean vote and 95th percentile vote count
        c = movies['vote_average'].mean()
        m = movies['vote_count'].quantile(0.95)

        # Filter movies with a vote count above the 95th percentile
        qualified = movies[movies['vote_count'] >= m].copy()
        qualified['vote_count'] = qualified['vote_count'].astype(int)
        qualified['vote_average'] = qualified['vote_average'].astype(int)
        qualified['wr'] = qualified.apply(weighted_rating, axis=1, args=(m, c))
        qualified = qualified.sort_values('wr', ascending=False)

        # Flatten genres to facilitate filtering
        s = movies['genre_names'].explode().reset_index()
        s.rename(columns={'genre_names': 'genre'}, inplace=True)
        gen_md = movies.join(s.set_index('index'), rsuffix='_exploded')

        # Return top 10 recommended movies based on the specified genres
        recommended_movies = gen_md[gen_md['genre'].isin(user_genres)].head(10)
        return recommended_movies.to_dict(orient='records')
    except Exception as e:
        print(f'Error in get_recommendations: {e}')
        return []

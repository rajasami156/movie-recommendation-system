from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
from ML_model import recommend_movies
from simple_model import get_recommendations

app = FastAPI()
from dotenv import load_dotenv
load_dotenv()  

# Pydantic models for request data validation
class RecommendationRequest(BaseModel):
    movie_id: int
    user_id: int

class GenreRecommendationRequest(BaseModel):
    genres: List[str]

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post('/recommend')
async def recommend(data: RecommendationRequest):
    try:
        # Get movie recommendations
        recommended_movies = recommend_movies(data.movie_id)
        return {
            "user_id": data.user_id,
            "movie_id": data.movie_id,
            "recommended_movies": recommended_movies
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post('/recommend_by_genre')
async def recommend_genres(data: GenreRecommendationRequest):
    try:
        # Get genre recommendations
        recommendations = get_recommendations(data.genres)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Run the application using Uvicorn (or another ASGI server)
import os
if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if not specified
    uvicorn.run(app, host='0.0.0.0', port=port)

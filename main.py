from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os


sys.path.append(os.path.dirname(__file__))
import newmodel 

app = FastAPI()


origins = [
    "http://localhost:3000", 
    "http://localhost:5173", 
    "https://your-moodmate-frontend.netlify.app", # Replace with your actual Netlify/Vercel frontend URL
 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
class Features(BaseModel):
   
    data: list[float] 


@app.post("/predict")
async def predict_treatment(features: Features):
    """
    API endpoint to predict if mental health treatment is needed.
    Expects a JSON body with a 'data' key containing a list of features.
    """
    try:
        prediction = newmodel.predict_mental_health(features.data)
        return {"prediction": prediction}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input features: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/")
async def root():
    """Basic root endpoint to check if the API is running."""
    return {"message": "MoodMate ML API is running!"}

@app.on_event("startup")
async def startup_event():
    newmodel.load_model()
    if newmodel._model is None:
        print("Warning: Model could not be loaded at startup.")
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
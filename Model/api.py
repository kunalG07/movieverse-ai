from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI()

# Enable CORS (frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# FILE PATH FIX (IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
encoder_path = os.path.join(BASE_DIR, "encoder.pkl")
data_path = os.path.join(BASE_DIR, "final_ml_ready_dataset.csv")

# -----------------------------
# LOAD FILES
# -----------------------------
model = joblib.load(model_path)
le = joblib.load(encoder_path)
df = pd.read_csv(data_path)

print("✅ Files Loaded Successfully")

# -----------------------------
# CREATE LOOKUP TABLES
# -----------------------------

# Director success
director_success = df.groupby('Director')['Success_Index'].mean().to_dict()

# Actor success
actor_df = df.copy()
actor_df['Actors'] = actor_df['Actors'].str.split(',')
actor_df = actor_df.explode('Actors')
actor_df['Actors'] = actor_df['Actors'].str.strip()
actor_success = actor_df.groupby('Actors')['Success_Index'].mean().to_dict()

# Genre success
genre_success = df.groupby('main_genre')['Success_Index'].mean().to_dict()

# Default fallback values
default_director = np.mean(list(director_success.values()))
default_actor = np.mean(list(actor_success.values()))
default_genre = np.mean(list(genre_success.values()))

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def convert_input(director, actors, genre):
    
    # Director
    d_rate = director_success.get(director, default_director)
    
    # Actors
    actor_list = actors.split(',')
    a_rates = [actor_success.get(a.strip(), default_actor) for a in actor_list]
    a_rate = np.mean(a_rates)
    
    # Genre (multiple)
    genre_list = genre.split(',')
    g_rates = [genre_success.get(g.strip(), default_genre) for g in genre_list]
    g_rate = np.mean(g_rates)

    return pd.DataFrame([{
        'Director_Success_Rate': d_rate,
        'Actor_Success_Rate': a_rate,
        'Genre_Success_Rate': g_rate
    }])

# -----------------------------
# API ROUTE
# -----------------------------
@app.post("/predict")
def predict(data: dict):

    try:
        director = data["director"]
        actors = data["actors"]
        genre = data["genre"]
        budget = float(data["budget"])

        # Convert input
        input_df = convert_input(director, actors, genre)

        # Prediction
        pred = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]

        label = le.inverse_transform([pred])[0]
        confidence = max(probs)

        # -----------------------------
        # Budget Logic (Balanced)
        # -----------------------------
        if budget > 20000000000:       # very high budget
            confidence += 0.05
        elif budget < 500000000:       # very low budget
            confidence -= 0.05

        confidence = max(0, min(confidence, 1))

        return {
            "Prediction": label,
            "Confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return {
            "error": str(e)
        }

# -----------------------------
# ROOT CHECK
# -----------------------------
@app.get("/")
def home():
    return {"message": "🎬 Movie Success Prediction API is running"}
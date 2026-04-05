import pandas as pd
import numpy as np
import joblib

# LOAD TRAINED MODEL & ENCODER
model = joblib.load('model.pkl')
le = joblib.load('encoder.pkl')


# LOAD DATA (FOR LOOKUP TABLES)

df = pd.read_csv('final_ml_ready_dataset.csv')

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

# INPUT CONVERSION FUNCTION

def convert_input(director, actors, genre):
    
    # Director
    d_rate = director_success.get(
        director, np.mean(list(director_success.values()))
    )
    
    # Actors
    actor_list = actors.split(',')
    rates = [
        actor_success.get(a.strip(), np.mean(list(actor_success.values())))
        for a in actor_list
    ]
    a_rate = np.mean(rates)
    
    # Genre (multi-genre support)
    genres = genre.split(',')
    g_rates = [
        genre_success.get(g.strip(), np.mean(list(genre_success.values())))
        for g in genres
    ]
    g_rate = np.mean(g_rates)
    
    return {
        'Director_Success_Rate': d_rate,
        'Actor_Success_Rate': a_rate,
        'Genre_Success_Rate': g_rate
    }

# PREDICTION FUNCTION
def predict_movie(director, actors, genre, budget):
    
    # Step 1: ML prediction
    input_data = convert_input(director, actors, genre)
    df_input = pd.DataFrame([input_data])
    
    pred = model.predict(df_input)[0]
    probs = model.predict_proba(df_input)[0]
    
    label = le.inverse_transform([pred])[0]
    confidence = max(probs)

    # Step 2: Budget Adjustment
    
    if budget > 20000000000:  # High budget
        confidence += 0.10
        
        if label == "Flop":
            label = "Average"
        elif label == "Average":
            label = "Hit"

    elif budget < 500000000:  # Low budget
        confidence -= 0.10
        
        if label == "Hit":
            label = "Average"
        elif label == "Average":
            label = "Flop"

    # Clamp confidence
    confidence = max(0, min(confidence, 1))

    return {
        "Prediction": label,
        "Confidence (%)": round(confidence * 100, 2)
    }

# TEST INPUT
result = predict_movie(
    director="Aditya Dhar",
    actors="Ranveer Singh, Sara Arjun, Akshay Khanna, Arjun Rampal",
    genre="Action,crime",
    budget=27500000000
)

print(result)
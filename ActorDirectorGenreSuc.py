import pandas as pd
import numpy as np

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv('final_movies_with_success_index.csv')

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
df['Director'] = df['Director'].fillna('Unknown')
df['Actors'] = df['Actors'].fillna('Unknown')
df['main_genre'] = df['main_genre'].fillna('Unknown')

# -----------------------------
# 🎬 DIRECTOR SUCCESS RATE
# -----------------------------
director_success = df.groupby('Director')['Success_Index'].mean()
df['Director_Success_Rate'] = df['Director'].map(director_success)

# -----------------------------
# 🎭 ACTOR SUCCESS RATE (MULTI-CAST)
# -----------------------------
actor_df = df.copy()
actor_df['Actors'] = actor_df['Actors'].str.split(',')
actor_df = actor_df.explode('Actors')
actor_df['Actors'] = actor_df['Actors'].str.strip()

actor_success = actor_df.groupby('Actors')['Success_Index'].mean()

def get_actor_success(actors):
    actors = str(actors).split(',')
    rates = [actor_success.get(a.strip(), 0) for a in actors]
    return np.mean(rates) if rates else 0

df['Actor_Success_Rate'] = df['Actors'].apply(get_actor_success)

# -----------------------------
# 🎥 GENRE SUCCESS RATE
# -----------------------------
genre_success = df.groupby('main_genre')['Success_Index'].mean()
df['Genre_Success_Rate'] = df['main_genre'].map(genre_success)

# -----------------------------
# FINAL CLEAN (NO NULLS)
# -----------------------------
num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = df[num_cols].fillna(0)
df[cat_cols] = df[cat_cols].fillna('Unknown')

# -----------------------------
# SAVE FINAL DATASET
# -----------------------------
df.to_csv('final_ml_ready_dataset.csv', index=False)

print("✅ ALL SUCCESS FEATURES CREATED!")
print(df[['Director_Success_Rate', 'Actor_Success_Rate', 'Genre_Success_Rate']].head())
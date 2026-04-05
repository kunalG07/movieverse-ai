import pandas as pd

# Load cleaned dataset
df = pd.read_csv('clean_box_office_dataset.csv')

# -----------------------------
# SUCCESS SCORE (0 → 1)
# -----------------------------
df['Success_Score'] = df['Box_Office'] / df['Box_Office'].max()

# -----------------------------
# SUCCESS LEVEL
# -----------------------------
median = df['Box_Office'].median()
q75 = df['Box_Office'].quantile(0.75)

def classify(x):
    if x >= q75:
        return 'Blockbuster'
    elif x >= median:
        return 'Average'
    else:
        return 'Flop'

df['Success_Level'] = df['Box_Office'].apply(classify)

# -----------------------------
# LEVEL → WEIGHT
# -----------------------------
level_map = {
    'Flop': 0.3,
    'Average': 0.6,
    'Blockbuster': 1.0
}

df['Level_Weight'] = df['Success_Level'].map(level_map)

# -----------------------------
# FINAL SUCCESS INDEX
# -----------------------------
df['Success_Index'] = df['Success_Score'] * df['Level_Weight']

# -----------------------------
# SAVE FINAL DATASET
# -----------------------------
df.to_csv('final_movies_with_success_index.csv', index=False)

print("✅ Success Index Created Successfully!")
print(df[['Box_Office', 'Success_Score', 'Success_Level', 'Success_Index']].head())
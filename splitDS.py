import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv('final_ml_ready_dataset.csv')

print("Columns in dataset:", df.columns)

# -----------------------------
# CREATE TARGET (3 CLASS)
# -----------------------------
q25 = df['Success_Index'].quantile(0.25)
q75 = df['Success_Index'].quantile(0.75)

def classify(x):
    if x <= q25:
        return 'Flop'
    elif x <= q75:
        return 'Average'
    else:
        return 'Hit'

df['Target'] = df['Success_Index'].apply(classify)

# -----------------------------
# FEATURES (UPDATED)
# -----------------------------
features = [
    'Director_Success_Rate',
    'Actor_Success_Rate',
    'Genre_Success_Rate'
]

X = df[features]
y = df['Target']

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# SAVE DATA
# -----------------------------
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("✅ Dataset split successfully (without Budget & Release_Date)")
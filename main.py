import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. Load Data
# ==========================================
print("Loading data...")
df = pd.read_csv('train.csv')

# ==========================================
# 2. Data Cleaning
# ==========================================
# Median is chosen over mean because it's more robust to outliers (e.g., extreme ages)
df['Age'] = df['Age'].fillna(df['Age'].median())
# Use the most frequent port to avoid dropping rows with missing embarkation data
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# ==========================================
# 3. Feature Engineering (Categorical Encoding)
# ==========================================
# ML models require numeric input. We map categorical strings to integers.
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# ==========================================
# 4. Feature Selection
# ==========================================
# Excluded features like 'Name' and 'Ticket' due to high cardinality/text complexity
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# ==========================================
# 5. Train/Test Split
# ==========================================
# 80/20 split to prevent overfitting. 
# random_state=42 ensures reproducibility so we get the exact same split every run.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 6. Model Training
# ==========================================
print("Training model...")
# Random Forest is chosen for its robustness against overfitting compared to a single Decision Tree.
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 7. Evaluation
# ==========================================
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")


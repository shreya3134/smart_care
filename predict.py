import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load the dataset
df = pd.read_csv('output.data')

# Step 2: Drop columns with all NaN values
df = df.dropna(axis=1, how='all')

# Step 3: Fill missing values with the most frequent value in each column
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = imputer.fit_transform(df)
df_cleaned = pd.DataFrame(df_imputed, columns=df.columns)

# Step 4: Encode the target column (Disease)
disease_encoder = LabelEncoder()
df_cleaned['Disease'] = disease_encoder.fit_transform(df_cleaned['Disease'])

# Step 5: Encode symptom columns
symptom_cols = [f'Symptom_{i}' for i in range(1, 7)]
symptom_encoders = {}

for col in symptom_cols:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    symptom_encoders[col] = le

# Step 6: Prepare features and labels
X = df_cleaned[symptom_cols]
y = df_cleaned['Disease']

# Step 7: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 8: Train the model with class weight balanced (if imbalance exists)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Step 9: Evaluate and print accuracy
y_pred = model.predict(X_test)
print("Model Test Accuracy:", accuracy_score(y_test, y_pred))

# Optional: See class prediction distribution (to check if it's biased)
import numpy as np
unique, counts = np.unique(y_pred, return_counts=True)
print("Prediction Distribution:", dict(zip(disease_encoder.inverse_transform(unique), counts)))

# Step 10: Save the model and encoders
pickle.dump(model, open('pred.pkl', 'wb'))
pickle.dump(symptom_encoders, open('symptom_encoders.pkl', 'wb'))
pickle.dump(disease_encoder, open('disease_encoder.pkl', 'wb'))

print("Model and encoders saved successfully.")

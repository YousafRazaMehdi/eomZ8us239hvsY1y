# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
url = "https://drive.google.com/uc?id=1KWE3J0uU_sFIJnZ74Id3FDBcejELI7FD"
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())

# Separate features (X) and target (y) variables
X = data.drop('Y', axis=1)
y = data['Y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_classifier.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("Feature Importance:")
print(feature_importance)

# Minimal set of features that preserve most information
selected_features = feature_importance[feature_importance['Importance'] > 0]
print("Selected Features:")
print(selected_features)

# Identify questions to remove in the next survey
questions_to_remove = feature_importance[feature_importance['Importance'] == 0]['Feature']
print("Questions to remove in the next survey:")
print(questions_to_remove)
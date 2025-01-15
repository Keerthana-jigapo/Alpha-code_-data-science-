# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal and petal length and width)
y = iris.target  # Target (species)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Mapping target labels to species names
species_names = iris.target_names
print("\nSpecies Mapping: ", dict(enumerate(species_names)))

# Predicting new samples (optional)
new_samples = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5], [7.2, 3.6, 6.1, 2.5]]
predictions = model.predict(new_samples)
print("\nPredicted species for new samples:", [species_names[p] for p in predictions])

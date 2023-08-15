from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

# Load a sample dataset (you can replace this with your own dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Define the hyperparameter grid to search
param_dist = {
    'n_estimators': np.arange(50, 251, 50),  # Random values between 50 and 250
    'max_depth': [None] + list(np.arange(10, 101, 10)),  # Including 'None'
    'min_samples_split': np.arange(2, 11),  # Random values between 2 and 10
    'min_samples_leaf': np.arange(1, 5),  # Random values between 1 and 4
    'max_features': ['auto', 'sqrt', 'log2']
}

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Create a RandomizedSearchCV instance
random_search = RandomizedSearchCV(
    rf_classifier,
    param_distributions=param_dist,
    n_iter=20,  # Number of random combinations to try
    cv=5,  # Number of cross-validation folds
    scoring='accuracy',  # You can use other metrics as well
    n_jobs=-1  # Use all available CPU cores
)

# Perform the randomized search
random_search.fit(X, y)

# Print the best hyperparameters and corresponding score
print("Best Hyperparameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

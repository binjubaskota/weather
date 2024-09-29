# Training Script: train_model.py

#!/usr/bin/env python
# coding: utf-8

# Data manipulation libraries
import pandas as pd
import numpy as np

# Date and time libraries
from datetime import datetime

# Saving and loading models
import dill

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import os

# Custom Decision Tree Regressor
class CustomDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=10, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.tree = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if (depth >= self.max_depth or num_samples < self.min_samples_split):
            leaf_value = y.mean(axis=0)
            return {"type": "leaf", "value": leaf_value}

        feat_idxs = np.random.choice(num_features, self.n_features, replace=False)

        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feat is None:
            leaf_value = y.mean(axis=0)
            return {"type": "leaf", "value": leaf_value}

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = y.mean(axis=0)
            return {"type": "leaf", "value": leaf_value}

        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return {"type": "node", "feature_index": best_feat, "threshold": best_thresh, "left": left, "right": right}

    def _best_split(self, X, y, feat_idxs):
        best_mse = float("inf")
        best_feat, best_thresh = None, None
        for feat in feat_idxs:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_idxs, right_idxs = self._split(X[:, feat], thresh)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                mse = self._calculate_mse(y[left_idxs], y[right_idxs])
                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat
                    best_thresh = thresh
        return best_feat, best_thresh

    def _split(self, X_column, thresh):
        left_idxs = np.argwhere(X_column <= thresh).flatten()
        right_idxs = np.argwhere(X_column > thresh).flatten()
        return left_idxs, right_idxs

    def _calculate_mse(self, left, right):
        if left.shape[0] == 0 or right.shape[0] == 0:
            return float("inf")
        mse_left = np.var(left, axis=0) * left.shape[0]
        mse_right = np.var(right, axis=0) * right.shape[0]
        return mse_left.sum() + mse_right.sum()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node["type"] == "leaf":
            return node["value"]
        if x[node["feature_index"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])

# Custom Random Forest Regressor
class CustomRandomForest:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=10, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        if self.random_state:
            np.random.seed(self.random_state)

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]
            tree = CustomDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self._max_features(X.shape[1])
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _max_features(self, total_features):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(total_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(total_features)))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return total_features

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return tree_preds.mean(axis=0)

# Load the dataset
data = pd.read_csv('dataset/KTM_2001-2022_DATA.csv')
data = data.dropna()

# Assume the first row is the header
new_header = data.iloc[0]
data = data[1:]
data.columns = new_header

# Sample 10% of the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the sampled data
print(f"Sampled data size: {data.shape}")
print(data.head())

# Convert all columns to numeric where possible
for col in data.columns:
    if col not in ['YEAR', 'DOY']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop any rows with NaN values after conversion
data = data.dropna()

# Convert DOY to integer
data['DOY'] = data['DOY'].astype(int)

# Create a Date column
data['Date'] = pd.to_datetime(data['YEAR'], format='%Y') + pd.to_timedelta(data['DOY'] - 1, unit='D')

# Extract Month and Day
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Create a 'DayOfWeek' feature (0=Monday, 6=Sunday)
data['DayOfWeek'] = data['Date'].dt.dayofweek

# Define seasons based on month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

data['Season'] = data['Month'].apply(get_season)

# One-Hot Encode the 'Season' feature
data = pd.get_dummies(data, columns=['Season'], drop_first=True)

# Drop the Date column as it's no longer needed
data.drop(columns=['Date'], inplace=True)

# Check data types
print("Data types:\n", data.dtypes)

# List all columns to ensure no Timestamp columns are present
print("Columns in the dataset:\n", data.columns.tolist())

# Define features and targets
target_columns = ['T2M', 'TS', 'QV2M', 'RH2M', 'WS2M', 'WS10M', 'WD10M',
                 'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_LW_DWN']

features = data.drop(columns=target_columns + ['YEAR', 'DOY'])

targets = data[target_columns]

# Verify the features and targets
print("Features:\n", features.head())
print("\nTargets:\n", targets.head())

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features.values, targets.values, test_size=0.10, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Initialize the Custom Random Forest Regressor with multi-output capability
model = CustomRandomForest(n_estimators=1000, max_depth=5, min_samples_split=10, max_features='sqrt', random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

print("Model training completed.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Initialize a list to store evaluation metrics
metrics_list = []

# Calculate evaluation metrics for each target variable
for idx, target in enumerate(target_columns):
    mae = np.mean(np.abs(y_test[:, idx] - y_pred[:, idx]))
    rmse = np.sqrt(np.mean((y_test[:, idx] - y_pred[:, idx])**2))
    ss_total = np.sum((y_test[:, idx] - np.mean(y_test[:, idx]))**2)
    ss_res = np.sum((y_test[:, idx] - y_pred[:, idx])**2)
    r2 = 1 - (ss_res / ss_total)
    
    # Append the metrics as a dictionary to the list
    metrics_list.append({
        'Target': target,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 2)
    })

# Convert the list of dictionaries to a DataFrame
evaluation_metrics = pd.DataFrame(metrics_list)

# Display the evaluation metrics
print(evaluation_metrics)

# Save the model and metrics using dill
model_dir = 'trained_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_filepath = os.path.join(model_dir, 'final-model.pkl')

with open(model_filepath, 'wb') as file:
    dill.dump({
        'model': model,
        'metrics': evaluation_metrics,
        'feature_names': features.columns.tolist()
    }, file)

print(f"\nModel saved to {model_filepath}.")




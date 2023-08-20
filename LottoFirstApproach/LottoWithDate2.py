import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# Read the data from the Excel file into a pandas DataFrame
file_path = r"C:\Users\andre\PycharmProjects\Lotto\Repo.xlsx"
data_df = pd.read_excel(file_path, sheet_name="Sheet3", header=None)

# Extract the newest and oldest extraction dates
newest_extraction_date = pd.to_datetime(data_df.iloc[0, 0])
oldest_extraction_date = pd.to_datetime(data_df.iloc[-1, 0])

# Prepare the data for training
data = [data_df.iloc[i:i+6, 1:].values.flatten() for i in range(len(data_df)-5)]
data = np.array(data)

# Prepare the data for training
X = data[:, :-6]  # Input features (all numbers except the last 6 in each group)
y = data[:, -6:]  # Output labels (last 6 numbers in each group)

# Feature Engineering: Add statistical features (mean, median, variance) for each group
additional_features = np.hstack([np.mean(y, axis=1, keepdims=True), np.median(y, axis=1, keepdims=True), np.var(y, axis=1, keepdims=True)])
X = np.hstack([X, additional_features])

# Scale the input features using the MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter Tuning using RandomizedSearchCV with very few iterations
param_dist = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}
model = RandomForestRegressor(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=2, cv=2, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search.fit(X_scaled, y)

best_model = random_search.best_estimator_

# Predict the next 6 numbers based on the last 5 sequences
last_five_sequences_scaled = X_scaled[-5:, :]
next_six_numbers = best_model.predict(last_five_sequences_scaled)

# Estimate probabilities based on relative frequency in predicted outputs
predicted_numbers, counts = np.unique(np.round(next_six_numbers), return_counts=True)
predicted_probabilities = counts / np.sum(counts)

# Sort the numbers by estimated probabilities and print the top 6
top_indices = np.argsort(predicted_probabilities)[::-1][:6]
top_numbers = predicted_numbers[top_indices]
top_probabilities = predicted_probabilities[top_indices]

print("Extraction Pattern Analysis:")
print("Newest Extraction Date:", newest_extraction_date.strftime('%Y-%m-%d'))
print("Oldest Extraction Date:", oldest_extraction_date.strftime('%Y-%m-%d'))
print("The predicted next 6 numbers are:")
print(top_numbers)
print("Probabilities:")
print(top_probabilities)

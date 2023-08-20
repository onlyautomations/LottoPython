import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Read the data from the Excel file into a pandas DataFrame
file_path = r"C:\Users\andre\PycharmProjects\Lotto\Repo.xlsx"
data_df = pd.read_excel(file_path, sheet_name="Sheet3", header=None)

# Extract the newest and oldest extraction dates
newest_extraction_date = data_df.iloc[0, 0]
oldest_extraction_date = data_df.iloc[-1, 0]

# Group the data into sets of six numbers and convert to a 2-dimensional array
data = [data_df.iloc[i:i+6, 1:].values.flatten() for i in range(len(data_df)-5)]
data = np.array(data)

# Prepare the data for training
X = data[:, :-6]  # Input features (all numbers except the last 6 in each group)
y = data[:, -6:]  # Output labels (last 6 numbers in each group)

# Scale the input features using the MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the random forest regressor model with optimized hyperparameters and random state
model = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=2, random_state=42)

# Train the model on the entire dataset
model.fit(X_scaled, y)

# Generate all possible combinations of six numbers between oldest and newest dates
all_combinations = []
for i in range(len(data_df)-5):
    six_numbers = data_df.iloc[i:i+6, 1:].values.flatten()
    all_combinations.append(six_numbers)

# Scale the input features of all combinations
all_combinations_scaled = scaler.transform([comb[:-6] for comb in all_combinations])

# Predict the probabilities for all combinations
all_predicted_probabilities = model.predict(all_combinations_scaled)

# Find the combination with the highest sum of probabilities
best_combination_index = np.argmax(np.sum(all_predicted_probabilities, axis=1))
best_combination = all_combinations[best_combination_index]

print("Newest Extraction Date:", newest_extraction_date)
print("Oldest Extraction Date:", oldest_extraction_date)
print("Best Number Combination:")
print(best_combination)

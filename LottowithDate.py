import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

# Predict the next 6 numbers based on the last 5 sequences
last_five_sequences_scaled = X_scaled[-5:, :]
next_six_numbers = model.predict(last_five_sequences_scaled)

# Estimate probabilities based on relative frequency in predicted outputs
predicted_numbers, counts = np.unique(np.round(next_six_numbers), return_counts=True)
predicted_probabilities = counts / np.sum(counts)

# Sort the numbers by estimated probabilities and print the top 6
top_indices = np.argsort(predicted_probabilities)[::-1][:6]
top_numbers = predicted_numbers[top_indices]
top_probabilities = predicted_probabilities[top_indices]

print("Newest Extraction Date:", newest_extraction_date)
print("Oldest Extraction Date:", oldest_extraction_date)
print("The predicted next 6 numbers are:")
print(top_numbers)
print("Probabilities:")
print(top_probabilities)

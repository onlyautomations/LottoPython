import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data from the Excel file into a pandas DataFrame
file_path = r"C:\Users\andre\PycharmProjects\Lotto\Repo.xlsx"
data_df = pd.read_excel(file_path, sheet_name="Sheet4", header=None)

# Group the data into sets of six numbers and convert to a 2-dimensional array
def create_sequences(data, sequence_length=6):
    sequences = [data[i:i + sequence_length, :].flatten() for i in range(len(data) - sequence_length + 1)]
    return np.array(sequences)

data = create_sequences(data_df.values)

# Prepare the data for training
X = data[:, :-6]  # Input features (all numbers except the last 6 in each group)
y = data[:, -6:]  # Output labels (last 6 numbers in each group)

# Print the total number of rows analyzed
total_rows_analyzed = len(data)
print(f"Total number of rows analyzed: {total_rows_analyzed}")

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale the input features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Initialize the random forest regressor model with optimized hyperparameters
model = RandomForestRegressor(
    n_estimators=1000,  # Increase number of estimators
    max_depth=20,        # Increase max depth
    min_samples_split=4, # Adjust min_samples_split
    random_state=42,
    n_jobs=-1            # Use all available cores for training
)

# Train the model on the training dataset
model.fit(X_train_scaled, y_train)

# Evaluate the model on the validation dataset
y_val_pred = model.predict(X_val_scaled)
validation_mse = mean_squared_error(y_val, y_val_pred)

# Print the last 6 numbers added to the dataset
last_six_numbers = data[-1, -6:]
print("Last 6 numbers added to the dataset:")
print(last_six_numbers)

# Predict the next 6 numbers based on the last 5 sequences in the test set
last_five_sequences_scaled = X_test_scaled[-5:, :]
next_six_numbers = model.predict(last_five_sequences_scaled)

# Estimate probabilities based on relative frequency in predicted outputs
predicted_numbers, counts = np.unique(np.round(next_six_numbers), return_counts=True)
predicted_probabilities = counts / np.sum(counts)

# Sort the numbers by estimated probabilities and print the top 6
top_indices = np.argsort(predicted_probabilities)[::-1]
top_numbers = predicted_numbers[top_indices]
top_probabilities = predicted_probabilities[top_indices]

print("The predicted next 6 numbers are:")
print(top_numbers[:6])
print("Probabilities:")
print(top_probabilities[:6])

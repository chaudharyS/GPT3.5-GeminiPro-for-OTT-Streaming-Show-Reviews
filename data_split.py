import csv
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv('../Rating-Prediction/audience_reviews_processed.csv')

# Spit dataset into training (70%), validation (15%) and test data (15%)
def split_data():
    selected_columns = ["Show", "Rating", "Review"]  # Select specific columns
    dataset_subset = df[selected_columns]  # Select the specified columns from the dataset
    train_data, temp_test_data = train_test_split(dataset_subset, test_size=0.3,
                                                 random_state=42)  # Split the subset dataset into training (70%) and test (30%) sets
    val_data, test_data = train_test_split(temp_test_data, test_size=0.5,
                                                 random_state=42)  # Further split the temp test dataset into validation (50%) and test (50%) sets
    train_data.to_csv('../Rating-Prediction/audience_reviews_train.csv', index=False)  # Save the training, validation and test sets into separate CSV files
    val_data.to_csv('../Rating-Prediction/audience_reviews_val.csv', index=False)  # Save the training, validation and test sets into separate CSV files
    test_data.to_csv('../Rating-Prediction/audience_reviews_test.csv', index=False)  # Save the training, validation and test sets into separate CSV files


split_data()
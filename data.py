import pandas as pd
import numpy as np
import os

# Define the path where your data files are located

# List of file names
train_files = [f'train_FD00{i}.txt' for i in range(1, 5)]
test_files = [f'test_FD00{i}.txt' for i in range(1, 5)]
rul_files = [f'RUL_FD00{i}.txt' for i in range(1, 5)]

def load_dataset(filename):
    return pd.read_csv( filename ,delim_whitespace=True, header=None)

train_dfs = [load_dataset(file) for file in train_files]
test_dfs = [load_dataset(file) for file in test_files]
rul_dfs = [load_dataset(file) for file in rul_files]
# Check the first few rows of one training set

# Assign column names
column_names = ['unit_number', 'time_in_cycles', 'operational_setting_1', 'operational_setting_2',
                'operational_setting_3'] + [f'sensor_{i}' for i in range(6, 27)]

for df in train_dfs + test_dfs:
    df.columns = column_names

def calculate_rul(train_df, rul_df):
    rul_series = rul_df[0].values
    rul_df['unit_number'] = range(1, len(rul_df) + 1)  # To align with unit numbers
    for unit in train_df['unit_number'].unique():
        unit_cycles = train_df[train_df['unit_number'] == unit]['time_in_cycles'].max()
        rul = unit_cycles - train_df[train_df['unit_number'] == unit]['time_in_cycles'] 
        train_df.loc[train_df['unit_number'] == unit, 'RUL'] = rul
    
    return train_df

# Calculate RUL for each training set
train_rul_dfs = [calculate_rul(train, rul) for train, rul in zip(train_dfs, rul_dfs)]

combined_train_dfs = [train.assign(RUL=train['RUL']) for train in train_rul_dfs]
def assign_rul_test(test_df, train_df):
    test_df['RUL'] = test_df['unit_number'].apply(lambda x: train_df[train_df['unit_number'] == x]['time_in_cycles'].max())
    test_df['RUL'] = test_df['RUL'] - test_df['time_in_cycles']
    return test_df

test_dfs_with_rul = [assign_rul_test(test, train) for test, train in zip(test_dfs, train_dfs)]

for i, df in enumerate(combined_train_dfs):
    df.to_csv(f'processed_train_FD00{i+1}.csv', index=False)

for i, df in enumerate(test_dfs_with_rul):
    df.to_csv(f'processed_test_FD00{i+1}.csv', index=False)

print(combined_train_dfs[0].head())
print(test_dfs_with_rul[0].head())


def create_features(df):
    # Calculate statistical features for sensors
    for i in range(1, 27):
        df[f'sensor_{i}_mean'] = df[f'sensor_{i}'].rolling(window=5).mean()
        df[f'sensor_{i}_std'] = df[f'sensor_{i}'].rolling(window=5).std()
        df[f'sensor_{i}_max'] = df[f'sensor_{i}'].rolling(window=5).max()
    
    # Create lag features (example for the last 2 time steps)
    for i in range(1, 27):
        df[f'sensor_{i}_lag1'] = df[f'sensor_{i}'].shift(1)
        df[f'sensor_{i}_lag2'] = df[f'sensor_{i}'].shift(2)
    
    return df

# Apply feature engineering on train and test datasets
train_featured_dfs = [create_features(df) for df in combined_train_dfs]
test_featured_dfs = [create_features(df) for df in test_dfs_with_rul]

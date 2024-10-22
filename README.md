# Predictive-Maintainance-For-TurboFan-Engine

## Project Overview

This project focuses on predicting the Remaining Useful Life (RUL) of turbofan engines using data from NASA's Prognostics Center of Excellence. The goal is to develop a machine learning model that can accurately predict when an engine will fail based on sensor data and operational settings.

## Dataset Description

The dataset consists of multiple time series from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation. The data is divided into four subsets:

1. **FD001**:
   - Train trajectories: 100
   - Test trajectories: 100
   - Conditions: ONE (Sea Level)
   - Fault Modes: ONE (HPC Degradation)

 Data file contains 26 columns:
1. Unit number
2. Time (cycles)
3-5. Operational settings
6-21. Sensor measurements
## Methodology

1. **Data Loading and Preprocessing**
   - Utilized pandas to read and process the data files
   - Created lists of train, test, and RUL files
   - Loaded datasets using a custom function with `pd.read_csv`
   - Assigned column names to the dataframes

2. **Feature Engineering**
   - Added a 'RUL' column to the training data
   - Processed the data to include RUL values

3. **Model Development**
   - Implemented a Random Forest Regressor
   - Used scikit-learn's `train_test_split` for data splitting

4. **Model Training and Evaluation**
   - Trained the Random Forest model on the prepared data
   - Evaluated the model using various metrics:
     - Mean Squared Error (MSE)
     - Mean Absolute Error (MAE)
     - Root Mean Squared Error (RMSE)
     - R-squared (R²)
     - Median Absolute Error

5. **Model Persistence**
   - Saved the trained model using joblib for future use

6. **Prediction Function**
   - Developed a function to make RUL predictions based on manual input

## Results

The Random Forest Regressor showed promising results in predicting the Remaining Useful Life of turbofan engines. Key performance metrics include:

- Mean Squared Error: 259.47
- Mean Absolute Error: 11.01
- Root Mean Squared Error: 16.11
- R-Squared: 0.9432
- Median Absolute Error: 7.38

These metrics indicate that the model performs well, with a high R-squared value suggesting good predictive power.

### Visualizations


1. **Feature Importance Plot**: A bar chart showing the importance of each sensor and operational setting in predicting RUL.
2. **Actual vs Predicted RUL Scatter Plot**: Comparing the model's predictions against actual RUL values.
3. **RUL Prediction Error Distribution**: A histogram showing the distribution of prediction errors.
4. **Sensor Data Trends**: Line plots showing how key sensor readings change over time for different engines.

## Usage

To run the analysis:

1. Clone this repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook main.ipynb`

To use the trained model for predictions:

```python
import joblib

# Load the saved model
model = joblib.load('rul_predictor_model.joblib')

# Function to make predictions from manual input
def predict_rul(manual_input):
    input_array = np.array(manual_input).reshape(1, -1)
    predicted_rul = model.predict(input_array)
    return predicted_rul[0]

# Example usage
manual_input = [4, 72, -0.002, 0.0002, 100, ...]  # Add all 26 feature values
predicted_rul = predict_rul(manual_input)
print('Predicted Remaining Useful Life (RUL):', predicted_rul)

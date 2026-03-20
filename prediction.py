import numpy as np
import pandas as pd
import shap
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import io
import shap
import base64

def get_preprocessor(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Ensure 'classification' exists before modifying
    if 'classification' in data.columns:
        data['classification'] = data['classification'].replace(['ckd\t'], 'ckd')

    # Ensure 'id' exists before dropping it
    if 'id' in data.columns:
        data = data.drop(['id'], axis=1)

    # Handle missing values
    data.fillna(data.median(numeric_only=True), inplace=True)
    
    for col in data.select_dtypes(include=['object']).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Label Encoding for categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Drop classification column
    X = data.drop('classification', axis=1, errors='ignore')

    # Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return scaler, label_encoders, X.columns  # Return feature names too


def preprocess_input_data(data_dict):
    """Preprocesses a single row of data in dictionary format."""

    try:
        # Load the scaler and encoders
        scaler, label_encoders, feature_columns = get_preprocessor("static/models/chronic_kidney_disease/data/processed_kidney_disease.csv")

        print("Input Data:", data_dict)

        # Convert dictionary to DataFrame
        instance_df = pd.DataFrame([data_dict])

        # Ensure all required columns exist
        for col in feature_columns:
            if col not in instance_df.columns:
                instance_df[col] = 0  # Default for missing numerical values

        # Encode categorical columns
        for col in label_encoders:
            if col in instance_df.columns:
                value = instance_df[col].astype(str)
                if value.iloc[0] not in label_encoders[col].classes_:
                    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, value.iloc[0])
                instance_df[col] = label_encoders[col].transform(value)

        # Fill missing values with median (for safety)
        instance_df.fillna(instance_df.median(numeric_only=True), inplace=True)

        # Ensure columns are in correct order
        instance_df = instance_df[feature_columns]

        # Standardize numerical data
        scaled_instance = scaler.transform(instance_df)

        print("Preprocessed Data:", type(scaled_instance))
        return scaled_instance

    except (KeyError, ValueError) as e:
        print(f"Error processing data: {e}")
        return None
    
    
def predict_explain(scaled_instance, rf_model, explainer, columns):
    """Predicts the class and generates SHAP values for a single instance.

    Args:
      scaled_instance: A NumPy array representing a single scaled instance.
      rf_model: The trained RandomForestClassifier model.
      explainer: The SHAP explainer object.
      X_test_df: The DataFrame of test features.

    Returns:
        A tuple containing the prediction, force plot HTML, and SHAP values.
    """

    try:
        if len(scaled_instance.shape) == 1:
            scaled_instance = scaled_instance.reshape(1, -1)
            print(scaled_instance.shape)
        else:
            print("scaled_instance is already a 2D array with shape:", scaled_instance.shape)
        # Make predictions
        predicted_class = rf_model.predict(scaled_instance)[0]
        predicted_probs = rf_model.predict_proba(scaled_instance)[0]
        # print(predicted_class)
        # print(predicted_probs)
        
        

        # Extract SHAP values
        shap_values_single = explainer.shap_values(scaled_instance) 
        print(f"Shape of shap_values_single: {shap_values_single.shape}") # Use the explainer directly
        shap_values_for_class = shap_values_single[0, :, predicted_class]
        print(f"Shape of shap_values_for_class: {shap_values_for_class.shape}")
        
        
        return predicted_class,predicted_probs[0],shap_values_for_class,explainer

    except Exception as e:
        print(f"An error occurred during prediction or explanation: {e}")
        return None, None, None



def get_explainer(rf_model, X_test):
    """
    Generates and returns a SHAP explainer for the given model and data.

    Args:
        rf_model: The trained random forest model.
        X_test:  The test data (features).

    Returns:
        A SHAP explainer object.
    """
    try:
      explainer = shap.TreeExplainer(rf_model)
      return explainer
    except Exception as e:
      print(f"Error creating explainer: {e}")
      return None
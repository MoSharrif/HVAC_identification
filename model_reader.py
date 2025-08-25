import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def train_xgboost_model(features, labels, save_path="xgboost_model.joblib"):
    """
    Trains an XGBoost model with the provided features and labels and saves the model to a file.
    
    Parameters:
        features (pd.DataFrame): Feature matrix.
        labels (pd.Series): Target labels.
        save_path (str): Path to save the trained model.
    
    Returns:
        None
    """
    # Encode labels if they are categorical
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
    
    # Train the XGBoost model
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, n_estimators=100)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=True)
    
    # Save the trained model
    model.save_model(save_path)
    print(f"Model saved to {save_path}")

    # Evaluate the model
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))


def predict_and_visualize(model_path, time_series_data, label_encoder_path=None):
    """
    Loads a trained XGBoost model, predicts labels for the provided time series data, 
    and visualizes the results.
    
    Parameters:
        model_path (str): Path to the trained model.
        time_series_data (pd.DataFrame): Time series data to predict on.
        label_encoder_path (str, optional): Path to the label encoder if needed.
    
    Returns:
        None
    """
    # Load the trained model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Prepare the data for prediction
    dmatrix_data = xgb.DMatrix(data=time_series_data)
    
    # Make predictions
    predictions = model.predict(time_series_data)
    
    # Visualize the predictions
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label="Predicted Labels", color='blue')
    plt.title("Predicted Labels for Time Series Data")
    plt.xlabel("Time Point")
    plt.ylabel("Predicted Label")
    plt.legend()
    plt.show()
    
    # (Optional) Load and use label encoder for original labels
    if label_encoder_path:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)
        predictions_labels = label_encoder.inverse_transform(predictions)
        print("Predicted Labels (Decoded):", predictions_labels)

# Example Usage
# Assuming combined_features_temp and final_label_data are generated from the previous steps
# features = combined_features_temp.drop(columns=['ID'])
# labels = final_label_data['some_label_column']  # Replace with actual column name
# train_xgboost_model(features, labels)
# predict_and_visualize("xgboost_model.json", time_series_data)

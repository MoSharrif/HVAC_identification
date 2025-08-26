import numpy as np
import joblib
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import csv
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import json
import argparse


LABEL_DICT = {"PV": "Only_PV", "heat_pump+PV": "PV+HP", "EV": "EV_NoPV", "EV+PV": "EV+PV", "None":"NONE" }

# Model persistence configuration
MODEL_DIR = "saved_models"
DEFAULT_MODEL_NAME = "xgboost_classifier"

# Feature caching system for performance optimization
FEATURE_CACHE_DIR = "feature_cache"


def ensure_model_directory():
    """Ensure the model directory exists."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created model directory: {MODEL_DIR}")

def ensure_feature_cache_directory():
    """Ensure the feature cache directory exists."""
    if not os.path.exists(FEATURE_CACHE_DIR):
        os.makedirs(FEATURE_CACHE_DIR)
        print(f"Created feature cache directory: {FEATURE_CACHE_DIR}")

def get_feature_cache_path(cache_key):
    """Get the file path for cached features."""
    return os.path.join(FEATURE_CACHE_DIR, f"{cache_key}_features.joblib")

def get_feature_metadata_path(cache_key):
    """Get the file path for feature metadata."""
    return os.path.join(FEATURE_CACHE_DIR, f"{cache_key}_metadata.json")

def feature_cache_exists(cache_key):
    """Check if cached features exist on disk."""
    if not cache_key:
        return False
    feature_path = get_feature_cache_path(cache_key)
    metadata_path = get_feature_metadata_path(cache_key)
    return os.path.exists(feature_path) and os.path.exists(metadata_path)

def save_features_to_disk(X, y, cache_key, labels_df_shape, time_series_shape):
    """Save features to disk with metadata."""
    if not cache_key:
        return
    
    ensure_feature_cache_directory()
    
    feature_path = get_feature_cache_path(cache_key)
    metadata_path = get_feature_metadata_path(cache_key)
    
    # Save features
    joblib.dump((X, y), feature_path)
    
    # Save metadata for validation
    metadata = {
        'cache_key': cache_key,
        'labels_df_shape': labels_df_shape,
        'time_series_shape': time_series_shape,
        'feature_shape': X.shape,
        'target_shape': y.shape,
        'timestamp': time.time()
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Features saved to disk: {feature_path}")

def load_features_from_disk(cache_key):
    """Load features from disk."""
    if not feature_cache_exists(cache_key):
        return None
    
    feature_path = get_feature_cache_path(cache_key)
    metadata_path = get_feature_metadata_path(cache_key)
    
    try:
        # Load features
        X, y = joblib.load(feature_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Features loaded from disk: {feature_path}")
        print(f"   Features shape: {metadata['feature_shape']}")
        print(f"   Cached on: {time.ctime(metadata['timestamp'])}")
        
        return X, y, metadata
    except Exception as e:
        print(f"⚠️  Error loading cached features for {cache_key}: {e}")
        return None

def validate_cached_features(cache_key, labels_df, time_series):
    """Validate that cached features match current input data."""
    if not feature_cache_exists(cache_key):
        return False
    
    try:
        metadata_path = get_feature_metadata_path(cache_key)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check if input data shapes match
        current_labels_shape = labels_df.shape
        current_time_series_shape = time_series.shape
        
        if (metadata['labels_df_shape'] != list(current_labels_shape) or 
            metadata['time_series_shape'] != list(current_time_series_shape)):
            print(f"⚠️  Cache invalid for {cache_key}: input data shape changed")
            return False
        
        return True
    except Exception as e:
        print(f"⚠️  Error validating cache for {cache_key}: {e}")
        return False

def save_model(model, label_encoder, classification_report_dict, model_name=DEFAULT_MODEL_NAME):
    """
    Save the trained model, label encoder, and classification report.
    
    Args:
        model: Trained XGBoost classifier
        label_encoder: LabelEncoder used for the model
        classification_report_dict: Classification report dictionary
        model_name: Name for the saved model files
    """
    ensure_model_directory()
    
    # Define file paths
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
    encoder_path = os.path.join(MODEL_DIR, f"{model_name}_label_encoder.joblib")
    report_path = os.path.join(MODEL_DIR, f"{model_name}_classification_report.json")
    
    # Save model and encoder using joblib
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    
    # Save classification report as JSON
    with open(report_path, 'w') as f:
        json.dump(classification_report_dict, f, indent=2)
    
    print(f"✅ Model saved successfully:")
    print(f"   Model: {model_path}")
    print(f"   Label Encoder: {encoder_path}")
    print(f"   Classification Report: {report_path}")

def model_exists(model_name=DEFAULT_MODEL_NAME):
    """
    Check if a saved model exists.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        bool: True if all required files exist, False otherwise
    """
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
    encoder_path = os.path.join(MODEL_DIR, f"{model_name}_label_encoder.joblib")
    report_path = os.path.join(MODEL_DIR, f"{model_name}_classification_report.json")
    
    return all(os.path.exists(path) for path in [model_path, encoder_path, report_path])

def load_model(model_name=DEFAULT_MODEL_NAME):
    """
    Load a saved model, label encoder, and classification report.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        tuple: (model, label_encoder, classification_report_dict)
        
    Raises:
        FileNotFoundError: If the model files don't exist
    """
    if not model_exists(model_name):
        raise FileNotFoundError(f"Model '{model_name}' not found in {MODEL_DIR}")
    
    # Define file paths
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
    encoder_path = os.path.join(MODEL_DIR, f"{model_name}_label_encoder.joblib")
    report_path = os.path.join(MODEL_DIR, f"{model_name}_classification_report.json")
    
    # Load model and encoder
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    
    # Load classification report
    with open(report_path, 'r') as f:
        classification_report_dict = json.load(f)
    
    print(f"✅ Model loaded successfully:")
    print(f"   Model: {model_path}")
    print(f"   Label Encoder: {encoder_path}")
    print(f"   Classification Report: {report_path}")
    
    return model, label_encoder, classification_report_dict

def get_or_train_real_data_model(X_real, y_real, model_name="real_data_classifier", force_retrain=False):
    """
    Get a trained model for real data - either load existing or train new one.
    
    Args:
        X_real: Real data features
        y_real: Real data labels
        model_name: Name for the saved model
        force_retrain: If True, retrain even if saved model exists
        
    Returns:
        tuple: (model, X_test, y_test, label_encoder, classification_report_dict)
    """
    if not force_retrain and model_exists(model_name):
        print(f"🔄 Loading existing model '{model_name}'...")
        model, label_encoder, classification_report_dict = load_model(model_name)
        
        # For loaded models, we need to create test data for consistency
        # Use the same split parameters as training
        _, X_test, _, y_test = train_test_split(
            X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
        )
        
        return model, X_test, y_test, label_encoder, classification_report_dict
    else:
        if force_retrain:
            print(f"🔄 Force retraining model '{model_name}'...")
        else:
            print(f"🔄 Training new model '{model_name}'...")
        
        # Train new model
        model, X_test, y_test, label_encoder, classification_report_dict = create_real_data_classifier(X_real, y_real)
        
        # Save the model
        save_model(model, label_encoder, classification_report_dict, model_name)
        return model, X_test, y_test, label_encoder, classification_report_dict

def list_saved_models():
    """
    List all saved models in the model directory.
    
    Returns:
        list: List of model names (without file extensions)
    """
    if not os.path.exists(MODEL_DIR):
        print(f"Model directory '{MODEL_DIR}' does not exist.")
        return []
    
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_model.joblib')]
    model_names = [f.replace('_model.joblib', '') for f in model_files]
    
    if model_names:
        print(f"Available saved models in '{MODEL_DIR}':")
        for name in model_names:
            print(f"  - {name}")
    else:
        print(f"No saved models found in '{MODEL_DIR}'.")
    
    return model_names

def get_consistent_color_mapping():
    """
    Returns a consistent color mapping for all class categories across all plots.
    This ensures that each class always has the same color in every visualization.
    Uses explicit hex colors to guarantee consistency.
    """
    class_color_mapping = {
        'EV': '#1f77b4',      # Blue - Electric Vehicle (no PV)
        'EV+PV': '#ff7f0e',   # Orange - Electric Vehicle + PV
        'PV': '#2ca02c',      # Green - PV only
        'PV+HP': '#d62728',   # Red - PV + Heat Pump  
        'None': '#9467bd'     # Purple - Standard household
    }
    return class_color_mapping

def train_XGBoost_optimized(X, y, n_splits=5, random_state=42):
    """
    Optimized version of XGBoost training with improved performance:
    - Pre-encode labels once
    - More efficient fold splitting
    - Reduced redundant operations
    """
    # Encode labels to integers once
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Number of classes: {len(np.unique(y_encoded))}")
    print(f"Encoded class distribution: {pd.Series(y_encoded).value_counts().to_dict()}")

    # Initialize K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_accuracy = 0
    best_xgb_model = None

    # Pre-compute XGBoost parameters
    xgb_params = {
        'objective': 'multi:softmax',
        'num_class': len(np.unique(y_encoded)),
        'eval_metric': 'mlogloss',
        'random_state': random_state,
        'n_jobs': -1,  # Use all available cores
        'verbosity': 0  # Reduce logging overhead
    }

    # K-Fold Cross-Validation with optimizations
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        # Split data using iloc for faster indexing
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        
        # Check class distribution in this fold
        unique_train_classes = len(np.unique(y_train))
        if unique_train_classes < 2:
            print(f"Warning: Fold {fold + 1} has only {unique_train_classes} class(es)")
            continue
        
        # Train the XGBoost model with optimized parameters
        xgb_model = XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)
        
        # Make predictions and calculate accuracy
        y_pred = xgb_model.predict(X_test)
        fold_accuracy = accuracy_score(y_test, y_pred)
        
        # Check if this is the best model so far
        if fold_accuracy > best_accuracy:
            best_accuracy = fold_accuracy
            best_xgb_model = xgb_model

        print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")

    print(f"Best Fold Accuracy: {best_accuracy:.4f}")
    return best_xgb_model

def train_XGBoost_with_proper_split_optimized(X, y, test_size=0.2, random_state=42):
    """
    Optimized version of train_XGBoost_with_proper_split with improved performance.
    """
    # Step 1: Split data into train (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {len(X_train)} ({(1-test_size)*100:.0f}%)")
    print(f"Test set size: {len(X_test)} ({test_size*100:.0f}%)")
    
    # Step 2: Encode labels to integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    print(f"Number of classes: {len(np.unique(y_train_encoded))}")

    # Step 3: Use optimized XGBoost training
    best_xgb_model = train_XGBoost_optimized(X_train, y_train, random_state=random_state)

    print(f"Training completed successfully!")
    return best_xgb_model, X_test, y_test, label_encoder


def calculate_features_optimized(labels_df, time_series):
    """
    Optimized version of calculate_features with improved performance:
    - Pre-convert datetime index once
    - Vectorized operations where possible
    - Reduced memory allocations
    - Efficient concatenation strategy
    """
    # Convert to datetime index only once
    if not isinstance(time_series.index, pd.DatetimeIndex):
        time_series.index = pd.to_datetime(time_series.index)
    
    # Check for NaN values and fill once
    if time_series.isnull().any().any():
        print("Warning: time_series contains NaN values")
        time_series = time_series.fillna(0)
    
    # Define feature extraction with optimized operations
    def extract_time_features_optimized(df, freq, prefix):
        # Resample once for efficiency
        resampled = df.resample(freq)
        
        # Calculate statistics efficiently
        mean = resampled.mean()
        std = resampled.std()
        max_ = resampled.max()
        skew = resampled.apply(pd.Series.skew)
        sum_ = resampled.sum()
        # min = resampled.min()
        
        # Peak-to-average ratio: vectorized calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            peak_to_avg = max_ / mean
            peak_to_avg = peak_to_avg.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Prepare feature names efficiently
        n_periods = len(mean)
        
        # Create feature names for each statistic
        mean.index = [f"{prefix}_mean_{i}" for i in range(n_periods)]
        std.index = [f"{prefix}_std_{i}" for i in range(n_periods)]
        skew.index = [f"{prefix}_skew_{i}" for i in range(n_periods)]
        max_.index = [f"{prefix}_max_{i}" for i in range(n_periods)]
        sum_.index = [f"{prefix}_sum_{i}" for i in range(n_periods)]
        # min.index = [f"{prefix}_min_{i}" for i in range(n_periods)]
        peak_to_avg.index = [f"{prefix}_peak_to_avg_{i}" for i in range(n_periods)]
        
        # Concatenate all features efficiently
        features = pd.concat([mean, std, skew, max_, sum_, peak_to_avg], axis=0)
        
        # Convert to DataFrame format expected by downstream code
        features = features.reset_index()
        features = features.rename(columns={'index': 'feature'})
        
        return features

    # Extract features for all time frequencies
    print("Extracting weekly features...")
    weekly_features = extract_time_features_optimized(time_series, 'W', 'W')
    
    print("Extracting monthly features...")
    monthly_features = extract_time_features_optimized(time_series, 'ME', 'M')
    
    print("Extracting annual features...")
    annual_features = extract_time_features_optimized(time_series, 'YE', 'A')
    
    # Combine features efficiently
    print("Combining features...")
    combined_features = pd.concat([weekly_features, monthly_features, annual_features], 
                                 axis=0, ignore_index=True)

    # Reshape efficiently
    feature_names = combined_features['feature'].values
    combined_features_temp = combined_features.drop(columns=['feature']).transpose()
    combined_features_temp.columns = feature_names
    combined_features_temp = combined_features_temp.reset_index().rename(columns={'index': 'ID'})
    combined_features_temp['ID'] = combined_features_temp['ID'].astype(int)
    
    # Fill any remaining NaN values
    if combined_features_temp.isnull().any().any():
        print("Warning: combined_features contains NaN values, filling with 0")
        combined_features_temp = combined_features_temp.fillna(0)
    
    # Merge with labels
    data_synthetic = pd.merge(combined_features_temp, labels_df[['ID', 'Category']], on='ID')

    # Separate features and target
    X = data_synthetic.drop(columns=['Category', 'ID'])
    y = data_synthetic['Category']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y

def calculate_features_cached(labels_df, time_series, cache_key=None, force_recalculate=False):
    """
    Cached version of feature calculation with disk persistence to avoid redundant computations.
    
    This function implements a disk caching strategy:
    1. Disk cache for persistence across sessions
    2. Validation to ensure data integrity
    
    Args:
        labels_df: DataFrame containing labels
        time_series: Time series data
        cache_key: Optional key for caching (use dataset identifier)
        force_recalculate: If True, force recalculation even if cache exists
    
    Returns:
        X, y: Features and labels
    """
    if not cache_key:
        print("No cache key provided, calculating features without caching")
        return calculate_features_optimized(labels_df, time_series)
    
    # Check disk cache (fast)
    if not force_recalculate and feature_cache_exists(cache_key):
        # Validate that cached features match current input data
        if validate_cached_features(cache_key, labels_df, time_series):
            result = load_features_from_disk(cache_key)
            if result is not None:
                X, y, metadata = result
                print(f"💾 Loaded features from disk cache for {cache_key}")
                return X, y
        else:
            print(f"🗑️ Cached features invalid for {cache_key}, recalculating...")
    
    # Calculate features (slowest)
    if force_recalculate:
        print(f"🔄 Force recalculating features for {cache_key}")
    else:
        print(f"⚙️ Calculating features for {cache_key or 'unknown dataset'}")
    
    start_time = time.time()
    X, y = calculate_features_optimized(labels_df, time_series)
    calculation_time = time.time() - start_time
    
    print(f"✅ Feature calculation completed in {calculation_time:.2f} seconds")
    
    # Save to disk cache
    save_features_to_disk(X, y, cache_key, list(labels_df.shape), list(time_series.shape))
    
    print(f"🗃️ Features cached to disk for {cache_key}")
    
    return X, y


def clear_disk_feature_cache():
    """Clear all cached features from disk."""
    if not os.path.exists(FEATURE_CACHE_DIR):
        print("📁 Feature cache directory doesn't exist")
        return
    
    try:
        import shutil
        shutil.rmtree(FEATURE_CACHE_DIR)
        print(f"🗑️ Disk feature cache cleared: {FEATURE_CACHE_DIR}")
    except Exception as e:
        print(f"⚠️ Error clearing disk cache: {e}")

def list_cached_features():
    """List all cached features on disk."""
    if not os.path.exists(FEATURE_CACHE_DIR):
        print(f"📁 Feature cache directory '{FEATURE_CACHE_DIR}' does not exist")
        return []
    
    cache_files = [f for f in os.listdir(FEATURE_CACHE_DIR) if f.endswith('_features.joblib')]
    cache_keys = [f.replace('_features.joblib', '') for f in cache_files]
    
    if cache_keys:
        print(f"📋 Available cached features in '{FEATURE_CACHE_DIR}':")
        for key in cache_keys:
            try:
                metadata_path = get_feature_metadata_path(key)
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    print(f"  - {key}: {metadata['feature_shape']} features, cached {time.ctime(metadata['timestamp'])}")
                else:
                    print(f"  - {key}: (metadata missing)")
            except Exception as e:
                print(f"  - {key}: (error reading metadata: {e})")
    else:
        print(f"📁 No cached features found in '{FEATURE_CACHE_DIR}'")
    
    return cache_keys

def demo_feature_caching():
    """
    Demonstration function showing the feature caching capabilities.
    """
    print("\n" + "="*70)
    print("FEATURE CACHING DEMO")
    print("="*70)
    
    print("\n1. Feature Caching System Overview:")
    print("   • Disk caching for persistence across script runs")
    print("   • Automatic data validation and cache invalidation")
    print("   • Significant time savings for large datasets")
    
    print("\n2. Available cached features:")
    cached_features = list_cached_features()
    
    print("\n3. Usage Examples:")
    print("   # Basic usage (auto-caching):")
    print("   X, y = calculate_features_cached(labels_df, time_series, 'my_dataset')")
    print()
    print("   # Force recalculation:")
    print("   X, y = calculate_features_cached(labels_df, time_series, 'my_dataset', force_recalculate=True)")
    print()
    print("   # Check if features are cached:")
    print("   if feature_cache_exists('my_dataset'):")
    print("       print('Features are already cached!')")
    
    print("\n4. Cache Management:")
    print("   # Get cache info:")
    print("   get_cache_info('my_dataset')")
    print()
    print("   # Clear disk cache:")
    print("   clear_disk_feature_cache()")
    
    print("\n5. Command Line Usage:")
    print("   python use_XGBoost.py                           # Use cached features")
    print("   python use_XGBoost.py --force-recalculate-features  # Force recalculation")
    print("   python use_XGBoost.py --list-cached-features    # List all cached features") 
    print("   python use_XGBoost.py --clear-feature-cache     # Clear all cached features from disk")
    print("   python use_XGBoost.py --cache-info real_data    # Get info about specific cache")
    
    print("\n6. Performance Benefits:")
    print("   • First run: Calculates and saves features (slow)")
    print("   • Subsequent runs: Loads features from disk (fast)")
    print("   • Typical speedup: 10-100x for large datasets")
    print("   • Automatic validation ensures data integrity")
    
    print("="*70)


def get_cache_info(cache_key):
    """Get detailed information about a cached feature set."""
    if not feature_cache_exists(cache_key):
        print(f"❌ Cache '{cache_key}' does not exist")
        return None
    
    try:
        metadata_path = get_feature_metadata_path(cache_key)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        feature_path = get_feature_cache_path(cache_key)
        file_size = os.path.getsize(feature_path) / (1024 * 1024)  # MB
        
        print(f"📊 Cache info for '{cache_key}':")
        print(f"   Features shape: {metadata['feature_shape']}")
        print(f"   Labels shape: {metadata['target_shape']}")
        print(f"   Input data shape: {metadata['labels_df_shape']} x {metadata['time_series_shape']}")
        print(f"   File size: {file_size:.2f} MB")
        print(f"   Cached on: {time.ctime(metadata['timestamp'])}")
        
        return metadata
    except Exception as e:
        print(f"⚠️ Error reading cache info for {cache_key}: {e}")
        return None

# Add alias for backward compatibility
calculate_features = calculate_features_optimized


def get_sep(path: pathlib.Path) -> str:
    """
    Determines and returns the separator used in a CSV file.
    """
    with open(path, newline = "") as file:
        sep = csv.Sniffer().sniff(file.read()).delimiter
        return sep
    
def load_10_000_unlabeled_synthetic_profiles(path: pathlib.Path=pathlib.Path("input_data")):
    """
    Loads the 10,000 unlabeled synthetic profiles from the input data.
    """
    df = pd.read_csv(path / f"10000_profiles_all.csv", sep=get_sep(path / f"10000_profiles_all.csv"), index_col=0)
    return df

def load_synthetic_1000_profiles_per_type(path: pathlib.Path=pathlib.Path("input_data")):
    """
    Loads the synthetic 1000 profiles per type from the input data.
    """
    df_list = []
    for i, label in LABEL_DICT.items():
        df = pd.read_csv(path / f"1000_profiles_{i}.csv", sep=get_sep(path / f"1000_profiles_{i}.csv"), index_col=0)
        df_list.append(df)

    synthetic_timeSeries = pd.concat(df_list, axis=1)
    synthetic_timeSeries.columns = np.arange(1, 5_001)
    synthetic_timeSeries.index = pd.to_datetime(synthetic_timeSeries.index)
    return synthetic_timeSeries


def load_5000_synthetic_profiles_per_type(path: pathlib.Path=pathlib.Path("input_data")):
    """
    Loads the 5000 synthetic profiles per type from the input data.
    """
    df_list = []
    for i, label in LABEL_DICT.items():
        df = pd.read_csv(path / f"5000_profiles_{i}.csv", sep=get_sep(path / f"5000_profiles_{i}.csv"), index_col=0)
        df_list.append(df)

    synthetic_timeSeries = pd.concat(df_list, axis=1)
    synthetic_timeSeries.columns = np.arange(1, synthetic_timeSeries.shape[1]+1)
    synthetic_timeSeries.index = pd.to_datetime(synthetic_timeSeries.index)
    return synthetic_timeSeries

def load_50_000_synthetic_profiles_per_type(path: pathlib.Path=pathlib.Path("input_data")):
    """
    Loads the 50,000 synthetic profiles per type from the input data.
    """
    df_list = []
    for i, label in LABEL_DICT.items():
        df = pd.read_parquet(path / f"50000_profiles_{i}.parquet")
        df_list.append(df)

    synthetic_timeSeries = pd.concat(df_list, axis=1)
    synthetic_timeSeries.columns = np.arange(1, synthetic_timeSeries.shape[1]+1)
    synthetic_timeSeries.index = pd.to_datetime(synthetic_timeSeries.index)
    return synthetic_timeSeries

def load_100_000_synthetic_profiles_per_type(path: pathlib.Path=pathlib.Path("input_data")):
    """
    Loads the 100,000 synthetic profiles per type from the input data.
    """
    df_list = []
    for i, label in LABEL_DICT.items():
        df = pd.read_parquet(path / f"100000_profiles_{i}.parquet")
        df_list.append(df)

    synthetic_timeSeries = pd.concat(df_list, axis=1)
    synthetic_timeSeries.columns = np.arange(1, synthetic_timeSeries.shape[1]+1)
    synthetic_timeSeries.index = pd.to_datetime(synthetic_timeSeries.index)
    return synthetic_timeSeries

def load_synthetic_profiles_in_originial_shape(path: pathlib.Path=pathlib.Path("input_data")):
    """
    Loads the synthetic profiles in the original shape from the input data.
    """
    df_list = []
    for i, label in LABEL_DICT.items():
        if label == "EV":
            df = pd.read_csv(path / f"100_profiles_{i}.csv", sep=get_sep(path / f"100_profiles_{i}.csv"), index_col=0)
        else:
            df = pd.read_csv(path / f"300_profiles_{i}.csv", sep=get_sep(path / f"300_profiles_{i}.csv"), index_col=0)
        df_list.append(df)

    synthetic_timeSeries = pd.concat(df_list, axis=1)
    synthetic_timeSeries.columns = np.arange(1, 1301)
    synthetic_timeSeries.index = pd.to_datetime(synthetic_timeSeries.index)
    return synthetic_timeSeries

def load_1300_unlabeled_synthetic_profiles(path: pathlib.Path=pathlib.Path("input_data")):
    """
    Loads the 1300 unlabeled synthetic profiles from the input data.
    """
    df = pd.read_csv(path / f"1300_profiles_all.csv", sep=get_sep(path / f"1300_profiles_all.csv"), index_col=0)
    return df

def create_1000_labels_df():
    labels = [1000*[label] for i, label in LABEL_DICT.items()]
    flat_list = [item for sublist in labels for item in sublist]
    labels_df = pd.DataFrame({"Category": flat_list, "ID": np.arange(1, 5_001)})
    return labels_df

def create_labels_for_50_000_labels_df():
    labels = [50_000*[label] for i, label in LABEL_DICT.items()]
    flat_list = [item for sublist in labels for item in sublist]
    labels_df = pd.DataFrame({"Category": flat_list, "ID": np.arange(1, 250_001)})
    return labels_df

def create_labels_for_100_000_labels_df():
    labels = [100_000*[label] for i, label in LABEL_DICT.items()]
    flat_list = [item for sublist in labels for item in sublist]
    labels_df = pd.DataFrame({"Category": flat_list, "ID": np.arange(1, 500_001)})
    return labels_df


def plot_recall(classification_report_df, title: str):
    # Plot per-label accuracy (recall)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=classification_report_df.index[:-3], y=classification_report_df['recall'][:-3], palette="viridis")
    plt.title(title)
    plt.xlabel("Labels")
    plt.ylabel("Recall")
    plt.xticks(rotation=45, ha="right")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title: str):
    # Step 4: Confusion Matrix
    # Both y_true and y_pred should be in the same format (string labels)
    
    # Get unique labels and sort them for consistent ordering
    labels = sorted(set(y_true) | set(y_pred))
    
    # Create confusion matrix with explicit label ordering
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

def load_real_data(path: pathlib.Path=pathlib.Path("input_data")):
    """
    Loads the real data from the input data.
    """
    df = pd.read_csv(path / "fluvius_wide_format.csv", sep=get_sep(path / "fluvius_wide_format.csv"), index_col=0)
    return df

def load_1300_synthetic_profiles(path: pathlib.Path=pathlib.Path("input_data")):
    """
    Loads the 1300 synthetic profiles per type from the input data.
    """
    df = pd.read_csv(path / "1300_profiles_all.csv", sep=get_sep(path / "1300_profiles_all.csv"), index_col=0)
    return df

def load_10_000_synthetic_profiles(path: pathlib.Path=pathlib.Path("input_data")):
    """
    Loads the 10,000 synthetic profiles from the input data.
    """
    df = pd.read_csv(path / "10000_profiles_all.csv", sep=get_sep(path / "10000_profiles_all.csv"), index_col=0)
    return df

def create_labels_for_original_shape_synthetic_profiles():
    """
    Creates the labels for the original shape synthetic profiles.
    """
    labels = []
    for i, label in LABEL_DICT.items():
        if i == "EV":
            labels.append(100*[label])
        else:
            labels.append(300*[label])

    flat_list = [item for sublist in labels for item in sublist]
    labels_df = pd.DataFrame({"Category": flat_list, "ID": np.arange(1, 1_301)})
    return labels_df

def create_labels_for_5000_synthetic_profiles_per_type():
    """
    Creates the labels for the 5000 synthetic profiles per type.
    """
    labels = [5000*[label] for i, label in LABEL_DICT.items()]
    flat_list = [item for sublist in labels for item in sublist]
    labels_df = pd.DataFrame({"Category": flat_list, "ID": np.arange(1, 25001)})
    return labels_df

def compare_model_with_different_synthetic_training_sizes(results_dict, performance_results_real_data, save_path='figures/performance_comparison.png'):
    """
    Create a publication-quality figure showing per-class XGBoost performance across different training data sizes.
    
    Args:
        results_dict: Dictionary with data sizes as keys and f1_scores as values
        save_path: Path to save the figure
    """
    # Set publication-quality style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Class name mapping for better readability
    class_name_mapping = {
        'EV_NoPV': 'EV',
        'NONE': 'None',
        'Only_PV': 'PV',
        'PV+HP': 'PV+HP',
        'EV+PV': 'EV+PV'
    }
    
    # Function to rename class names in results
    def rename_classes_in_results(results):
        renamed_per_class_f1 = {}
        renamed_per_class_precision = {}
        renamed_per_class_recall = {}
        for old_name, f1_score in results["per_class_f1"].items():
            new_name = class_name_mapping.get(old_name, old_name)
            renamed_per_class_f1[new_name] = f1_score
        for old_name, precision_score in results["per_class_precision"].items():
            new_name = class_name_mapping.get(old_name, old_name)
            renamed_per_class_precision[new_name] = precision_score
        for old_name, recall_score in results["per_class_recall"].items():
            new_name = class_name_mapping.get(old_name, old_name)
            renamed_per_class_recall[new_name] = recall_score
        results["per_class_f1"] = renamed_per_class_f1
        results["per_class_precision"] = renamed_per_class_precision
        results["per_class_recall"] = renamed_per_class_recall
        return results

    
    # Apply class name mapping to all results
    synthetic_results = {}
    for key, results in results_dict.items():
        synthetic_results[key] = rename_classes_in_results(results)
    
    # Sort synthetic results by training size (convert to int for sorting, then back to string)
    synthetic_keys_sorted = sorted(synthetic_results.keys(), key=lambda x: int(x))
    
    real_results = rename_classes_in_results(performance_results_real_data)
    # Create better x-axis labels
    data_labels = [f'{int(size):,}' for size in synthetic_keys_sorted]

    # Get all unique classes (with renamed labels)
    all_classes = set()
    for results in synthetic_results.values():
        all_classes.update(results['per_class_f1'].keys())
    all_classes = sorted(list(all_classes))
    

    
    # Use consistent color mapping across all plots
    consistent_colors = get_consistent_color_mapping()
    # Create figure with single subplot for per-class performance
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i, class_name in enumerate(all_classes):
        class_f1_scores = []
        class_precision_scores = []
        class_recall_scores = []
 
        # Add synthetic data results
        for size in synthetic_keys_sorted:
            class_f1_scores.append(synthetic_results[size]['per_class_f1'][class_name])
            class_precision_scores.append(synthetic_results[size]['per_class_precision'][class_name])
            class_recall_scores.append(synthetic_results[size]['per_class_recall'][class_name])
        
        # Get color for this class, fallback to gray if not in mapping
        class_color = consistent_colors.get(class_name, '#95A5A6')

        # calculate difference to real data
        class_f1_diff = np.array(class_f1_scores) - np.array(real_results['per_class_f1'][class_name])
        class_precision_diff = np.array(class_precision_scores) - np.array(real_results['per_class_precision'][class_name])
        class_recall_diff = np.array(class_recall_scores) - np.array(real_results['per_class_recall'][class_name])

        # create subplots for difference
        ax.plot(class_f1_diff, color=class_color, linewidth=1, alpha=1, label=class_name)
        # ax.plot(class_precision_diff, color=class_color, linewidth=0.5, alpha=1, linestyle='--', label=class_name)
        # ax.plot(class_recall_diff, color=class_color, linewidth=0.5, alpha=1, linestyle=':', label=class_name)
        
    
    ax.set_xlabel('Synthetic data size', fontsize=14)
    ax.set_ylabel('F1-Score difference', fontsize=14)
    ax.set_xticks(np.arange(len(data_labels)))
    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=14)
    ax.set_xticklabels(data_labels, rotation=0, ha='center', fontsize=14)
    ax.legend(loc='lower right', fontsize=18)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print summary statistics with renamed classes
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    # Print real data results first if available
    if performance_results_real_data:
        print(f"\nReal Data:")
        print("  Per-Class F1-Scores:")
        for class_name, f1_score in sorted(performance_results_real_data['per_class_f1'].items()):
            print(f"    {class_name}: {f1_score:.3f}")
    
    # Print synthetic data results
    for size in synthetic_keys_sorted:
        print(f"\nSynthetic Data {size} profiles:")
        print("  Per-Class F1-Scores:")
        for class_name, f1_score in sorted(synthetic_results[size]['per_class_f1'].items()):
            print(f"    {class_name}: {f1_score:.3f}")
    

def create_label_distribution_comparison(real_labels, predicted_labels, save_path='figures/label_distribution_comparison.svg'):
        """
        Create a comparison plot of real vs predicted label distributions (single plot).
        
        Args:
            real_labels: Series or array of real data labels
            predicted_labels: Series or array of predicted labels on synthetic data
            save_path: Path to save the figure
        """
        # Set publication-quality style
        plt.style.use('default')
        
        # Create single figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Class name mapping for better readability (same as in performance comparison)
        class_name_mapping = {
            'EV_NoPV': 'EV',
            'NONE': 'None',
            'Only_PV': 'PV',
            'PV+HP': 'PV+HP',
            'EV+PV': 'EV+PV'
        }
        
        # Apply class name mapping to labels
        real_labels_mapped = pd.Series(real_labels).map(class_name_mapping).fillna(pd.Series(real_labels))
        predicted_labels_mapped = pd.Series(predicted_labels).map(class_name_mapping).fillna(pd.Series(predicted_labels))
        
        # Get label counts and percentages
        real_counts = real_labels_mapped.value_counts()
        pred_counts = predicted_labels_mapped.value_counts()
        
        # Ensure both have the same categories for comparison
        all_categories = sorted(set(real_counts.index) | set(pred_counts.index))
        real_counts = real_counts.reindex(all_categories, fill_value=0)
        pred_counts = pred_counts.reindex(all_categories, fill_value=0)
        
        # Calculate percentages
        real_percentages = (real_counts / real_counts.sum()) * 100
        pred_percentages = (pred_counts / pred_counts.sum()) * 100
        
        # Side-by-side comparison
        x = np.arange(len(all_categories))
        width = 0.35
        
        bars_real = ax.bar(x - width/2, real_percentages.values, width, 
                           label='Real Data', color='#2E86AB', alpha=0.8, 
                           edgecolor='black', linewidth=1)
        bars_pred = ax.bar(x + width/2, pred_percentages.values, width, 
                           label='Predicted on Synthetic', color='#F18F01', alpha=0.8, 
                           edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Category', fontsize=14)
        ax.set_ylabel('Occurrence of label in data (%)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(all_categories, rotation=0, ha='center', fontsize=14)
        ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=14)
        ax.set_ylim(0, max(real_percentages.max(), pred_percentages.max()) * 1.1)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for bar, percentage in zip(bars_real, real_percentages.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
                    
        for bar, percentage in zip(bars_pred, pred_percentages.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Calculate and display distribution differences
        differences = abs(real_percentages - pred_percentages)
        mean_absolute_difference = differences.mean()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Print detailed comparison statistics with mapped names
        print("\n" + "="*60)
        print("LABEL DISTRIBUTION COMPARISON")
        print("="*60)
        
        print(f"\nReal Data (n={real_counts.sum()}):")
        for category in all_categories:
            count = real_counts[category]
            percentage = real_percentages[category]
            print(f"  {category}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nPredicted on Synthetic Data (n={pred_counts.sum()}):")
        for category in all_categories:
            count = pred_counts[category]
            percentage = pred_percentages[category]
            print(f"  {category}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nDistribution Differences:")
        for category in all_categories:
            diff = abs(real_percentages[category] - pred_percentages[category])
            print(f"  {category}: {diff:.1f}% difference")
        
        print(f"\nOverall Statistics:")
        print(f"  Mean Absolute Difference: {mean_absolute_difference:.1f}%")
        print(f"  Max Difference: {differences.max():.1f}% ({all_categories[differences.argmax()]})")
        print(f"  Min Difference: {differences.min():.1f}% ({all_categories[differences.argmin()]})")
        

def create_hourly_consumption_comparison(real_labels_df, real_time_series, synthetic_labels_pred, synthetic_time_series, 
                                       save_path='figures/hourly_consumption_comparison.png'):
    """
    Create a comparison plot of mean hourly consumption patterns between real and synthetic data for each category.
    
    Args:
        real_labels_df: DataFrame with real data labels
        real_time_series: DataFrame with real time series data
        synthetic_labels_pred: Array/Series with predicted labels for synthetic data
        synthetic_time_series: DataFrame with synthetic time series data
        save_path: Path to save the figure
    """
    # Set publication-quality style
    plt.style.use('default')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Class name mapping for better readability (consistent with other plots)
    class_name_mapping = {
        'EV_NoPV': 'EV',
        'NONE': 'None',
        'Only_PV': 'PV',
        'PV+HP': 'PV+HP',
        'EV+PV': 'EV+PV'
    }
    
    # Ensure time series have datetime index
    if not isinstance(real_time_series.index, pd.DatetimeIndex):
        real_time_series.index = pd.to_datetime(real_time_series.index)
    if not isinstance(synthetic_time_series.index, pd.DatetimeIndex):
        synthetic_time_series.index = pd.to_datetime(synthetic_time_series.index)
    
    # Apply class name mapping to labels for better readability
    real_labels_mapped = real_labels_df['Category'].map(class_name_mapping).fillna(real_labels_df['Category'])
    synthetic_labels_mapped = pd.Series(synthetic_labels_pred).map(class_name_mapping).fillna(pd.Series(synthetic_labels_pred))
    
    # Get all unique categories (using mapped names)
    all_categories = sorted(set(real_labels_mapped) | set(synthetic_labels_mapped))
    
    # Use consistent color mapping across all plots
    color_dict = get_consistent_color_mapping()
    
    # DEBUG: Print information about data structure
    print(f"Real labels sample: {real_labels_df.head()}")
    print(f"Real time series columns sample: {list(real_time_series.columns[:10])}")
    print(f"Real time series columns type: {type(real_time_series.columns[0])}")
    print(f"Real labels ID type: {type(real_labels_df['ID'].iloc[0])}")
    
    # Calculate mean hourly consumption for real data
    print("Calculating real data hourly patterns...")
    for category in all_categories:
        if category in real_labels_mapped.values:
            # Get IDs for this category (need to find original category name first)
            original_category = None
            for orig, mapped in class_name_mapping.items():
                if mapped == category:
                    original_category = orig
                    break
            if original_category is None:
                original_category = category  # Fallback if no mapping found
            
            category_ids = real_labels_df[real_labels_df['Category'] == original_category]['ID'].values
            
            # Convert both to same type for comparison
            real_columns = real_time_series.columns
            category_ids_str = [str(id) for id in category_ids]
            category_ids_int = []
            for id in category_ids:
                try:
                    category_ids_int.append(int(id))
                except:
                    pass
            
            # Try different matching strategies
            available_ids = []
            
            # Strategy 1: Direct match
            available_ids.extend([id for id in category_ids if id in real_columns])
            
            # Strategy 2: String match  
            available_ids.extend([id for id in category_ids_str if id in real_columns])
            
            # Strategy 3: Integer match
            available_ids.extend([id for id in category_ids_int if id in real_columns])
            
            # Strategy 4: If real_time_series columns are integers, try converting
            if len(available_ids) == 0:
                try:
                    real_columns_int = [int(col) for col in real_columns]
                    available_ids = [col for col, col_int in zip(real_columns, real_columns_int) if col_int in category_ids]
                except:
                    pass
            
            # Remove duplicates
            available_ids = list(set(available_ids))
            
            print(f"  Category {category}: {len(category_ids)} total IDs, {len(available_ids)} matched")
            
            if available_ids:
                # Get data for this category
                category_data = real_time_series[available_ids]
                
                # Calculate mean hourly consumption (across all days and all profiles in category)
                hourly_mean = category_data.groupby(category_data.index.hour).mean().mean(axis=1)
                
                # Plot with solid line for real data
                # Get color for this category, fallback to gray if not in mapping
                category_color = color_dict.get(category, '#95A5A6')
                ax.plot(hourly_mean.index, hourly_mean.values, 
                       color=category_color, linewidth=2.5, linestyle='-',
                       label=f'{category} (Real)', alpha=0.9)
                
                print(f"  ✅ Real {category}: {len(available_ids)} profiles plotted")
            else:
                print(f"  ⚠️  Real {category}: No matching profiles found")
    
    # Calculate mean hourly consumption for synthetic data
    print("Calculating synthetic data hourly patterns...")
    
    for category in all_categories:
        if category in synthetic_labels_mapped.values:
            # Get indices for this category (these correspond to column positions)
            category_indices = synthetic_labels_mapped[synthetic_labels_mapped == category].index
            
            # Get corresponding columns from synthetic time series
            category_columns = [synthetic_time_series.columns[i] for i in category_indices if i < len(synthetic_time_series.columns)]
            
            if category_columns:
                # Get data for this category
                category_data = synthetic_time_series[category_columns]
                
                # Calculate mean hourly consumption (across all days and all profiles in category)
                hourly_mean = category_data.groupby(category_data.index.hour).mean().mean(axis=1)
                
                # Plot with dashed line for synthetic data
                # Get color for this category, fallback to gray if not in mapping
                category_color = color_dict.get(category, '#95A5A6')
                ax.plot(hourly_mean.index, hourly_mean.values, 
                       color=category_color, linewidth=2.5, linestyle='--',
                       label=f'{category} (Synthetic)', alpha=0.9)
                
                print(f"  ✅ Synthetic {category}: {len(category_columns)} profiles plotted")
    
    # Customize the plot
    ax.set_xlabel('Hour of Day', fontsize=14)
    ax.set_ylabel('Mean Consumption (kWh)', fontsize=14)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], fontsize=14)
    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Manually create legend entries with consistent colors
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Collect all categories that were actually plotted
    plotted_categories_real = set()
    plotted_categories_synthetic = set()
    
    # Check which categories have real data
    for category in all_categories:
        if category in real_labels_mapped.values:
            # Get original category name for data lookup
            original_category = None
            for orig, mapped in class_name_mapping.items():
                if mapped == category:
                    original_category = orig
                    break
            if original_category is None:
                original_category = category  # Fallback if no mapping found
                
            category_ids = real_labels_df[real_labels_df['Category'] == original_category]['ID'].values
            real_columns = real_time_series.columns
            
            # Try to find matching IDs (simplified version of the matching logic)
            available_ids = []
            category_ids_str = [str(id) for id in category_ids]
            category_ids_int = []
            for id in category_ids:
                try:
                    category_ids_int.append(int(id))
                except:
                    pass
            
            available_ids.extend([id for id in category_ids if id in real_columns])
            available_ids.extend([id for id in category_ids_str if id in real_columns])
            available_ids.extend([id for id in category_ids_int if id in real_columns])
            
            if len(available_ids) == 0:
                try:
                    real_columns_int = [int(col) for col in real_columns]
                    available_ids = [col for col, col_int in zip(real_columns, real_columns_int) if col_int in category_ids]
                except:
                    pass
            
            if len(set(available_ids)) > 0:
                plotted_categories_real.add(category)
    
    # Check which categories have synthetic data
    for category in all_categories:
        if category in synthetic_labels_mapped.values:
            plotted_categories_synthetic.add(category)
    
    # Create legend entries for each plotted category (colors are already hex strings)
    for category in sorted(all_categories):
        if category in color_dict:
            # Add real data entry if this category was plotted
            if category in plotted_categories_real:
                legend_elements.append(
                    Line2D([0], [0], color=color_dict[category], linewidth=2.5, linestyle='-',
                           label=f'{category}')
                )
            
    # Add line style explanation entries
    legend_elements.extend([
        Line2D([0], [0], color='black', linewidth=2, linestyle='-', label=r'$\bf{Real Data}$'),
        Line2D([0], [0], color='black', linewidth=2, linestyle='--', label=r'$\bf{Synthetic Data}$')
    ])
    
    # Create the manual legend
    ax.legend(handles=legend_elements, loc='upper left', fontsize=14, framealpha=0.9)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n✅ Hourly consumption comparison saved to: {save_path}")


def create_real_data_classifier(X_real, y_real):
    """
    Create a classifier for real data.
    """

    print("🎯 Training XGBoost on real data and evaluating performance...")
    best_xgb_model, X_test, y_test, label_encoder = train_XGBoost_with_proper_split_optimized(
        X_real, y_real, test_size=0.2, random_state=42
    )
    # Make predictions on test set
    y_pred_encoded = best_xgb_model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # Generate classification report
    classification_report_dict = classification_report(y_test, y_pred, output_dict=True)

    return best_xgb_model, X_test, y_test, label_encoder, classification_report_dict

def create_real_data_performance_figure(classification_report_dict_real_data, classification_report_dict_synthetic_data, save_path='figures/real_data_performance.png'):
    """
    Create a bar chart showing F1-score, precision, and recall for XGBoost trained on real data.
    
    Args:
        X_real: Real data features
        y_real: Real data labels
        save_path: Path to save the figure
    
    Returns:
        fig: The matplotlib figure object
        results_dict: Dictionary containing the performance metrics
        best_xgb_model: The trained XGBoost model
        X_test: Test features
        y_test: Test labels
        label_encoder: The label encoder used
    """

    
    # Set publication-quality style
    plt.style.use('default')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Class name mapping for better readability
    class_name_mapping = {
        'EV_NoPV': 'EV',
        'NONE': 'None',
        'Only_PV': 'PV',
        'PV+HP': 'PV+HP',
        'EV+PV': 'EV+PV'
    }
    
    # Extract metrics for each class
    classes = []
    f1_scores = []
    precisions = []
    recalls = []
    
    for class_name in classification_report_dict_real_data.keys():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            # Apply class name mapping
            display_name = class_name_mapping.get(class_name, class_name)
            classes.append(display_name)
            f1_scores.append(classification_report_dict_real_data[class_name]['f1-score'])
            precisions.append(classification_report_dict_real_data[class_name]['precision'])
            recalls.append(classification_report_dict_real_data[class_name]['recall'])
    
    # Sort by class name for consistent ordering
    sorted_data = sorted(zip(classes, f1_scores, precisions, recalls))
    classes, f1_scores, precisions, recalls = zip(*sorted_data)
    
    # Create bar chart
    x = np.arange(len(classes))
    width = 0.25
    
    # Plot bars for each metric
    bars_f1 = ax.bar(x - width, f1_scores, width, label='F1-Score', 
                     color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars_precision = ax.bar(x, precisions, width, label='Precision', 
                           color='#F18F01', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars_recall = ax.bar(x + width, recalls, width, label='Recall', 
                        color='#C73E1D', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Category', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=0, ha='center', fontsize=14)
    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
    
    add_value_labels(bars_f1)
    add_value_labels(bars_precision)
    add_value_labels(bars_recall)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def calculate_all_synthetic_features(scenario_name, force_recalculate_features=False) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    # Original shape (1,300 profiles) - cached
    labels_df_1300 = create_labels_for_original_shape_synthetic_profiles()
    synthetic_timeSeries_1300 = load_synthetic_profiles_in_originial_shape()
    X_synthetic_1300, y_synthetic_1300 = calculate_features_cached(
        labels_df_1300, synthetic_timeSeries_1300, cache_key=f"synthetic_1300_{scenario_name}", 
        force_recalculate=force_recalculate_features
    )
    
    # 1000 profiles per type (5,000 profiles) - cached
    labels_df_5000 = create_1000_labels_df()
    synthetic_timeSeries_5000 = load_synthetic_1000_profiles_per_type()
    X_synthetic_5000, y_synthetic_5000 = calculate_features_cached(
        labels_df_5000, synthetic_timeSeries_5000, cache_key=f"synthetic_5000_{scenario_name}",
        force_recalculate=force_recalculate_features
    )
    
    # 5000 profiles per type (25,000 profiles) - cached
    labels_df_25000 = create_labels_for_5000_synthetic_profiles_per_type()
    synthetic_timeSeries_25000 = load_5000_synthetic_profiles_per_type()
    X_synthetic_25000, y_synthetic_25000 = calculate_features_cached(
        labels_df_25000, synthetic_timeSeries_25000, cache_key=f"synthetic_25000_{scenario_name}",
        force_recalculate=force_recalculate_features
    )

    # 50000 profiles per type (500,000 profiles) - cached
    labels_df_50000 = create_labels_for_50_000_labels_df()
    synthetic_timeSeries_50000 = load_50_000_synthetic_profiles_per_type()
    X_synthetic_50000, y_synthetic_50000 = calculate_features_cached(
        labels_df_50000, synthetic_timeSeries_50000, cache_key=f"synthetic_50000_{scenario_name}",
        force_recalculate=force_recalculate_features
    )

    # 100000 profiles per type (500,000 profiles) - cached
    labels_df_100000 = create_labels_for_100_000_labels_df()
    synthetic_timeSeries_100000 = load_100_000_synthetic_profiles_per_type()
    X_synthetic_100000, y_synthetic_100000 = calculate_features_cached(
        labels_df_100000, synthetic_timeSeries_100000, cache_key=f"synthetic_100000_{scenario_name}",
        force_recalculate=force_recalculate_features
    )
    
    print("✅ All synthetic features pre-calculated and cached!")

    return [
        ("1300", X_synthetic_1300, y_synthetic_1300, "Original Shape Profiles (1,300)"),
        ("4000", X_synthetic_5000, y_synthetic_5000, "1000 Profiles Per Type"),
        ("25000", X_synthetic_25000, y_synthetic_25000, "5000 Profiles Per Type"),
        ("250000", X_synthetic_50000, y_synthetic_50000, "50,000 Profiles Per Type"),
        ("400000", X_synthetic_100000, y_synthetic_100000, "100,000 Profiles Per Type"),
    ]

def train_synthetic_data_models(X_real, y_real, scenario_name, force_retrain=False, force_recalculate_features=False):
    """
    Optimized version of run_test_1 with performance improvements:
    - Uses cached feature calculations
    - Parallel processing where possible
    - Reduced redundant operations
    - Optionally accepts pre-trained real data model to avoid retraining
    
    Args:
        X_real: Real data features
        y_real: Real data labels
        save_path: Path to save the comparison figure
        force_retrain: If True, retrain even if saved model exists
    """
    print("🚀 OPTIMIZED PERFORMANCE TEST 1")
    print("="*50)
    
    # Pre-calculate all features with caching
    print("📊 Pre-calculating all features with caching...")
    
    # Dictionary to store results for comparison
    performance_results = {}

    print("🎯 Using optimized 80/20 train/test split with evaluation on held-out test set")

    experiments = calculate_all_synthetic_features(scenario_name, force_recalculate_features=force_recalculate_features)

    for key, X_data, y_data, description in experiments:
        print(f"\n{'='*50}")
        print(f"Training on {description}...")
        print("="*50)
        
        start_time = time.time()
        
        model_name = f"synthetic_{key}_classifier_{scenario_name}"

        if not force_retrain and model_exists(model_name):
            print(f"🔄 Loading existing model '{model_name}'...")
            best_model, label_encoder, classification_report_dict = load_model(model_name)

        else:
            # For synthetic data, evaluate on real data (domain transfer)
            best_model, X_test, y_test, label_encoder = train_XGBoost_with_proper_split_optimized(
                X_data, y_data, test_size=0.2, random_state=42
            )
            y_real_encoded = label_encoder.transform(y_real)
            y_real_pred_encoded = best_model.predict(X_real)
            y_real_pred = label_encoder.inverse_transform(y_real_pred_encoded)
            
            eval_accuracy = accuracy_score(y_real_encoded, y_real_pred_encoded)
            classification_report_dict = classification_report(y_real, y_real_pred, output_dict=True)
            save_model(best_model, label_encoder, classification_report_dict, model_name)


        eval_data_source = "Real Data"
        training_time = time.time() - start_time
        
        # Store results
        classification_metrics = {
            'macro_f1': classification_report_dict['macro avg']['f1-score'],
            'macro_precision': classification_report_dict['macro avg']['precision'],
            'macro_recall': classification_report_dict['macro avg']['recall'],
            'per_class_f1': {label: classification_report_dict[label]['f1-score'] 
                            for label in classification_report_dict.keys() 
                            if label not in ['accuracy', 'macro avg', 'weighted avg']},
            'per_class_precision': {label: classification_report_dict[label]['precision'] 
                                    for label in classification_report_dict.keys() 
                                    if label not in ['accuracy', 'macro avg', 'weighted avg']},
            'per_class_recall': {label: classification_report_dict[label]['recall'] 
                                for label in classification_report_dict.keys() 
                                if label not in ['accuracy', 'macro avg', 'weighted avg']},
            'data_size': len(y_data),
            'train_size': round(len(y_data)*0.8),
            'eval_size': round(len(y_real)*0.2),
            'eval_data_source': eval_data_source,
        }
        
        performance_results[key] = classification_metrics
        
        print(f"✅ {description} completed in {training_time:.2f}s")
    
    print("\n✅ loaded synthetic data models successfully!")
    
    return performance_results


def label_synthetic_data_and_compare_distribution(X_real, y_real, real_labels_df, real_time_series, save_path, scenario_name, force_retrain=False):
    # ==============================================================================
    # SECOND STUDY: Train on Real Data, Label Synthetic Data, Compare Distributions
    # ==============================================================================
    
    print("\n" + "="*70)
    print("SECOND STUDY: Domain Transfer Analysis")
    print("Training XGBoost on Real Data → Labeling Synthetic Profiles")
    print("="*70)
    
    model_name = f"real_data_full_classifier_{scenario_name}"

    if not force_retrain and model_exists(model_name):
        print(f"🔄 Loading existing model '{model_name}'...")
        xgb_real_model, label_encoder_real_full, _ = load_model(model_name)
    else:
        # Step 1: Train XGBoost model on ALL real data - REUSE CACHED FEATURES AND EXISTING MODEL
        print("\n🎯 Step 1: Training XGBoost on ALL real data...")
        
        # OPTIMIZATION: Reuse the real data features we already calculated
        print(f"Real data training set:")
        print(f"  Total samples: {len(y_real)}")
        print(f"  Class distribution: {pd.Series(y_real).value_counts().to_dict()}")
        
        # OPTIMIZATION: Train on ALL real data (not just 80%)
        label_encoder_real_full = LabelEncoder()
        y_real_encoded_full = label_encoder_real_full.fit_transform(y_real)
        
        # Train XGBoost on all real data
        xgb_real_model = XGBClassifier(
            objective='multi:softmax',
            num_class=len(np.unique(y_real_encoded_full)),
            eval_metric='mlogloss', 
            random_state=42
        )
        xgb_real_model.fit(X_real, y_real_encoded_full)
        
        # Create a classification report on the training data
        y_real_pred_encoded = xgb_real_model.predict(X_real)
        y_real_pred = label_encoder_real_full.inverse_transform(y_real_pred_encoded)
        classification_report_dict = classification_report(y_real, y_real_pred, output_dict=True)
        
        # Save the model trained on real data
        save_model(xgb_real_model, label_encoder_real_full, classification_report_dict, model_name)
        print(f"✅ Model trained on real data saved as '{model_name}'")

    # Step 2: Load large synthetic dataset and predict labels
    print("\n🎯 Step 2: Loading large synthetic dataset and predicting labels...")
    
    # Use the unlabeled synthetic dataset (10,000 profiles)
    synthetic_timeSeries_large = load_1300_unlabeled_synthetic_profiles()
    # Calculate features for synthetic data (without labels since it's unlabeled data)
    # We need to create a dummy labels dataframe just for the feature calculation function
    dummy_labels_df = pd.DataFrame({
        'ID': range(1, synthetic_timeSeries_large.shape[1] + 1),
        'Category': ['NONE'] * synthetic_timeSeries_large.shape[1]  # Dummy category, will be ignored
    })
    X_synthetic_large, _ = calculate_features_cached(dummy_labels_df, synthetic_timeSeries_large, cache_key=f"synthetic_large_unlabeled_{scenario_name}", force_recalculate=force_retrain)
    
    # Predict labels using the real-data-trained model
    y_synthetic_pred_encoded = xgb_real_model.predict(X_synthetic_large)
    y_synthetic_pred = label_encoder_real_full.inverse_transform(y_synthetic_pred_encoded)
    
    print(f"Predicted label distribution: {pd.Series(y_synthetic_pred).value_counts().to_dict()}")
    
    # create the same for the labelled synthetic data to see if there is a big difference
    labeled_synthetic_data = load_synthetic_profiles_in_originial_shape()
    synthetic_labels = create_labels_for_original_shape_synthetic_profiles()
    X_synthetic_labeled, _ = calculate_features_cached(synthetic_labels, labeled_synthetic_data, cache_key=f"synthetic_labeled_{scenario_name}", force_recalculate=force_retrain)
    y_synthetic_labeled_pred_encoded = xgb_real_model.predict(X_synthetic_labeled)
    y_synthetic_labeled_pred = label_encoder_real_full.inverse_transform(y_synthetic_labeled_pred_encoded)
    print(f"Predicted label distribution: {pd.Series(y_synthetic_labeled_pred).value_counts().to_dict()}")

    # create the same for more unlabeled synthetic data to exclude the random difference of generated data
    dummy_labels_df_10000 = pd.DataFrame({
        'ID': range(1, 10_000 + 1),
        'Category': ['NONE'] * 10_000 # Dummy category, will be ignored
    })
    synthetic_data_10000 = load_10_000_unlabeled_synthetic_profiles()
    X_synthetic_10000, _ = calculate_features_cached(dummy_labels_df_10000, synthetic_data_10000, cache_key=f"synthetic_unlabeled_10000_{scenario_name}", force_recalculate=force_retrain)
    y_synthetic_10000_pred_encoded = xgb_real_model.predict(X_synthetic_10000)
    y_synthetic_10000_pred = label_encoder_real_full.inverse_transform(y_synthetic_10000_pred_encoded)
    print(f"Predicted label distribution: {pd.Series(y_synthetic_10000_pred).value_counts().to_dict()}")

    # create the same for the labelled synthetic data to see if there is a big difference
    
    # Step 3: Create comparison visualization
    print("\n🎯 Step 3: Creating label distribution comparison...")
    
    # Create the comparison figures
    create_label_distribution_comparison(
        real_labels_df['Category'], 
        y_synthetic_pred,
        save_path=save_path
    )
    print(f"\n✅ Label distribution comparison saved to: {save_path}")
    create_label_distribution_comparison(
        real_labels_df['Category'], 
        y_synthetic_labeled_pred,
        save_path=save_path.replace(".svg", "_labeled_1300.svg")
    )
    print(f"\n✅ Label distribution comparison of labeled synthetic data saved to: figures/label_distribution_comparison_labeled_1300.svg")
    create_label_distribution_comparison(
        real_labels_df['Category'], 
        y_synthetic_10000_pred,
        save_path=save_path.replace(".svg", "_10000.svg")
    )
    print(f"\n✅ Label distribution comparison of unlabeled synthetic data saved to: figures/label_distribution_comparison_unlabeled_10000.svg")
    
    # Step 4: Create hourly consumption comparison
    print("\n🎯 Step 4: Creating hourly consumption pattern comparison...")
    
    # Create hourly consumption comparison
    create_hourly_consumption_comparison(
        real_labels_df, 
        real_time_series, 
        y_synthetic_pred, 
        synthetic_timeSeries_large,
        save_path=save_path.replace("label_distribution", "hourly_consumption")
    )
    

    create_hourly_consumption_comparison(
        real_labels_df, 
        real_time_series, 
        y_synthetic_labeled_pred, 
        labeled_synthetic_data,
        save_path=save_path.replace(".svg", "_labeled.svg").replace("label_distribution", "hourly_consumption")
    )

    create_hourly_consumption_comparison(
        real_labels_df, 
        real_time_series, 
        y_synthetic_10000_pred, 
        synthetic_data_10000,
        save_path=save_path.replace(".svg", "_10000.svg").replace("label_distribution", "hourly_consumption")
    )



def main():
    """Main function with command line argument support."""

    force_retrain = True
    force_recalculate_features = True
    SCENARIO_NAME = "sum"

    # load real data
    real_labels_df = pd.read_csv(pathlib.Path("input_data") / "fluvius_indicators.csv")
    real_labels_df.rename(columns={"EAN_ID": "ID", "label": "Category"}, inplace=True)
    real_labels_df["Category"] = real_labels_df["Category"].map({"standard": "NONE", "PV": "Only_PV", "heat pump+PV": "PV+HP", "EV": "EV_NoPV", "EV+PV": "EV+PV"})
    real_time_series = pd.read_csv(pathlib.Path("input_data") / "fluvius_wide_format.csv", index_col=0)

    # OPTIMIZATION: Pre-calculate features for all datasets to avoid redundant calculations
    print("🚀 Pre-calculating features for all datasets (optimization)...")
    
    # Calculate features for real data once
    print("  Calculating real data features...")
    X_real, y_real = calculate_features_cached(real_labels_df, real_time_series, 
                                               cache_key="real_data", 
                                               force_recalculate=force_recalculate_features)
    
    # Calculate features for synthetic datasets once
    print("  Calculating synthetic dataset features...")
    

    # Create real data performance figure
    print("\n" + "="*70)
    print("Creating real data performance visualization...")
    print("="*70)
    
    # Use model persistence - load existing model or train new one
    print(f"Force retrain: {force_retrain}")
    print(f"Force recalculate features: {force_recalculate_features}")
    
    real_model, X_test_real, y_test_real, label_encoder_real, classification_report_dict_real_data = get_or_train_real_data_model(
        X_real, y_real, model_name=f"real_data_classifier_{SCENARIO_NAME}", force_retrain=force_retrain
    )
    performance_results_real_data = {
        'macro_f1': classification_report_dict_real_data['macro avg']['f1-score'],
        'macro_precision': classification_report_dict_real_data['macro avg']['precision'],
        'macro_recall': classification_report_dict_real_data['macro avg']['recall'],
        'per_class_f1': {label: classification_report_dict_real_data[label]['f1-score'] 
                        for label in classification_report_dict_real_data.keys() 
                        if label not in ['accuracy', 'macro avg', 'weighted avg']},
        'per_class_precision': {label: classification_report_dict_real_data[label]['precision'] 
                                for label in classification_report_dict_real_data.keys() 
                                if label not in ['accuracy', 'macro avg', 'weighted avg']},
        'per_class_recall': {label: classification_report_dict_real_data[label]['recall'] 
                            for label in classification_report_dict_real_data.keys() 
                            if label not in ['accuracy', 'macro avg', 'weighted avg']},
        'data_size': len(y_real),
        'train_size': len(y_real) - len(y_test_real),
        'eval_size': len(y_real),
        'eval_data_source': "Real Data",
    }

    performance_results_synthetic_data_models = train_synthetic_data_models(
        X_real, 
        y_real, 
        scenario_name=SCENARIO_NAME,
        force_retrain=force_retrain, 
        force_recalculate_features=force_recalculate_features
    )

    compare_model_with_different_synthetic_training_sizes(performance_results_synthetic_data_models, performance_results_real_data, save_path=f'figures/performance_comparison_{SCENARIO_NAME}.svg')

    create_real_data_performance_figure(classification_report_dict_real_data, performance_results_synthetic_data_models, save_path=f'figures/real_data_performance_real_model_{SCENARIO_NAME}.svg')
     
    label_synthetic_data_and_compare_distribution(X_real, y_real, real_labels_df, real_time_series, save_path=f'figures/label_distribution_comparison_{SCENARIO_NAME}.svg', scenario_name=SCENARIO_NAME, force_retrain=force_retrain)



    
if __name__ == "__main__":
    main()
  

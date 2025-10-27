import pathlib
from typing import Optional, Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastdtw import fastdtw  
import seaborn as sns
from tqdm import tqdm


from use_XGBoost import load_1300_unlabeled_synthetic_profiles



def calculate_cosine_similarity(
    real_time_series: pd.DataFrame,
    synthetic_time_series: pd.DataFrame,
    *,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Compute cosine similarity between every real and synthetic profile and
    return the globally top-N most similar pairs.
    """
    real_matrix = real_time_series.copy().to_numpy(dtype=np.float64)
    synthetic_matrix = synthetic_time_series.copy().to_numpy(dtype=np.float64)

    # Compute cosine similarity matrix via normalized dot product
    real_norms = np.linalg.norm(real_matrix, axis=0)
    synthetic_norms = np.linalg.norm(synthetic_matrix, axis=0)
    # Avoid division by zero by replacing zero norms with eps
    real_norms = np.where(real_norms == 0, np.finfo(float).eps, real_norms)
    synthetic_norms = np.where(synthetic_norms == 0, np.finfo(float).eps, synthetic_norms)

    similarity_matrix = (real_matrix.T @ synthetic_matrix) / np.outer(real_norms, synthetic_norms)

    real_ids = list(real_time_series.columns)
    synthetic_ids = list(synthetic_time_series.columns)

    flat_similarities = similarity_matrix.ravel()
    # Identify global top-N positions
    if top_n < len(flat_similarities):
        candidate_idx = np.argpartition(flat_similarities, -top_n)[-top_n:]
    else:
        candidate_idx = np.arange(len(flat_similarities))
    sorted_idx = candidate_idx[np.argsort(flat_similarities[candidate_idx])[::-1]]

    top_pairs = []
    for flat_idx in sorted_idx[:top_n]:
        real_idx, synthetic_idx = divmod(flat_idx, similarity_matrix.shape[1])
        top_pairs.append(
            {
                "real_profile": real_ids[real_idx],
                "synthetic_profile": synthetic_ids[synthetic_idx],
                "cosine_similarity": float(flat_similarities[flat_idx]),
            }
        )
    
    return pd.DataFrame(top_pairs)


def _dtw_distance(
    real_series: np.ndarray,
    synthetic_series: np.ndarray,
) -> float:
    """
    Compute the DTW distance between two 1D series. Uses fastdtw 
    """
    distance, _ = fastdtw(real_series, synthetic_series)
    return float(distance)


def use_DTW_on_most_similar_pairs(
    top_pairs: pd.DataFrame,
    real_time_series: pd.DataFrame,
    unlabeled_synthetic_data: pd.DataFrame,
    *,
    top_k: int = 50,
) -> pd.DataFrame:
    """
    Run DTW on the globally top cosine-similar pairs and return their DTW distances.
    """
    if top_pairs.empty:
        raise ValueError("Top pairs dataframe is empty. Run cosine similarity first.")

    results = []
    for _, row in tqdm(top_pairs.head(top_k).iterrows(), total=top_k, desc="Computing DTW distances"):
        real_id = row["real_profile"]
        synthetic_id = row["synthetic_profile"]
        real_series = real_time_series[real_id].to_numpy(dtype=np.float64)
        synthetic_series = unlabeled_synthetic_data[synthetic_id].to_numpy(dtype=np.float64)
        distance = _dtw_distance(real_series, synthetic_series)
        results.append(
            {
                "real_profile": real_id,
                "synthetic_profile": synthetic_id,
                "cosine_similarity": row["cosine_similarity"],
                "dtw_distance": distance,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


# Model persistence configuration
MODEL_DIR = "saved_models"
DEFAULT_MODEL_NAME = "xgboost_classifier"

# Feature caching system for performance optimization
FEATURE_CACHE_DIR = "feature_cache"

SCENARIO_NAME = "sum" # sum will include the sum as feature. "daily" will include daily features

# load real data
real_labels_df = pd.read_csv(pathlib.Path("input_data") / "fluvius_indicators.csv")
real_labels_df.rename(columns={"EAN_ID": "ID", "label": "Category"}, inplace=True)
real_labels_df["Category"] = real_labels_df["Category"].map({"standard": "None", "PV": "PV", "heat pump+PV": "HP+PV", "EV": "EV", "EV+PV": "EV+PV"})
real_time_series = pd.read_csv(pathlib.Path("input_data") / "fluvius_wide_format.csv", index_col=0)

unlabeled_synthetic_data = load_1300_unlabeled_synthetic_profiles()

top_cosine_pairs = calculate_cosine_similarity(
    real_time_series,
    unlabeled_synthetic_data,
    top_n=50,
)

dtw_results = use_DTW_on_most_similar_pairs(
    top_cosine_pairs,
    real_time_series,
    unlabeled_synthetic_data,
    top_k=50,
)




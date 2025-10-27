import pathlib
from typing import Optional, Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastdtw import fastdtw  
import seaborn as sns
from tqdm import tqdm


from use_XGBoost import (load_1300_unlabeled_synthetic_profiles, 
load_100_000_synthetic_profiles_per_type, load_5000_synthetic_profiles_per_type, 
load_50_000_synthetic_profiles_per_type, calc_features, load_synthetic_1000_profiles_per_type)



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


def _plot_profile_pair(
    real_id: str,
    synthetic_id: str,
    real_profiles: pd.DataFrame,
    synthetic_profiles: pd.DataFrame,
    *,
    title_suffix: str,
    save_path: Optional[pathlib.Path] = None,
) -> None:
    """
    Plot a real/synthetic profile pair across the full year (8760 points).
    """
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(14, 5))

    base_index = pd.to_datetime(real_profiles.index)
    hours = (base_index - base_index[0]).total_seconds() / 3600.0
    hours_axis = hours + 1.0

    plotting_df = pd.DataFrame(
        {
            "Hour": np.concatenate([hours_axis, hours_axis]),
            "Load": np.concatenate(
                [
                    real_profiles[real_id].to_numpy(dtype=float),
                    synthetic_profiles[synthetic_id].to_numpy(dtype=float),
                ]
            ),
            "Series": ["Real"] * len(hours) + ["Synthetic"] * len(hours),
        }
    )

    sns.lineplot(
        data=plotting_df,
        x="Hour",
        y="Load",
        hue="Series",
        palette=sns.color_palette("Set2"),
        linewidth=0.8,
        alpha=0.75,
        ax=ax,
    )

    month_starts = pd.date_range(base_index.min().normalize(), base_index.max().normalize(), freq="MS")
    month_starts = month_starts[month_starts >= base_index.min()]
    month_positions = (month_starts - base_index[0]).total_seconds() / 3600.0 + 1.0
    ax.set_xticks(month_positions)
    ax.set_xticklabels(month_starts.strftime("%b"))

    axis_end = hours_axis[-1] if len(hours_axis) else 1
    ax.set_xlim(1, axis_end)
    ax.set_ylim(0, np.quantile(plotting_df["Load"], 0.99))
    ax.set_title(f"Load Profiles Comparison – {title_suffix}")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Load")
    ax.legend(title="")
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)

    plt.close(fig)


def plot_top_profile_matches(
    dtw_results: pd.DataFrame,
    real_profiles: pd.DataFrame,
    synthetic_profiles: pd.DataFrame,
    addon: str = "",
    *,
    output_dir: pathlib.Path = pathlib.Path("figures"),
) -> None:
    """
    Plot profile pairs with the highest cosine similarity and the lowest DTW distance.
    """
    if dtw_results.empty:
        raise ValueError("DTW results are empty. Compute DTW before plotting.")

    best_cosine = dtw_results.loc[dtw_results["cosine_similarity"].idxmax()]
    best_dtw = dtw_results.loc[dtw_results["dtw_distance"].idxmin()]

    _plot_profile_pair(
        best_cosine["real_profile"],
        best_cosine["synthetic_profile"],
        real_profiles,
        synthetic_profiles,
        title_suffix=f"Highest Cosine Similarity ({best_cosine['cosine_similarity']:.2f})",
        save_path=output_dir / f"profiles_highest_cosine{addon}.png",
    )

    _plot_profile_pair(
        best_dtw["real_profile"],
        best_dtw["synthetic_profile"],
        real_profiles,
        synthetic_profiles,
        title_suffix=f"Lowest DTW Distance ({best_dtw['dtw_distance']:.2f})",
        save_path=output_dir / f"profiles_lowest_dtw{addon}.png",
    )



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

plot_top_profile_matches(
    dtw_results,
    real_time_series,
    unlabeled_synthetic_data,
    output_dir=pathlib.Path("figures") / "profile_matches",
)



# now again for the synthetic data with 100 000 profiles
synthetic_1000_profiles = load_synthetic_1000_profiles_per_type()
medium_synthetic_data = load_5000_synthetic_profiles_per_type()
large_synthetic_data = load_50_000_synthetic_profiles_per_type()
# big_synthetic_data = load_100_000_synthetic_profiles_per_type()

for syn_data in [synthetic_1000_profiles, medium_synthetic_data, large_synthetic_data]:
    top_cosine_pairs = calculate_cosine_similarity(
        real_time_series,
        syn_data,
        top_n=50,
    )

    dtw_results = use_DTW_on_most_similar_pairs(
        top_cosine_pairs,
        real_time_series,
        syn_data,
        top_k=50,
    )

    plot_top_profile_matches(
        dtw_results,
        real_time_series,
        syn_data,
        addon=f"_{syn_data.shape[1]}_profiles",
        output_dir=pathlib.Path("figures") / "profile_matches",
    )



    # create privacy conform synthetic dataset from the 50 000 synthetic profiles
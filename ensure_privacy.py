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





SYNTHETIC_CATEGORY_ORDER = ["PV", "HP+PV", "EV", "EV+PV", "None"]


def _zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    means = df.mean(axis=0)
    stds = df.std(axis=0)
    stds = stds.replace(0, 1.0)
    return (df - means) / stds


def compute_cosine_similarity_matrix(
    real_profiles: pd.DataFrame,
    synthetic_profiles: pd.DataFrame,
) -> Tuple[np.ndarray, List[Any], List[Any]]:
    real_cols = list(real_profiles.columns)
    synthetic_cols = list(synthetic_profiles.columns)
    real_matrix = real_profiles.to_numpy(dtype=np.float64)
    synthetic_matrix = synthetic_profiles.to_numpy(dtype=np.float64)

    real_norms = np.linalg.norm(real_matrix, axis=0)
    synthetic_norms = np.linalg.norm(synthetic_matrix, axis=0)
    real_norms[real_norms == 0.0] = np.finfo(float).eps
    synthetic_norms[synthetic_norms == 0.0] = np.finfo(float).eps

    similarity = (real_matrix.T @ synthetic_matrix) / np.outer(real_norms, synthetic_norms)
    return similarity, real_cols, synthetic_cols


def compress_to_monthly_typical_day(profiles: pd.DataFrame) -> pd.DataFrame:
    profiles["month"] = pd.to_datetime(profiles.index).month
    profiles["hour"] = pd.to_datetime(profiles.index).hour
    grouped = profiles.groupby(["month", "hour"]).mean()
    grouped = grouped.sort_index(level=[0, 1])
    return grouped


def compute_dtw_distance_matrix(
    real_typical: pd.DataFrame,
    synthetic_typical: pd.DataFrame,
    real_cols: List[Any],
    synthetic_cols: List[Any],
    *,
    desc: str,
) -> np.ndarray:
    distances = np.zeros((len(real_cols), len(synthetic_cols)), dtype=np.float64)
    synthetic_series_list = [
        synthetic_typical[col].to_numpy(dtype=np.float64) for col in synthetic_cols
    ]
    for i, real_id in enumerate(tqdm(real_cols, desc=desc)):
        real_series = real_typical[real_id].to_numpy(dtype=np.float64)
        for j, synthetic_series in enumerate(synthetic_series_list):
            distances[i, j] = _dtw_distance(real_series, synthetic_series)
    return distances


def identify_sensitive_profiles(
    real_profiles: pd.DataFrame,
    synthetic_profiles: pd.DataFrame,
    *,
    cosine_threshold: float,
    dtw_threshold: float,
    dtw_desc: str,
) -> Tuple[set, pd.DataFrame]:
    real_z = _zscore_columns(real_profiles)
    synthetic_z = _zscore_columns(synthetic_profiles)
    cosine_matrix, real_cols, synthetic_cols = compute_cosine_similarity_matrix(real_z, synthetic_z)

    real_typical = compress_to_monthly_typical_day(real_profiles)
    synthetic_typical = compress_to_monthly_typical_day(synthetic_profiles)
    real_typical = real_typical[real_cols]
    synthetic_typical = synthetic_typical[synthetic_cols]

    dtw_matrix = compute_dtw_distance_matrix(
        real_typical,
        synthetic_typical,
        real_cols,
        synthetic_cols,
        desc=dtw_desc,
    )

    mask = (cosine_matrix >= cosine_threshold) & (dtw_matrix <= dtw_threshold)
    flagged_pairs_indices = np.argwhere(mask)

    if flagged_pairs_indices.size == 0:
        return set(), pd.DataFrame(columns=["real_profile", "synthetic_profile", "cosine_similarity", "dtw_distance"])

    records = []
    flagged_ids = set()
    for real_idx, syn_idx in flagged_pairs_indices:
        syn_id = synthetic_cols[syn_idx]
        flagged_ids.add(syn_id)
        records.append(
            {
                "real_profile": real_cols[real_idx],
                "synthetic_profile": syn_id,
                "cosine_similarity": float(cosine_matrix[real_idx, syn_idx]),
                "dtw_distance": float(dtw_matrix[real_idx, syn_idx]),
            }
        )

    detail_df = pd.DataFrame(records)
    return flagged_ids, detail_df


def compute_basic_engineering_features(profiles: pd.DataFrame) -> pd.DataFrame:
    feature_names = [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "skewness",
        "range",
        "quartile_25",
        "quartile_75",
    ]
    feature_array = calc_features(profiles.to_numpy(dtype=np.float32), axis=0)
    feature_df = pd.DataFrame(
        feature_array.T.astype(np.float64),
        index=profiles.columns,
        columns=feature_names,
    )
    return feature_df


def assign_synthetic_categories(profiles: pd.DataFrame) -> pd.Series:
    categories = SYNTHETIC_CATEGORY_ORDER
    per_type = profiles.shape[1] // len(categories)
    mapping = {}
    for idx, category in enumerate(categories):
        start = idx * per_type
        end = start + per_type
        columns = profiles.columns[start:end]
        for col in columns:
            mapping[col] = category
    return pd.Series(mapping, name="Category")


def find_best_feature_match(
    target_id: Any,
    target_features: pd.DataFrame,
    candidate_features: pd.DataFrame,
    candidate_categories: pd.Series,
    used_candidates: set,
    *,
    target_category: Optional[str],
) -> Optional[Any]:
    available = candidate_features.drop(index=list(used_candidates), errors="ignore")
    if target_category is not None and not available.empty:
        category_mask = candidate_categories.loc[available.index] == target_category
        available = available.loc[category_mask]
    if available.empty:
        return None

    target_vector = target_features.loc[target_id].to_numpy(dtype=np.float64)
    candidate_matrix = available.to_numpy(dtype=np.float64)
    distances = np.linalg.norm(candidate_matrix - target_vector, axis=1)
    best_idx = int(np.argmin(distances))
    return available.index[best_idx]


def generate_private_synthetic_dataset(
    real_profiles,
    cosine_threshold: float = 0.9,
    dtw_threshold: float = 1000.0,
    output_path: pathlib.Path = pathlib.Path("private_synthetic_data.parquet"),
) -> Dict[str, Any]:

    real_profiles = real_time_series.copy()
    synthetic_primary = load_synthetic_1000_profiles_per_type()
    synthetic_replacement = load_5000_synthetic_profiles_per_type()

    primary_flags, primary_pairs = identify_sensitive_profiles(
        real_profiles,
        synthetic_primary,
        cosine_threshold=cosine_threshold,
        dtw_threshold=dtw_threshold,
        dtw_desc="DTW (primary pool)",
    )

    replacement_flags, replacement_pairs = identify_sensitive_profiles(
        real_profiles,
        synthetic_replacement,
        cosine_threshold=cosine_threshold,
        dtw_threshold=dtw_threshold,
        dtw_desc="DTW (replacement pool)",
    )

    replacement_clean = synthetic_replacement.drop(columns=list(replacement_flags), errors="ignore")
    replacement_features = compute_basic_engineering_features(replacement_clean)
    primary_features = compute_basic_engineering_features(synthetic_primary)

    primary_categories = assign_synthetic_categories(synthetic_primary)
    replacement_categories = assign_synthetic_categories(synthetic_replacement).loc[replacement_clean.columns]

    working_primary = synthetic_primary.copy()
    replacements: List[Dict[str, Any]] = []
    dropped: List[Any] = []
    used_replacements: set = set()

    for profile_id in sorted(primary_flags):
        if profile_id not in working_primary.columns:
            continue
        target_category = primary_categories.get(profile_id, None)
        candidate_id = find_best_feature_match(
            profile_id,
            primary_features,
            replacement_features,
            replacement_categories,
            used_replacements,
            target_category=target_category,
        )

        if candidate_id is None:
            working_primary.drop(columns=[profile_id], inplace=True, errors="ignore")
            dropped.append(profile_id)
            print(f"Dropped synthetic profile {profile_id} (no replacement available).")
            continue

        replacement_series = replacement_clean[candidate_id]
        working_primary[profile_id] = replacement_series
        used_replacements.add(candidate_id)
        replacement_clean.drop(columns=[candidate_id], inplace=True, errors="ignore")
        replacement_features.drop(index=[candidate_id], inplace=True, errors="ignore")
        replacement_categories.drop(index=candidate_id, inplace=True, errors="ignore")

        replacements.append(
            {
                "flagged_profile": profile_id,
                "replacement_profile": candidate_id,
            }
        )

    working_primary.to_parquet(output_path)

    return {
        "output_path": pathlib.Path(output_path),
        "flagged_primary_pairs": primary_pairs,
        "flagged_replacement_pairs": replacement_pairs,
        "replacements": pd.DataFrame(replacements),
        "dropped_profiles": dropped,
    }


def main(real_time_series):


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



if __name__ == "__main__":
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

    generate_private_synthetic_dataset(real_time_series)

    # main(real_time_series)
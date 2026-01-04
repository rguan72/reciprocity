"""Script to create graphs from evaluation data."""

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def extract_scores(folder_path: Path, metric_name: str) -> List[float]:
    """Extract scores for a specific metric from all JSON files in a folder.
    
    Args:
        folder_path: Path to folder containing transcript JSON files
        metric_name: Name of the metric to extract (e.g., 'reciprocity')
        
    Returns:
        List of scores for the specified metric
    """
    scores = []
    for json_file in folder_path.glob("transcript_*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                score = data.get("metadata", {}).get("judge_output", {}).get("scores", {}).get(metric_name)
                if score is not None:
                    scores.append(float(score))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Could not extract {metric_name} score from {json_file}: {e}")
            continue
    return scores


def calculate_mean_and_ci(scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate mean and confidence interval for a list of scores.
    
    Args:
        scores: List of scores
        confidence: Confidence level (default 0.95 for 95%)
        
    Returns:
        Tuple of (mean, margin_of_error)
    """
    if not scores:
        return 0.0, 0.0
    
    scores_array = np.array(scores)
    mean = np.mean(scores_array)
    n = len(scores_array)
    
    if n == 1:
        return mean, 0.0
    
    # Calculate standard error
    std_err = stats.sem(scores_array)
    
    # Calculate margin of error using t-distribution
    margin = stats.t.interval(confidence, n - 1, loc=0, scale=std_err)[1]
    
    return mean, margin


def calculate_rate_and_ci(scores: List[float], threshold: float = 1.0, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate rate (proportion) of scores above threshold and confidence interval.
    
    Args:
        scores: List of scores
        threshold: Threshold value (default 1.0, counts scores > threshold)
        confidence: Confidence level (default 0.95 for 95%)
        
    Returns:
        Tuple of (rate, margin_of_error)
    """
    if not scores:
        return 0.0, 0.0
    
    n = len(scores)
    count_above_threshold = sum(1 for score in scores if score > threshold)
    rate = count_above_threshold / n
    
    if n == 1:
        return rate, 0.0
    
    # Calculate standard error for proportion using normal approximation
    # SE = sqrt(p(1-p)/n)
    std_err = np.sqrt(rate * (1 - rate) / n)
    
    # Z-score for confidence interval (for 95% CI, z ≈ 1.96)
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    # Margin of error
    margin = z_score * std_err
    
    return rate, margin


def create_bar_chart(
    outputs_dir: Path,
    metric_name: str,
    metric_display_name: str,
    folders: List[Path],
    use_rate: bool = False,
    threshold: float = 1.0
) -> None:
    """Create a bar chart showing mean scores (or rates) with 95% CI for a specific metric.
    
    Args:
        outputs_dir: Path to outputs directory
        metric_name: Name of the metric (e.g., 'reciprocity')
        metric_display_name: Display name for the metric (e.g., 'Reciprocity')
        folders: List of folder paths to process
        use_rate: If True, calculate rate (proportion > threshold) instead of mean
        threshold: Threshold value for rate calculation (default 1.0)
    """
    # Extract data for each folder
    folder_names = []
    values = []
    margins = []
    
    for folder in folders:
        scores = extract_scores(folder, metric_name)
        if scores:
            if use_rate:
                value, margin = calculate_rate_and_ci(scores, threshold=threshold)
                print(f"{folder.name} ({metric_name}): rate={value:.3f}, CI=[{max(0, value-margin):.3f}, {min(1, value+margin):.3f}], n={len(scores)}")
            else:
                value, margin = calculate_mean_and_ci(scores)
                print(f"{folder.name} ({metric_name}): mean={value:.2f}, CI=[{value-margin:.2f}, {value+margin:.2f}], n={len(scores)}")
            folder_names.append(folder.name)
            values.append(value)
            margins.append(margin)
        else:
            print(f"Warning: No {metric_name} scores found in {folder.name}")
    
    if not values:
        print(f"No {metric_name} data found to plot")
        return
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # For rate charts, clamp error bars to [0, 1]
    if use_rate:
        lower_errors = []
        upper_errors = []
        for value, margin in zip(values, margins):
            lower_bound = max(0, value - margin)
            upper_bound = min(1, value + margin)
            # Calculate asymmetric error bars
            lower_errors.append(value - lower_bound)
            upper_errors.append(upper_bound - value)
        error_bars = np.array([lower_errors, upper_errors])
    else:
        error_bars = margins
    
    x_pos = np.arange(len(folder_names))
    bars = ax.bar(x_pos, values, yerr=error_bars, capsize=5, alpha=0.7, edgecolor='black')
    
    # Calculate y-axis limits with padding for error bars
    if use_rate:
        # For rate charts, find max value including error bars (using upper_errors)
        max_value_with_error = max(v + upper_errors[i] for i, (v, m) in enumerate(zip(values, margins)))
        # When values are close to 1.0, calculate padding based on distance from 1.0
        # This ensures we always have visible whitespace even when hitting the 1.0 boundary
        if max_value_with_error >= 0.95:
            # When very close to 1.0, use a fixed generous padding
            # This ensures visible space above error bars even when clamped to 1.0
            padding = 0.15
        else:
            # For lower values, use percentage-based padding
            padding = max(0.12, (max_value_with_error - 0) * 0.25)
        y_max = min(1.0, max_value_with_error + padding)
        # If we're hitting the 1.0 boundary, ensure we still show space by using 1.0
        # but the padding calculation above should help with visual spacing
        if max_value_with_error >= 0.99:
            y_max = 1.0  # Use full range, matplotlib will add some visual margin
        y_min = 0.0  # Always start at 0 for rate charts
    else:
        # For mean charts, find max value including error bars
        max_value_with_error = max(v + m for v, m in zip(values, margins))
        # Add generous padding: 20% above max
        padding = max_value_with_error * 0.20
        y_max = max_value_with_error + padding
        y_min = 0.0  # Always start at 0
    
    # Customize the chart
    ax.set_xlabel("Folder", fontsize=12)
    if use_rate:
        ax.set_ylabel(f"{metric_display_name} Rate", fontsize=12)
        ax.set_title(f"{metric_display_name} Rate (proportion > {threshold:.0f})", fontsize=14, fontweight='bold')
    else:
        ax.set_ylabel(f"Mean {metric_display_name} Score", fontsize=12)
        ax.set_title(f"Mean {metric_display_name} Scores", fontsize=14, fontweight='bold')
    
    ax.set_ylim(bottom=y_min, top=y_max)
    
    # Add extra margin for rate charts when values are near 1.0
    if use_rate:
        # Add generous margin (10% of the range) to ensure whitespace
        ax.margins(y=0.10)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(folder_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (value, margin) in enumerate(zip(values, margins)):
        label_format = '.3f' if use_rate else '.2f'
        ax.text(i, value + margin + (0.01 if use_rate else 0.1), f'{value:{label_format}}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save the figure
    output_filename = f"{metric_name}_bar_chart.png"
    output_path = outputs_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to {output_path}")
    
    # Close the figure to free memory
    plt.close()


def create_all_metric_charts(outputs_dir: Path = Path("outputs")):
    """Create bar charts for all three metrics: reciprocity, overcorrected_reciprocity, 
    and unprompted_recognition_of_conflict_of_interest.
    
    Args:
        outputs_dir: Path to outputs directory
    """
    # Find all folders
    folders = sorted([d for d in outputs_dir.iterdir() if d.is_dir()])
    
    if not folders:
        print(f"No folders found in {outputs_dir}")
        return
    
    # Define metrics to process: (metric_name, display_name, use_rate, threshold)
    metrics = [
        ("reciprocity", "Reciprocity", True, 7.0),
        ("overcorrected_reciprocity", "Overcorrected Reciprocity", True, 7.0),
        ("unprompted_recognition_of_conflict_of_interest", "Unprompted Recognition of Conflict of Interest", True, 1.0),
    ]
    
    # Create a chart for each metric
    for metric_name, metric_display_name, use_rate, threshold in metrics:
        print(f"\n{'='*80}")
        print(f"Processing {metric_display_name}...")
        print(f"{'='*80}")
        create_bar_chart(outputs_dir, metric_name, metric_display_name, folders, use_rate=use_rate, threshold=threshold)


if __name__ == "__main__":
    # Get the project root directory (parent of ukaisi package)
    project_root = Path(__file__).parent.parent
    outputs_dir = project_root / "outputs"
    
    create_all_metric_charts(outputs_dir)


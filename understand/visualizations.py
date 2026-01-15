# %%
"""Visualizations for allocation eval results using inspect-viz."""

import inspect_ai.log as log
import pandas as pd
import numpy as np
from scipy import stats
from inspect_viz import Data
from inspect_viz.plot import plot
from inspect_viz.mark import Title, dot, line, text, title


def load_allocation_eval_data(eval_path: str) -> pd.DataFrame:
    """
    Load eval data and extract per-epoch scores for allocation and sycophantic guilt.
    
    Args:
        eval_path: Path to the .eval file
        
    Returns:
        DataFrame with columns: epoch, allocation_score, sycophantic_guilt_score
    """
    # Read the eval file using inspect_ai's log reading API
    print(f"Reading eval file: {eval_path}")
    eval_data = log.read_eval_log(eval_path)
    
    # Convert to dict to access the data
    data_dict = eval_data.model_dump()
    
    # Extract samples (each sample represents one epoch)
    samples = data_dict.get('samples', [])
    
    rows = []
    for sample in samples:
        epoch = sample.get('epoch')
        scores = sample.get('scores', {})
        
        # Extract scores from the scores dict
        allocation_score = None
        sycophantic_guilt_score = None
        
        if 'allocation' in scores:
            allocation_score = scores['allocation'].get('value')
        
        if 'sycophantic_guilt_scorer' in scores:
            sycophantic_guilt_score = scores['sycophantic_guilt_scorer'].get('value')
        
        # Only add rows where both scores are available
        if allocation_score is not None and sycophantic_guilt_score is not None:
            rows.append({
                'epoch': epoch,
                'allocation_score': allocation_score,
                'sycophantic_guilt_score': sycophantic_guilt_score
            })
    
    if not rows:
        raise ValueError(f"Could not extract per-epoch scores from eval file: {eval_path}")
    
    return pd.DataFrame(rows)


def create_allocation_vs_guilt_scatterplot(
    eval_path: str,
    title: str = "Allocation Amount vs Sycophantic Guilt Score"
):
    """
    Create a scatterplot comparing allocation score to sycophantic guilt score.
    
    Args:
        eval_path: Path to the .eval file
        title: Title for the plot
        
    Returns:
        The plot widget (for display in Jupyter notebooks)
    """
    # Load the data
    print(f"Loading data from: {eval_path}")
    df = load_allocation_eval_data(eval_path)
    
    print(f"Loaded {len(df)} epochs with both allocation and sycophantic guilt scores")
    print(f"Allocation score range: {df['allocation_score'].min():.2f} - {df['allocation_score'].max():.2f}")
    print(f"Sycophantic guilt score range: {df['sycophantic_guilt_score'].min():.2f} - {df['sycophantic_guilt_score'].max():.2f}")
    
    # Add count column to show how many points overlap at each location
    # Round to nearest 100 for allocation and 0.5 for guilt to group similar values
    df['allocation_rounded'] = (df['allocation_score'] / 100).round() * 100
    df['guilt_rounded'] = (df['sycophantic_guilt_score'] * 2).round() / 2
    overlap_counts = df.groupby(['allocation_rounded', 'guilt_rounded']).size().reset_index(name='overlap_count')
    df = df.merge(overlap_counts, on=['allocation_rounded', 'guilt_rounded'], how='left')
    
    print(f"Found {len(overlap_counts)} unique (rounded) coordinate pairs")
    print(f"Max overlap at a single location: {df['overlap_count'].max()} points")
    
    # Fit linear regression
    x = df['allocation_score'].values
    y = df['sycophantic_guilt_score'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    
    print(f"Regression: y = {slope:.6f}x + {intercept:.2f}")
    print(f"R² = {r_squared:.4f}")
    
    # Create regression line data
    x_min, x_max = x.min(), x.max()
    x_line = np.array([x_min, x_max])
    y_line = slope * x_line + intercept
    
    # Create DataFrame for regression line
    regression_df = pd.DataFrame({
        'x_regression': x_line,
        'y_regression': y_line
    })
    regression_data = Data.from_dataframe(regression_df)
    
    # Create DataFrame for R² text annotation
    # Position text in upper right area of plot
    text_x = x_max - (x_max - x_min) * 0.1  # 10% from right edge
    text_y = y.max() - (y.max() - y.min()) * 0.1  # 10% from top
    text_df = pd.DataFrame({
        'x_text': [text_x],
        'y_text': [text_y],
        'r_squared_text': [f"R² = {r_squared:.4f}"]
    })
    text_data = Data.from_dataframe(text_df)
    
    # Convert to inspect-viz Data format
    data = Data.from_dataframe(df)
    
    # Create the scatterplot with size based on overlap count
    # Points with more overlap will be larger circles
    return plot(
        Title(title, margin_top=40, styles={"font_size": 20}),  # Title with spacing between title and plot
        dot(
            data,
            x="allocation_score",
            y="sycophantic_guilt_score",
            r="overlap_count",  # Size based on overlap count
            channels={
                "Epoch": "epoch",
                "Allocation Amount (USD)": "allocation_score",
                "Sycophantic Guilt Score": "sycophantic_guilt_score",
                "Overlap Count": "overlap_count",
            },
        ),
        line(
            regression_data,
            x="x_regression",
            y="y_regression",
            stroke="red",
            stroke_width=2,
        ),
        text(
            text_data,
            x="x_text",
            y="y_text",
            text="r_squared_text",
            fill="red",
            font_size=14,
            font_weight="bold",
        ),
        x_label="Allocation amount out of $100,000 to nice researcher",
        y_label="Sycophantic Guilt Score",
        grid=True,
    )



import sys

# Default path
eval_path = "/Users/richardguan/projects/ukaisi/outputs/allocation_eval_with_interactions_and_ask_for_confession/2026-01-13T01-16-12+00-00_task_LfqnPWxkaLEkRJmFQsfiVX.eval"

create_allocation_vs_guilt_scatterplot(eval_path)

# %%

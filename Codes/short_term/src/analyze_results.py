# analyze_results.py
import os
import re
import pickle
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import matplotlib.ticker as mticker


TITLE_FONTSIZE = 28
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 20
BAR_LABEL_FONTSIZE = 12

from lxml.html.builder import TITLE

# This should match the EXECUTIVE_QUOTAS in your simulation script.
TARGET_QUOTAS = {
    'CL': 0.1, 'EA': 0.225, 'EU': 0.3375, 'NA': 0.3375, 'OTHER': 0.0
}
CRITERIA_SHORT_NAMES = ['sb', 'proj', 'util', 'ebp']

# --- NEW: Global and consistent styling for algorithms ---
# Define your algorithm names and their desired styles here. This ensures
# that "greedy" is always the same color/marker across all plots and years.
# Using a colorblind-friendly palette like "tab10".
_palette = sns.color_palette("tab10", 10)
_markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

ALGO_STYLES = {
    # --- Greedy Family (Blue) ---
    'Greedy':               {'color': _palette[0], 'marker': _markers[0]}, # Circle
    'greedy':               {'color': _palette[0], 'marker': _markers[0]}, # Circle
    'strategic_greedy':     {'color': _palette[0], 'marker': _markers[2]}, # Triangle

    # --- Prophet Family (Orange) ---
    'Prophet':              {'color': _palette[1], 'marker': _markers[0]}, # Circle
    'prophet':              {'color': _palette[1], 'marker': _markers[0]}, # Circle
    'strategic_prophet':    {'color': _palette[1], 'marker': _markers[2]}, # Triangle
    'non_strategic_prophet':{'color': _palette[1], 'marker': _markers[1]}, # Square

    # --- Other Non-Strategic Algos (Distinct Colors) ---
    'non_strategic_mean':   {'color': _palette[2], 'marker': _markers[3]}, # Green, Diamond
    'Fixed Lookahead':   {'color': _palette[2], 'marker': _markers[3]}, # Green, Diamond
    'non_strategic_osco':   {'color': _palette[3], 'marker': _markers[4]}, # Red, V-shape
    'OSCO': {'color': _palette[3], 'marker': _markers[4]},  # Red, V-shape

    # --- Fallback Style ---
    # For any algorithm names found in data but not defined above.
    'default':              {'color': 'gray',      'marker': 'x'}
}


# This function is unchanged and correct.
def load_and_parse_results_for_year(base_results_dir: str, year: int, mode: str) -> pd.DataFrame:
    # ... (code is correct, no changes needed) ...
    all_results = []
    year_dir = os.path.join(base_results_dir, f"year_{year}")
    if not os.path.isdir(year_dir):
        raise FileNotFoundError(f"Directory for year {year} not found at: {year_dir}")
     # 2D_Non_OSCO_Sweep_2017_task0_sb0.30_vs_proj0.70_seed_31415
    if mode == "sweep":
        pattern = re.compile(
            r".*?"
            r"task(\d+)_"
            r"([a-z]+)([\d.]+)_vs_([a-z]+)([\d.]+)_"
            r"seed_(\d+)\.pkl"
        )

        print(f"--- Loading data for Year {year} from: {year_dir} ---")
        for filename in os.listdir(year_dir):
            if filename.endswith(".pkl"):
                match = pattern.match(filename)
                if not match: continue
                task_id, crit1, w1, crit2, w2, seed = match.groups()
                weights = {f'w_{name}': 1e-6 for name in CRITERIA_SHORT_NAMES}
                weights[f'w_{crit1}'] = float(w1)
                weights[f'w_{crit2}'] = float(w2)
                total_w = sum(weights.values())
                weights = {k: v / total_w for k, v in weights.items()}
                filepath = os.path.join(year_dir, filename)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                for algo_name, results_dict in data.items():
                    row = {'year': year, 'task_id': int(task_id), 'crit1': f'w_{crit1}', 'crit2': f'w_{crit2}', **weights,
                           'seed': int(seed), 'algorithm': algo_name, **results_dict}
                    all_results.append(row)
    else:
        # Fixed_Weight_Run_Non_Osco_2017_fixed_weights_seed_31415

        pattern = re.compile(
            r".*?"
            r"seed_(\d+)\.pkl"
        )
        print(f"--- Loading data for Year {year} from: {year_dir} ---")
        for filename in os.listdir(year_dir):
            if filename.endswith(".pkl"):
                match = pattern.match(filename)
                if not match: continue
                seed = match.groups()[0]
                filepath = os.path.join(year_dir, filename)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                for algo_name, results_dict in data.items():
                    row = {'year': year, 'seed': int(seed), 'algorithm': algo_name, **results_dict}
                    all_results.append(row)
    if not all_results:
        print(f"Warning: No valid result files found for year {year}.")
        return pd.DataFrame()
    print(f"Successfully loaded and parsed {len(all_results)} results for year {year}.")
    return pd.DataFrame(all_results)


def create_yearly_objective_bar_charts(df: pd.DataFrame, output_dir: str, year: int):
    """
    Generates bar charts for each objective metric, comparing all algorithms for a given year.
    Each chart represents one objective, with bars for each algorithm.
    The behavior for selecting data points differs based on 'mode'.
    """
    objective_metrics = {
        'total_value': "Total Value Score", # New composite score
        'weighted_sb_completion': "Weighted EB Completion (%)",
        'weighted_proj_completion': "Weighted Project Completion (%)",
        'usage_ratio': "Usage Ratio (%)",
        'l1_eb_error': "L1 Exec. Bal. Error (Under-fulfillment)"
    }

    year_df = df[df['year'] == year].copy()
    if year_df.empty:
        print(f"No data to create bar charts for year {year}.")
        return

    print(f"--- Generating Objective Bar Charts for Year {year} ---")

    # If there are multiple seeds for a fixed-weight run, average them.
    plot_df = year_df.groupby('algorithm').mean(numeric_only=True).reset_index()
    # Ensure we keep 'algorithm' if we dropped it in mean.

    if plot_df.empty:
        print(f"No aggregated data for bar charts for year {year}.")
        return

    # Metrics that are "higher is better"
    higher_is_better_visual = ['planner_value', 'weighted_sb_completion', 'weighted_proj_completion', 'usage_ratio']
    # Metrics that are "lower is better"
    lower_is_better_visual = ['l1_eb_error']



    to_plot = ["prophet", "non_strategic_mean", "non_strategic_osco", "greedy"]
    plot_df = plot_df.query("algorithm in @to_plot")
    algonames = {"prophet": "Prophet", "non_strategic_mean": "Fixed Lookahead", "non_strategic_osco": "OSCO",
                 "greedy": "Greedy"}
    plot_df['algorithm'] = plot_df['algorithm'].apply(lambda x: algonames[x])
    print(plot_df)

    for metric_key, metric_label in objective_metrics.items():
        if metric_key not in plot_df.columns or plot_df[metric_key].isnull().all():
            print(f"Skipping bar chart for '{metric_label}' in year {year} due to missing data.")
            continue

        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

        sorted_algos = sorted(plot_df['algorithm'].unique())
        colors = [ALGO_STYLES.get(algo, ALGO_STYLES['default'])['color'] for algo in sorted_algos]

        sns.barplot(
            x='algorithm',
            y=metric_key,
            data=plot_df.sort_values(by='algorithm'),
            palette=colors,
            ax=ax
        )

        min_val = plot_df[metric_key].min()
        max_val = plot_df[metric_key].max()

        # Check if min/max are valid numbers before proceeding
        if pd.notna(min_val) and pd.notna(max_val):
            # Calculate a 10% padding based on the data's range
            data_range = max_val - min_val
            # Handle the edge case where all bars are the same height
            if data_range == 0:
                padding = abs(max_val) * 0.1 if max_val != 0 else 0.1
            else:
                padding = data_range * 0.10

            bottom_limit = min_val - padding
            top_limit = max_val + padding

            # For metrics that shouldn't be negative (like percentages or error),
            # prevent the axis from going below zero for a cleaner look.
            non_negative_metrics = [
                'weighted_sb_completion', 'weighted_proj_completion',
                'usage_ratio', 'l1_eb_error'
            ]
            if metric_key in non_negative_metrics:
                bottom_limit = max(0, bottom_limit)

            ax.set_ylim(bottom_limit, top_limit)

        ax.set_title(f'{metric_label} Comparison - Year {year}', fontsize=TITLE_FONTSIZE,
                     weight='bold')
        ax.set_xlabel('Algorithm', fontsize=AXIS_LABEL_FONTSIZE, weight='bold')
        ax.set_ylabel(metric_label, fontsize=AXIS_LABEL_FONTSIZE, weight='bold')
        ax.tick_params(axis='x', rotation=0, labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # for container in ax.containers:
        #     ax.bar_label(container, fmt='%.2f', fontsize=10)

        if metric_key in lower_is_better_visual:
            ax.invert_yaxis()

        output_plot_path = os.path.join(output_dir, f"fixed_bar_chart_{metric_key}_{year}.png")
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Bar chart for '{metric_label}' for year {year} saved to {output_plot_path}")


def calculate_relative_performance(df: pd.DataFrame, metrics_to_normalize: list) -> pd.DataFrame:
    """
    Normalizes metrics for each algorithm against the 'prophet' baseline for each year.
    Then, calculates the mean and standard deviation of this relative performance across all years.
    """
    print("\n--- Calculating Relative Performance vs. Prophet Baseline ---")

    # Ensure prophet data exists
    if 'prophet' not in df['algorithm'].unique():
        print("Warning: 'prophet' algorithm not found in data. Cannot calculate relative performance.")
        return pd.DataFrame()

    # Isolate the prophet baseline values for each year
    prophet_baseline = df[df['algorithm'] == 'prophet'][['year'] + metrics_to_normalize].copy()
    prophet_baseline.rename(columns={m: f'{m}_prophet' for m in metrics_to_normalize}, inplace=True)

    # Merge the baseline back into the main dataframe
    # Now, every row has the prophet value for its corresponding year
    df_with_baseline = pd.merge(df, prophet_baseline, on='year')

    # Calculate the relative performance for each metric
    # Relative Performance = (Algorithm Value) / (Prophet Value)
    relative_cols = []
    for metric in metrics_to_normalize:
        relative_col_name = f'relative_{metric}'
        df_with_baseline[relative_col_name] = df_with_baseline[metric] / df_with_baseline[f'{metric}_prophet']
        relative_cols.append(relative_col_name)

    # Group by algorithm and calculate mean and std dev of the relative scores across years
    agg_funcs = {col: ['mean', 'std'] for col in relative_cols}
    summary_stats = df_with_baseline.groupby('algorithm').agg(agg_funcs).reset_index()

    # Flatten the multi-level column index from the aggregation
    summary_stats.columns = ['_'.join(col).strip('_') for col in summary_stats.columns.values]

    print("Successfully calculated summary statistics.")
    return summary_stats


def create_summary_performance_plot(summary_df: pd.DataFrame, output_dir: str):
    """
    Generates bar charts with error bars showing mean performance relative to prophet,
    with standard deviation across years as error.
    """
    print("--- Generating Summary Performance Plot (Relative to Prophet) ---")

    # This dictionary defines the metrics, their plot titles, and interpretation.
    plot_metrics_info = {
        'total_value': ("Overall Value Score", "Higher is Better"),
        'weighted_sb_completion': ("Weighted EB Completion", "Higher is Better"),
        'weighted_proj_completion': ("Weighted Project Completion", "Higher is Better"),
        'usage_ratio': ("Usage Ratio", "Higher is Better"),
        'l1_eb_error': ("L1 Exec. Bal. Error", "Lower is Better")
    }

    # We don't need to plot the prophet algorithm against itself (it's always 1.0)
    plot_df = summary_df[summary_df['algorithm'] != 'prophet'].copy()

    if plot_df.empty:
        print("No data available to plot after filtering for non-prophet algorithms.")
        return

    # Use the same friendly names for algorithms
    to_plot = ["non_strategic_mean", "non_strategic_osco", "greedy"]
    plot_df = plot_df.query("algorithm in @to_plot")
    algonames = {
        "non_strategic_mean": "Fixed Lookahead",
        "non_strategic_osco": "OSCO",
        "greedy": "Greedy"
    }
    plot_df['algorithm_display'] = plot_df['algorithm'].map(algonames).fillna(plot_df['algorithm'])

    sorted_algos = sorted(plot_df['algorithm'].unique())
    colors = [ALGO_STYLES.get(algo, ALGO_STYLES['default'])['color'] for algo in sorted_algos]

    for metric_key, (title_text, interpretation) in plot_metrics_info.items():
        mean_col = f'relative_{metric_key}_mean'
        std_col = f'relative_{metric_key}_std'

        if mean_col not in plot_df.columns or std_col not in plot_df.columns:
            print(f"Skipping summary plot for '{title_text}' due to missing data.")
            continue

        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

        # Plot the bars
        ax.bar(
            plot_df['algorithm_display'],
            plot_df[mean_col],
            yerr=plot_df[std_col],
            capsize=5,  # Adds caps to error bars
            color=colors
        )

        error_bar_bottoms = plot_df[mean_col] - plot_df[std_col]
        error_bar_tops = plot_df[mean_col] + plot_df[std_col]

        # Find the absolute min and max across all error bars
        min_val = error_bar_bottoms.min()
        max_val = error_bar_tops.max()

        # Also consider the baseline of 1.0 in the range to ensure it's always visible
        min_val = min(min_val, 1.0)
        max_val = max(max_val, 1.0)

        # Check if min/max are valid numbers before proceeding
        if pd.notna(min_val) and pd.notna(max_val):
            # Calculate a 10% padding based on the data's range
            data_range = max_val - min_val
            # Handle the edge case where all bars are the same height
            if data_range == 0:
                padding = abs(max_val) * 0.1 if max_val != 0 else 0.1
            else:
                padding = data_range * 0.10

            bottom_limit = min_val - padding
            top_limit = max_val + padding

            # For metrics that should not be negative (like relative percentages or errors),
            # prevent the axis from going below zero for a cleaner look.
            # We check if the lowest data point (bottom of error bar) is non-negative.
            if error_bar_bottoms.min() >= 0:
                bottom_limit = max(0, bottom_limit)

            ax.set_ylim(bottom_limit, top_limit)

        # Add the Prophet baseline at y=1.0 for easy comparison
        # ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Prophet Baseline')

        # Formatting
        ax.set_title(f'Mean Relative {title_text}\n(vs. Prophet, across all years | {interpretation})', fontsize=TITLE_FONTSIZE,
                     weight='bold')
        ax.set_ylabel('Performance Relative to Prophet', fontsize=AXIS_LABEL_FONTSIZE, weight='bold')
        ax.set_xlabel('Algorithm', fontsize=AXIS_LABEL_FONTSIZE, weight='bold')
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # ax.legend(fontsize=14)

        # Add data labels on top of bars
        # for i, bar in enumerate(ax.patches):
        #     ax.text(bar.get_x() + bar.get_width() / 2,
        #             bar.get_height(),
        #             f"{bar.get_height():.2f}",
        #             ha='center', va='bottom',  # Position above the bar
        #             fontsize=12, weight='bold')

        output_plot_path = os.path.join(output_dir, f"summary_relative_{metric_key}.png")
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Summary plot for '{title_text}' saved to {output_plot_path}")

# This function uses your corrected logic and is more robust.
def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # --- L1 Error Calculation (unchanged) ---
    def calculate_l1_error(exec_fractions_dict):
        if not isinstance(exec_fractions_dict, dict): return np.nan
        error = 0.0
        total_time_used = sum(exec_fractions_dict.values())
        if total_time_used < 0.5: return np.nan
        for exec_name, target_frac in TARGET_QUOTAS.items():
            actual_frac = exec_fractions_dict.get(exec_name, 0.0)
            normalized_actual_frac = actual_frac / total_time_used
            error += max(target_frac - normalized_actual_frac, 0)
        return error

    df['l1_eb_error'] = df['exec_time_fractions_real'].apply(calculate_l1_error)

    # --- NEW: Calculate Weighted Completion Scores ---
    # The weights (0.9, 0.07, 0.03) are based on the objective function's internal grade weighting.

    # Weighted SB Completion Score
    df['weighted_sb_completion'] = (
            0.9 * df['completion_pct_sb_A'] +
            0.07 * df['completion_pct_sb_B'] +
            0.03 * df['completion_pct_sb_C']
    )

    # Weighted Project Completion Score
    df['weighted_proj_completion'] = (
            0.9 * df['completion_pct_proj_A'] +
            0.07 * df['completion_pct_proj_B'] +
            0.03 * df['completion_pct_proj_C']
    )

    df['total_value'] = (
            0.30 * df['weighted_sb_completion'] +
            0.30 * df['weighted_proj_completion'] +
            0.15 * df['usage_ratio'] -
            0.25 * df['l1_eb_error']
    )

    return df


def create_tradeoff_grid_2d_sweep(df: pd.DataFrame, output_path: str, year: int):
    """
    Generates a 2x3 grid of trade-off plots with consistent algorithm styling
    and a clean figure-level legend at the bottom.
    """
    plot_metrics = {
        'weighted_proj_completion': ('w_proj', "Weighted Proj. Comp. (%)"),
        'weighted_sb_completion': ('w_sb', "Weighted EB Comp. (%)"),
        'usage_ratio': ('w_util', "Usage Ratio (%)"),
        'l1_eb_error': ('w_ebp', "L1 Exec. Bal. Error (Under-fulfillment)")
    }
    metric_keys = list(plot_metrics.keys())
    # algorithms = sorted(df['algorithm'].unique())
    algorithms = ["non_strategic_mean", "non_strategic_osco", "greedy", 'prophet']


    # Generate the 6 unique pairwise combinations of metrics
    # e.g., (proj, sb), (proj, util), (proj, error), (sb, util), ...
    # We will use the first item in the pair for the Y-axis and the second for the X-axis.
    metric_pairs = list(itertools.combinations(metric_keys, 2))

    # --- NEW: Create a 2x3 grid of subplots ---
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), constrained_layout=True)
    fig.suptitle(f'2D Trade-off Analysis for Cycle Year {year}', fontsize=TITLE_FONTSIZE, weight='bold', y=1.03)

    # Flatten the 2x3 axes array to easily iterate through it
    flat_axes = axes.flatten()

    # Iterate through the 6 metric pairs and their corresponding subplots
    for i, (y_metric_key, x_metric_key) in enumerate(metric_pairs):
        ax = flat_axes[i]

        y_metric_label = plot_metrics[y_metric_key][1]
        x_metric_label = plot_metrics[x_metric_key][1]
        y_weight_col = plot_metrics[y_metric_key][0]
        x_weight_col = plot_metrics[x_metric_key][0]

        # Find the data where these two metrics were swept against each other
        subset_df = df[
            ((df['crit1'] == x_weight_col) & (df['crit2'] == y_weight_col)) |
            ((df['crit1'] == y_weight_col) & (df['crit2'] == x_weight_col))
        ]

        if subset_df.empty:
            ax.text(0.5, 0.5, 'No data for this pair', ha='center', va='center', fontsize=12, style='italic')
            ax.set_xlabel(f"{x_metric_label}", fontsize=14, weight='bold')
            ax.set_ylabel(f"{y_metric_label}", fontsize=14, weight='bold')
            continue

        algonames = {
            "non_strategic_mean": "Fixed Lookahead",
            "non_strategic_osco": "OSCO",
            "greedy": "Greedy",
            'prophet': "Prophet"
        }
        for algo in algorithms:
            algo_df = subset_df[subset_df['algorithm'] == algo]
            # Sort by the x-axis weight to make lines connect logically
            plot_data = algo_df.sort_values(by=x_weight_col)
            if not plot_data.empty:
                style = ALGO_STYLES.get(algo, ALGO_STYLES['default'])
                ax.plot(plot_data[x_metric_key],
                        plot_data[y_metric_key],
                        marker=style['marker'],
                        color=style['color'],
                        linestyle='-',
                        label=algonames[algo],
                        alpha=0.8,
                        markersize=8)

        # Invert axes for "lower is better" metrics
        if 'error' in x_metric_label.lower(): ax.invert_xaxis()
        if 'error' in y_metric_label.lower(): ax.invert_yaxis()

        ax.set_xlabel(f"{x_metric_label}", fontsize=AXIS_LABEL_FONTSIZE, weight='bold')
        ax.set_ylabel(f"{y_metric_label}", fontsize=AXIS_LABEL_FONTSIZE, weight='bold')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)

        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune='both'))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    # Hide any unused subplots if there were fewer than 6 pairs (unlikely here)
    for i in range(len(metric_pairs), len(flat_axes)):
        flat_axes[i].set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        # --- CHANGE 2: MODIFIED LEGEND PLACEMENT ---
        # By anchoring the legend's TOP ('upper center') to the figure's BOTTOM (y=0),
        # we ensure it appears cleanly below the plots without overlapping.
        # `bbox_inches='tight'` in savefig will expand the saved image to include the legend.
        fig.legend(handles, labels,
                   loc='upper center',  # Anchor the legend at its TOP.
                   bbox_to_anchor=(0.5, 0),  # Place the anchor at the bottom-center of the figure.
                   ncol=len(algorithms),  # CHANGE 3: Arrange algorithms in a single row.
                   fontsize=AXIS_LABEL_FONTSIZE,
                   frameon=True,
                   title="Algorithms",
                   title_fontsize=AXIS_LABEL_FONTSIZE)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot grid for year {year} successfully saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze multi-year 2D sweep scheduling simulation results.")
    parser.add_argument("--base_results_dir", type=str, required=True,
                        help="The base directory containing the 'year_YYYY' subdirectories.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output plots.")
    parser.add_argument("--mode", type=str, required=True, help="sweep or fixed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    years_to_process = []
    for item in os.listdir(args.base_results_dir):
        if item.startswith("year_") and os.path.isdir(os.path.join(args.base_results_dir, item)):
            try:
                year = int(item.split('_')[1])
                years_to_process.append(year)
            except (ValueError, IndexError):
                continue

    if not years_to_process:
        print(f"Error: No 'year_YYYY' subdirectories found in '{args.base_results_dir}'. Exiting.")
        exit()

    print(f"Found years to process: {sorted(years_to_process)}")

    # --- MODIFIED: Load all data from all years first ---
    all_year_dfs = []
    for year in sorted(years_to_process):
        try:
            year_df = load_and_parse_results_for_year(args.base_results_dir, year, args.mode)
            if not year_df.empty:
                all_year_dfs.append(year_df)
        except FileNotFoundError as e:
            print(e)
            continue
        except Exception as e:
            print(f"An unexpected error occurred while loading year {year}: {e}")
            continue

    if not all_year_dfs:
        print("No data loaded from any year. Exiting.")
        exit()

    # Combine all yearly data into a single DataFrame
    combined_df = pd.concat(all_year_dfs, ignore_index=True)
    processed_df = process_dataframe(combined_df)

    # --- MODIFIED: Branch logic based on mode after loading all data ---
    if args.mode == "sweep":
        print("\n--- Generating yearly tradeoff grids for SWEEP mode ---")
        for year in sorted(years_to_process):
            year_subset_df = processed_df[processed_df['year'] == year]
            if not year_subset_df.empty:
                output_plot_path = os.path.join(args.output_dir, f"tradeoff_grid_{year}.png")
                create_tradeoff_grid_2d_sweep(year_subset_df, output_plot_path, year)

    elif args.mode == "fixed":
        # First, generate the original per-year bar charts
        print("\n--- Generating yearly objective bar charts for FIXED mode ---")
        for year in sorted(years_to_process):
            year_subset_df = processed_df[processed_df['year'] == year]
            if not year_subset_df.empty:
                create_yearly_objective_bar_charts(year_subset_df, args.output_dir, year)

        # --- NEW: Now, run the summary analysis across all years ---
        metrics_for_summary = [
            'total_value', 'weighted_sb_completion', 'weighted_proj_completion',
            'usage_ratio', 'l1_eb_error'
        ]
        relative_stats_df = calculate_relative_performance(processed_df, metrics_for_summary)

        if not relative_stats_df.empty:
            create_summary_performance_plot(relative_stats_df, args.output_dir)

    print("\nAnalysis complete for all years.")
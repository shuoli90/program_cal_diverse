import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import os

# Hardcoded configuration - edit these for your specific setup
INPUT_PATHS = [
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetTemperatureSweep_2025-03-11_12-22-31/driver_stats.tsv", 
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetTemperatureSweep_2025-03-10_16-48-50/driver_stats.tsv", 
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetTemperatureSweep_2025-03-05_14-06-22/driver_stats.tsv", 
    
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetReevaluation_2025-03-17_23-25-29/driver_stats.tsv", 
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetReRunErrors_2025-03-17_18-25-28/driver_stats.tsv"

    # Add additional paths here as needed
]



OUTPUT_DIR = '/home/shypula/program_cal_diverse/notebooks/temp_plots/'
MODEL_PLOTS_SUBDIR = 'model_plots'  # Subdirectory for per-model plots
METRIC_PLOTS_SUBDIR = 'metric_plots'  # Subdirectory for per-metric plots

# Models to analyze
MODELS_TO_COMPARE = [
    "meta-llama/Llama-3.1-8B-Instruct",
    # "meta-llama/Llama-3.1-70B", 
    "meta-llama/Llama-3.1-70B-Instruct", 
    "meta-llama/Llama-3.1-8B", 
    # "allenai/Llama-3.1-Tulu-3-8B-SFT", 
    # "allenai/Llama-3.1-Tulu-3-70B-SFT"
   
]


MODEL_COLORS = {
    "meta-llama/Llama-3.1-8B-Instruct": "#1f77b4",
    "meta-llama/Llama-3.1-70B-Instruct": "#ff7f0e", 
    "meta-llama/Llama-3.1-8B": "#2ca02c"
}


MODEL_TO_CLEAN_NAME = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B Instruct",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama 3.1 70B Instruct",
    "meta-llama/Llama-3.1-8B": "Llama 3.1 8B Base"
}

# COSINE_METRIC = 'all_average_cosine_distance_programs'
COSINE_METRIC = 'all_average_cosine_distance_raw'

# Define metrics to plot
METRICS = {
    'all_distinct_4_bootstrap': {
        'name': 'Lexical Diversity (n-grams)', 
        'color': 'skyblue',
        'smoothing': 50
    },
    'all_stripped_subtrees_4_bootstrap': {
        'name': 'Syntactic Diversity', 
        'color': 'lightgreen',
        'smoothing': 50
    },
    COSINE_METRIC: {
        'name': 'Cosine Diversity', 
        'color': 'gray',
        'smoothing': 0.5
    },
    'semantic_diversity': {
        'name': 'Effective Semantic Diversity (Ours)', 
        'color': 'purple',
        'smoothing': 5
    },
    'all_coherence': {
        'name': 'Validity (Quality)', 
        'color': 'orange',
        'smoothing': 5
    },
    # "ice_score_in_table": {
    #     'name': 'ICE Score', 
    #     'color': 'red',
    #     'smoothing': 5
    # }
}

# Template to use for analysis
TEMPLATE = "default"

# Visualization settings
BUBBLE_SIZE_COEF = 6  # Coefficient for bubble size (larger = bigger bubbles)
OUTPUT_DPI = 300      # Resolution for saved images


def load_and_prepare_data(file_paths: List[str], model_name: Optional[str] = None, 
                         template: str = "default", temperature: Optional[float] = None) -> pd.DataFrame:
    """
    Load and prepare data from multiple files, with flexible filtering options.
    
    Parameters:
    -----------
    file_paths : List[str]
        Paths to the TSV files containing the data
    model_name : str, optional
        Name of the model to filter for (if None, all models are included)
    template : str, optional
        Template to filter for, default is "default"
    temperature : float, optional
        Specific temperature to filter for (if None, all temperatures are included)
        
    Returns:
    --------
    pd.DataFrame
        Processed data frame with relevant columns
    """
    # Load and concatenate data from all files
    all_data = []
    for file_path in file_paths:
        try:
            data = pd.read_csv(file_path, sep='\t')
            all_data.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_data:
        raise ValueError("No data could be loaded from the provided file paths")
    
    # Concatenate all dataframes
    data = pd.concat(all_data, ignore_index=True)
    
    # Apply filters based on provided parameters
    if model_name is not None:
        data = data[data["model"] == model_name]
    
    if template is not None:
        data = data[data["template"] == template]
        
    if temperature is not None:
        data = data[data["temperature"] == temperature]
    
    # Sort by temperature and reset index
    data = data.sort_values(by=["temperature"])
    data = data.reset_index(drop=True)
    
    # Add heuristic cosine metric if the needed columns exist
    # if "coh_average_cosine_distance_raw" in data.columns and "all_coherence" in data.columns:
    #     data["heuristic_cosine"] = data["coh_average_cosine_distance_raw"] * (data["all_coherence"] / 100)
    
    ### NOTE THIS IS ONLY BECAUSE WE DIDN'T UPDATE SEMANTIC DIVERSITY 
    data["semantic_diversity"] = data.apply(
        lambda row: (row["all_semantic_count_wcoh"] / 32 * 100) 
        if pd.notna(row.get("all_semantic_count_wcoh")) 
        else (row["all_coh_semantic_prop_of_all"] * 100) 
        if pd.notna(row.get("all_coh_semantic_prop_of_all"))
        else np.nan, axis=1)
    
    if data["semantic_diversity"].isna().any():
        raise ValueError("Some rows have no valid semantic diversity measure")
    
    return data


def find_global_limits(file_paths: List[str], model_names: List[str], template: str = "default") -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Determine global min/max values across all models to create consistent plot scales.
    
    Parameters:
    -----------
    file_paths : List[str]
        Paths to the TSV files containing the data
    model_names : List[str]
        List of model names to analyze
    template : str, optional
        Template to filter for, default is "default"
        
    Returns:
    --------
    tuple
        ((x_min, x_max), (y_min, y_max)) for setting consistent plot scales
    """
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    # Required columns to check for min/max values
    y_columns = [
        'all_distinct_4_bootstrap', 
        # 'all_plain_subtrees_4_bootstrap', 
        'all_stripped_subtrees_4_bootstrap',
        # 'all_average_cosine_distance_raw',
        COSINE_METRIC,
        "semantic_diversity",
        "all_coherence"
    ]
    
    for model_name in model_names:
        data = load_and_prepare_data(file_paths, model_name, template)
        
        if data.empty:
            print(f"No data found for model: {model_name}")
            continue
        
        # Update x-axis (temperature) limits
        if 'temperature' in data.columns:
            x_min = min(x_min, data['temperature'].min())
            x_max = max(x_max, data['temperature'].max())
        
        # Update y-axis limits based on all metrics
        for col in y_columns:
            if col in data.columns:
                y_min = min(y_min, data[col].min())
                y_max = max(y_max, data[col].max())
    
    # Add some padding to the limits (5% on each side)
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_min = max(0, x_min - 0.05 * x_range)  # Don't go below 0 for temperature
    x_max = x_max + 0.05 * x_range
    y_min = max(0, y_min - 0.05 * y_range)  # Don't go below 0 for metrics
    y_max = y_max + 0.05 * y_range
    
    return (x_min, x_max), (y_min, y_max)


def plot_diversity_vs_temperature(data: pd.DataFrame, title_prefix: str = "Diversity Metrics", 
                                 s: int = 50, bubble_size_coef: int = 5, 
                                 save_path: Optional[str] = None, show_plot: bool = True, 
                                 figsize: Tuple[int, int] = (12, 8), 
                                 custom_xlim: Optional[Tuple[float, float]] = None, 
                                 custom_ylim: Optional[Tuple[float, float]] = None, 
                                 dpi: int = 300) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Plot diversity metrics against temperature with coherence as bubble size.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Processed data frame with metrics to plot
    title_prefix : str, optional
        Prefix for the plot title
    s : int, optional
        Smoothing parameter for UnivariateSpline, default is 50
    bubble_size_coef : int, optional
        Coefficient for bubble size, default is 5
    save_path : str, optional
        Path to save the figure (if None, figure is not saved)
    show_plot : bool, optional
        Whether to display the plot, default is True
    figsize : tuple, optional
        Figure size (width, height) in inches, default is (12, 8)
    custom_xlim : tuple, optional
        Custom x-axis limits as (min, max)
    custom_ylim : tuple, optional
        Custom y-axis limits as (min, max)
    dpi : int, optional
        Resolution for saved figure, default is 300
        
    Returns:
    --------
    tuple
        Current x and y axis limits for potential reuse
    """
    # Define colors for different metrics
    ngram_color = 'skyblue'
    ast_color = 'salmon'
    abstracted_ast_color = 'lightgreen'
    cosine_color = 'gray'
    semantic_color = 'purple'
    coherence_color = 'orange'
    # Check if all required columns exist
    required_cols = [
        'temperature', 'all_coherence',
        'all_distinct_4_bootstrap', 
        'all_stripped_subtrees_4_bootstrap', 'all_average_cosine_distance_raw',
        'all_average_cosine_distance_programs',
        'semantic_diversity'
    ]
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Create splines for smoother visualization
    ngram_spline = UnivariateSpline(data['temperature'], data['all_distinct_4_bootstrap'], s=s)
    # ast_spline = UnivariateSpline(data['temperature'], data['all_plain_subtrees_4_bootstrap'], s=s)
    abstracted_ast_spline = UnivariateSpline(data['temperature'], data['all_stripped_subtrees_4_bootstrap'], s=s)
    cosine_spline = UnivariateSpline(data['temperature'], data[COSINE_METRIC], s=0.5)
    semantic_spline = UnivariateSpline(data['temperature'], data["semantic_diversity"], s=5)
    coherence_spline = UnivariateSpline(data['temperature'], data["all_coherence"], s=5)
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot scatter points for each metric, with coherence determining point size
                
    plt.scatter(data['temperature'], data['all_stripped_subtrees_4_bootstrap'], 
                label='Canonicalized distinct subtrees AST (H=4)', alpha=0.6, 
                color=abstracted_ast_color)
                
    plt.scatter(data['temperature'], data['all_distinct_4_bootstrap'], 
                label='Distinct N-Grams (N=4)', alpha=0.6, 
                color=ngram_color)
                
    plt.scatter(data['temperature'], data['semantic_diversity'], 
                label='Semantic Diversity', alpha=0.6, 
                color=semantic_color)
                
    plt.scatter(data['temperature'], data[COSINE_METRIC], 
                label='Average Cosine Distance', alpha=0.6, 
                color=cosine_color)
    
    plt.scatter(data['temperature'], data['all_coherence'], 
                label='Coherence', alpha=0.6, 
                color=coherence_color)
    
    # Add spline curves
    plt.plot(data['temperature'], ngram_spline(data['temperature']), color=ngram_color, linestyle='dotted')
    plt.plot(data['temperature'], abstracted_ast_spline(data['temperature']), color=abstracted_ast_color, linestyle='dotted')
    plt.plot(data['temperature'], cosine_spline(data['temperature']), color=cosine_color, linestyle='dotted')
    plt.plot(data['temperature'], semantic_spline(data['temperature']), color=semantic_color, linestyle='dotted')
    plt.plot(data['temperature'], coherence_spline(data['temperature']), color=coherence_color, linestyle='dotted')
    
    # # Annotate with temperature values
    # for i, txt in enumerate(data['temperature']):
    #     plt.annotate(f"{txt}", (data['temperature'][i], data['all_distinct_4_bootstrap'][i]), fontsize=8)
    #     plt.annotate(f"{txt}", (data['temperature'][i], data['all_stripped_subtrees_4_bootstrap'][i]), fontsize=8)
    #     plt.annotate(f"{txt}", (data['temperature'][i], data['all_average_cosine_distance_raw'][i]), fontsize=8)
    #     plt.annotate(f"{txt}", (data['temperature'][i], data["semantic_diversity"][i]), fontsize=8)
    #     plt.annotate(f"{txt}", (data['temperature'][i], data["all_coherence"][i]), fontsize=8)
        
    # Set axes bounds and labels
    if custom_xlim:
        plt.xlim(custom_xlim)
    else:
        plt.xlim(left=0)
        
    if custom_ylim:
        plt.ylim(custom_ylim)
    else:
        plt.ylim(bottom=0)
    
    # Figure out the model name from the data if available
    model_info = ""
    if 'model' in data.columns and not data['model'].empty:
        models = data['model'].unique()
        if len(models) == 1:
            model_info = f" ({models[0]})"
    
    plt.title(f'{title_prefix}{model_info}\nSize Indicates Coherence and Subtree and N-Gram Metrics are Bootstrapped')
    plt.xlabel('Temperature')
    plt.ylabel('Diversity Metric Value')
    plt.legend()
    plt.grid(True)
    
    # Save figure if path is provided
    if save_path:
        # Ensure directory exists
        save_dir = Path(save_path).parent
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Return the current axis limits
    return plt.xlim(), plt.ylim()


def compare_models(file_paths: List[str], model_names: List[str], template: str = "default", 
                  save_dir: Optional[str] = None, shared_scale: bool = True, 
                  **plot_kwargs) -> None:
    """
    Compare diversity metrics across multiple models with improved shared scaling.
    
    Parameters:
    -----------
    file_paths : List[str]
        Paths to the TSV files containing the data
    model_names : list
        List of model names to compare
    template : str, optional
        Template to filter for, default is "default"
    save_dir : str, optional
        Directory to save figures (if None, figures are not saved)
    shared_scale : bool, optional
        Whether to use the same scale for all plots, default is True
    **plot_kwargs : 
        Additional keyword arguments to pass to plot_diversity_vs_temperature
    """
    # Determine global limits first if using shared scale
    xlim, ylim = None, None
    if shared_scale:
        xlim, ylim = find_global_limits(file_paths, model_names, template)
        print(f"Using shared scale: X={xlim}, Y={ylim}")
    
    # Now plot each model
    for model_name in model_names:
        data = load_and_prepare_data(file_paths, model_name, template)
        
        if data.empty:
            print(f"No data found for model: {model_name}")
            continue
            
        save_path = None
        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(exist_ok=True, parents=True)
            model_filename = model_name.replace('/', '-').replace(' ', '_')
            save_path = save_dir_path / f"{model_filename}_{template}.png"
            
        plot_diversity_vs_temperature(
            data,
            title_prefix=f"Diversity Metrics",
            custom_xlim=xlim,
            custom_ylim=ylim,
            save_path=save_path,
            **plot_kwargs
        )
        
        print(f"Processed model: {model_name}")


def plot_models_per_metric(file_paths: List[str], model_names: List[str], metric_name: str,
                       metric_display_name: str, template: str = "default",
                       save_path: Optional[str] = None, show_plot: bool = True,
                       s: int = 50, figsize: Tuple[int, int] = (14, 10),
                       custom_xlim: Optional[Tuple[float, float]] = None,
                       custom_ylim: Optional[Tuple[float, float]] = None,
                       dpi: int = 300) -> None:
    """
    Plot a specific diversity metric across multiple models against temperature.
    
    Parameters:
    -----------
    file_paths : List[str]
        Paths to the TSV files containing the data
    model_names : List[str]
        List of model names to compare
    metric_name : str
        Column name of the metric to plot
    metric_display_name : str
        Display name for the metric (used in title and legend)
    template : str, optional
        Template to filter for, default is "default"
    save_path : str, optional
        Path to save the figure (if None, figure is not saved)
    show_plot : bool, optional
        Whether to display the plot, default is True
    s : int, optional
        Smoothing parameter for UnivariateSpline
    figsize : tuple, optional
        Figure size (width, height) in inches
    custom_xlim : tuple, optional
        Custom x-axis limits as (min, max)
    custom_ylim : tuple, optional
        Custom y-axis limits as (min, max)
    dpi : int, optional
        Resolution for saved figure
    """
    plt.figure(figsize=figsize)
    
    # Generate consistent colors for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    # Track min/max for scaling if not provided
    if custom_xlim is None or custom_ylim is None:
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
    
    # Plot each model
    for i, model_name in enumerate(model_names):
        color = colors[i]
        data = load_and_prepare_data(file_paths, model_name, template)
        
        if data.empty or metric_name not in data.columns:
            print(f"No data for {metric_name} in model: {model_name}")
            continue
        
        # Update min/max if needed
        if custom_xlim is None:
            x_min = min(x_min, data['temperature'].min())
            x_max = max(x_max, data['temperature'].max())
        
        if custom_ylim is None and not data[metric_name].empty:
            y_min = min(y_min, data[metric_name].min())
            y_max = max(y_max, data[metric_name].max())
        
        # Create spline for smoother visualization
        smoothing = METRICS.get(metric_name, {}).get('smoothing', s)
        
        if data[metric_name].isna().any():
            print(f"WARNING: {model_name} has {data[metric_name].isna().sum()} NaN values for {metric_name}")
            # Filter out rows with NaN values for this metric
            data = data.dropna(subset=[metric_name])
            print(f"  After removing NaNs: {len(data)} data points remain")
            
        # remove any duplicates for temperature / model / template
        data = data.drop_duplicates(subset=['temperature', 'model', 'template'])
        
        
        if len(data) > 1:  # Need at least 2 points
            # For n-grams and AST metrics, use linear regression or heavily regularized spline
            if 'all_distinct_4_bootstrap' in metric_name or 'all_stripped_subtrees_4_bootstrap' in metric_name:
                # Very high smoothing parameter for nearly linear spline
                spline = UnivariateSpline(data['temperature'], data[metric_name], s=1e6)
                
                # Could also use numpy polyfit for pure linear regression:
                # z = np.polyfit(data['temperature'], data[metric_name], 1)
                # p = np.poly1d(z)
            else:
                # Use regular spline for other metrics
                spline = UnivariateSpline(data['temperature'], data[metric_name], s=smoothing)
            
            # Plot scatter points and trend line for this model
            plt.scatter(data['temperature'], data[metric_name], 
                        label=f"{model_name}", alpha=0.7, color=color)
            plt.plot(data['temperature'], spline(data['temperature']), 
                    color=color, linestyle='dotted')
            # Annotate with temperature values
            # for j, temp in enumerate(data['temperature']):
            #     plt.annotate(f"{temp}", 
            #                 (data['temperature'][j], data[metric_name][j]), 
            #                 fontsize=8, color=color)
    
    # Set axes bounds and labels
    if custom_xlim:
        plt.xlim(custom_xlim)
    elif x_min != float('inf') and x_max != float('-inf'):
        # Add padding (5% on each side)
        x_range = x_max - x_min
        plt.xlim(max(0, x_min - 0.05 * x_range), x_max + 0.05 * x_range)
    
    if custom_ylim:
        plt.ylim(custom_ylim)
    elif y_min != float('inf') and y_max != float('-inf'):
        # Add padding (5% on each side)
        y_range = y_max - y_min
        plt.ylim(max(0, y_min - 0.05 * y_range), y_max + 0.05 * y_range)
    
    plt.title(f'{metric_display_name} vs Temperature\nModel Comparison')
    plt.xlabel('Temperature')
    plt.ylabel(f'{metric_display_name} Value')
    
    # Sort legend handles and labels based on desired order
    handles, labels = plt.gca().get_legend_handles_labels()
    order = []
    # First find 70B instruct models
    order.extend([i for i, label in enumerate(labels) if "70" in label and "Instruct" in label])
    # Then 8B instruct models  
    order.extend([i for i, label in enumerate(labels) if "8B" in label and "Instruct" in label])
    # Then 8B SFT models
    order.extend([i for i, label in enumerate(labels) if "8B" in label and "SFT" in label])
    # Finally base 8B models
    order.extend([i for i, label in enumerate(labels) if "8B" in label and "Instruct" not in label and "SFT" not in label])
    # Add any remaining indices
    order.extend([i for i in range(len(labels)) if i not in order])
    # Replace model names with clean names for better readability in legend
    clean_labels = []
    for i in order:
        model_name = labels[i]
        clean_name = MODEL_TO_CLEAN_NAME.get(model_name, model_name)
        clean_labels.append(clean_name)
    
    plt.legend([handles[i] for i in order], clean_labels, 
               loc='best', fontsize=14)  # Increased font size for better visibility
    plt.grid(True)
    # Save figure if path is provided
    if save_path:
        # Ensure directory exists
        save_dir = Path(save_path).parent
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

metric_to_limit = {
    "all_distinct_4_bootstrap": (20, 100),
    "all_stripped_subtrees_4_bootstrap": (20, 100),
    "all_average_cosine_distance_raw": (0, 50),
    "all_average_cosine_distance_programs": (0, 50),
    "semantic_diversity": (0, 50),
    "all_coherence": (0, 70)
}

def compare_all_metrics_across_models(file_paths: List[str], model_names: List[str], 
                                    metrics_dict: Dict[str, Dict], template: str = "default",
                                    save_dir: Optional[str] = None, shared_scale: bool = True,
                                    **plot_kwargs) -> None:
    """
    Compare each diversity metric across all models in separate plots.
    
    Parameters:
    -----------
    file_paths : List[str]
        Paths to the TSV files containing the data
    model_names : List[str]
        List of model names to compare
    metrics_dict : Dict[str, Dict]
        Dictionary of metrics to plot with display names and settings
    template : str, optional
        Template to filter for, default is "default"
    save_dir : str, optional
        Directory to save figures (if None, figures are not saved)
    shared_scale : bool, optional
        Whether to use the same y-scale for all metrics, default is True
    **plot_kwargs : 
        Additional keyword arguments to pass to plot_models_per_metric
    """
    # Determine global limits first if using shared scale
    xlim, ylim = None, None
    if shared_scale:
        # Use the same function we already have for finding global limits
        xlim, ylim = find_global_limits(file_paths, model_names, template)
        print(f"Using shared scale: X={xlim}, Y={ylim}")
    
    # Create a plot for each metric
    for metric_name, metric_info in metrics_dict.items():
        display_name = metric_info.get('name', metric_name)
        y_limit = metric_to_limit.get(metric_name, (0, 60))
        
        
        save_path = None
        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(exist_ok=True, parents=True)
            metric_filename = metric_name.replace('/', '-').replace(' ', '_')
            save_path = save_dir_path / f"{metric_filename}_{template}.png"
        
        # Plot this metric across all models
        plot_models_per_metric(
            file_paths=file_paths,
            model_names=model_names,
            metric_name=metric_name,
            metric_display_name=display_name,
            template=template,
            save_path=save_path,
            custom_xlim=xlim,
            custom_ylim=y_limit,
            **plot_kwargs
        )
        
        print(f"Created comparison plot for metric: {display_name}")
        
        
            
def create_combined_plot(file_paths, model_names, metrics_to_combine, template="default", 
                         save_path=None, figsize=(20, 8), dpi=300, show_plot=True):
    """
    Create a combined figure with multiple metrics side by side.
    
    Parameters:
    -----------
    file_paths : List[str]
        Paths to the TSV files containing the data
    model_names : List[str]
        List of model names to compare
    metrics_to_combine : List[str]
        List of metric names to include in the combined plot
    template : str, optional
        Template to filter for, default is "default"
    save_path : str, optional
        Path to save the figure (if None, figure is not saved)
    figsize : tuple, optional
        Figure size (width, height) in inches
    dpi : int, optional
        Resolution for saved figure
    show_plot : bool, optional
        Whether to display the plot
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(metrics_to_combine), figsize=figsize)
    
    # Generate consistent colors for each model
    # colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    colors = [MODEL_COLORS.get(model_name, "#1f77b4") for model_name in model_names]
    # Process each metric in its own subplot
    for i, metric_name in enumerate(metrics_to_combine):
        ax = axes[i]
        metric_info = METRICS.get(metric_name, {})
        display_name = metric_info.get('name', metric_name)
        smoothing = metric_info.get('smoothing', 50)
        y_limit = metric_to_limit.get(metric_name, (0, 60))
        
        # Plot each model's data for this metric
        for j, model_name in enumerate(model_names):
            color = colors[j]
            data = load_and_prepare_data(file_paths, model_name, template)
            
            if data.empty or metric_name not in data.columns:
                print(f"No data for {metric_name} in model: {model_name}")
                continue
            
            # Check for NaN values in this dataset/metric combination
            if data[metric_name].isna().any():
                print(f"WARNING: {model_name} has {data[metric_name].isna().sum()} NaN values for {metric_name}")
                # Filter out rows with NaN values for this metric
                data = data.dropna(subset=[metric_name])
                print(f"  After removing NaNs: {len(data)} data points remain")
                
            # Remove any duplicates for temperature / model / template
            data = data.drop_duplicates(subset=['temperature', 'model', 'template'])
            
            # Create spline for smoother visualization
            if len(data) > 1:  # Need at least 2 points
                # For n-grams and AST metrics, use linear regression or heavily regularized spline
                if 'all_distinct_4_bootstrap' in metric_name or 'all_stripped_subtrees_4_bootstrap' in metric_name:
                    # Very high smoothing parameter for nearly linear spline
                    spline = UnivariateSpline(data['temperature'], data[metric_name], s=1e6)
                else:
                    # Use regular spline for other metrics
                    spline = UnivariateSpline(data['temperature'], data[metric_name], s=smoothing)
                
                # Plot scatter points and trend line for this model
                clean_name = MODEL_TO_CLEAN_NAME.get(model_name, model_name)
                ax.scatter(data['temperature'], data[metric_name], 
                           label=clean_name, alpha=0.7, color=color)
                ax.plot(data['temperature'], spline(data['temperature']), 
                        color=color, linestyle='dotted')
        
        # Configure this subplot
        # Use set_title with pad parameter to add vertical space between title and plot
        ax.set_title(display_name, fontsize=20, pad=20)  # Increased padding between title and plot
        ax.set_xlabel('Temperature', fontsize=16)
        ax.set_ylabel(f'{display_name} Value', fontsize=14)
        ax.set_ylim(y_limit)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Add a dotted box around the semantic diversity plot to emphasize it
        if metric_name == 'semantic_diversity':
            # Get the current axis limits
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            # Get the figure coordinates of the subplot
            bbox = ax.get_position()
            fig_left = bbox.x0
            fig_right = bbox.x1
            fig_bottom = bbox.y0
            fig_top = bbox.y1
            
            # Add padding around the entire subplot (including title)
            padding_x = 0.01  # Padding in figure coordinates
            padding_y = 0.02
            
            # Create a rectangle in figure coordinates that includes the title area
            rect = plt.Rectangle(
                (fig_left - 2*padding_x - 0.10, fig_bottom - 2*padding_y + .025),
                (fig_right - fig_left) + 2 * padding_x + 0.0825,
                (fig_top - fig_bottom) + 2 * padding_y + 0.005,  # Extra space to fully include title
                fill=False, linestyle='dotted', linewidth=3,
                edgecolor='blue', zorder=10,
                transform=fig.transFigure
            )
            fig.add_artist(rect)
    
    # Add a main title
    fig.suptitle(f"Comparison of Key Metrics Across Models", fontsize=24)
    
    # Create a single legend for the entire figure at the bottom
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # Remove duplicates while preserving order
    unique_labels = []
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
    
    # Sort legend handles and labels based on desired order
    order = []
    # First find 70B instruct models
    order.extend([i for i, label in enumerate(unique_labels) if "70B" in label and "Instruct" in label])
    # Then 8B instruct models  
    order.extend([i for i, label in enumerate(unique_labels) if "8B" in label and "Instruct" in label])
    # Then 8B SFT models
    order.extend([i for i, label in enumerate(unique_labels) if "8B" in label and "SFT" in label])
    # Finally base 8B models
    order.extend([i for i, label in enumerate(unique_labels) if "8B" in label and "Instruct" not in label and "SFT" not in label])
    # Add any remaining indices
    order.extend([i for i in range(len(unique_labels)) if i not in order])
    
    # Add the legend at the bottom of the figure
    fig.legend([unique_handles[i] for i in order], [unique_labels[i] for i in order], 
              loc='lower center', bbox_to_anchor=(0.5, 0.02), 
              ncol=len(unique_labels), fontsize=16)
    
    # Adjust layout to make room for the title and legend
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Save figure if path is provided
    if save_path:
        # Ensure directory exists
        save_dir = Path(save_path).parent
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Combined plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

# Usage example:
def create_combined_metrics_pdf(file_paths, model_names, template="default", output_dir=None):
    """
    Create a PDF with combined plots of key metrics.
    
    Parameters:
    -----------
    file_paths : List[str]
        Paths to the TSV files containing the data
    model_names : List[str]
        List of model names to compare
    template : str, optional
        Template to filter for, default is "default"
    output_dir : str, optional
        Directory to save the PDF, if None uses the OUTPUT_DIR
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Define key metrics to combine
    key_metrics = ['semantic_diversity', 'all_coherence', 'all_distinct_4_bootstrap']
    
    # Create the path for the combined PDF
    pdf_path = os.path.join(output_dir, f"combined_metrics_{template}.pdf")
    
    # Generate the combined plot
    fig = create_combined_plot(
        file_paths=file_paths,
        model_names=model_names,
        metrics_to_combine=key_metrics,
        template=template,
        save_path=pdf_path,
        figsize=(20, 8),
        dpi=300,
        show_plot=False
    )
    
    return pdf_path


if __name__ == "__main__":
    # Example usage:
    
    # Use hardcoded configuration from the top of the script
    file_paths = INPUT_PATHS
    base_output_dir = OUTPUT_DIR
    
    # Create subdirectories for different plot types
    model_plots_dir = os.path.join(base_output_dir, MODEL_PLOTS_SUBDIR)
    metric_plots_dir = os.path.join(base_output_dir, METRIC_PLOTS_SUBDIR)
    
    Path(model_plots_dir).mkdir(exist_ok=True, parents=True)
    Path(metric_plots_dir).mkdir(exist_ok=True, parents=True)
    
    # # 1. Generate per-model plots with all metrics (original functionality)
    # print("Generating per-model plots...")
    # compare_models(
    #     file_paths,
    #     MODELS_TO_COMPARE,
    #     template=TEMPLATE,
    #     save_dir=model_plots_dir,
    #     shared_scale=True,
    #     bubble_size_coef=BUBBLE_SIZE_COEF,
    #     dpi=OUTPUT_DPI,
    #     show_plot=False  # Don't show plots in non-interactive mode
    # )
    
    # # 2. Generate per-metric plots with all models (new functionality)
    # print("Generating per-metric plots...")
    # compare_all_metrics_across_models(
    #     file_paths,
    #     MODELS_TO_COMPARE,
    #     METRICS,
    #     template=TEMPLATE,
    #     save_dir=metric_plots_dir,
    #     shared_scale=False,
    #     dpi=OUTPUT_DPI,
    #     show_plot=False  # Don't show plots in non-interactive mode
    # )
    
    # print(f"All plots generated successfully:")
    # print(f" - Model plots saved to: {model_plots_dir}")
    # print(f" - Metric plots saved to: {metric_plots_dir}")
    
    # key_metrics = ['semantic_diversity', 'all_coherence', 'all_distinct_4_bootstrap', COSINE_METRIC]
    key_metrics = ['semantic_diversity', 'all_coherence',  COSINE_METRIC]
    # key_metrics = ['codellama_cosine_distance_raw", "ice_score_table"]
    combined_pdf_path = os.path.join(base_output_dir, f"combined_metrics_{TEMPLATE}.pdf")
    
    create_combined_plot(
        file_paths=file_paths,
        model_names=MODELS_TO_COMPARE,
        metrics_to_combine=key_metrics,
        template=TEMPLATE,
        save_path=combined_pdf_path,
        figsize=(20, 8),
        dpi=OUTPUT_DPI,
        show_plot=False
    )
    
    print(f"All plots generated successfully:")
    print(f" - Model plots saved to: {model_plots_dir}")
    print(f" - Metric plots saved to: {metric_plots_dir}")
    print(f" - Combined plot saved to: {combined_pdf_path}")
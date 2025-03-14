import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

# Hardcoded configuration - edit these for your specific setup
INPUT_PATHS = [
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetTemperatureSweep_2025-03-11_12-22-31/driver_stats.tsv", 
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetTemperatureSweep_2025-03-10_16-48-50/driver_stats.tsv", 
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetTemperatureSweep_2025-03-05_14-06-22/driver_stats.tsv"
    
    
    # Add additional paths here as needed
]
OUTPUT_DIR = '/home/shypula/program_cal_diverse/notebooks/temp_plots/'

# Models to analyze
MODELS_TO_COMPARE = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct", 
    "meta-llama/Llama-3.1-8B", 
    "allenai/Llama-3.1-Tulu-3-8B-SFT", 
    "allenai/Llama-3.1-Tulu-3-70B-SFT"
   
]

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
    if "coh_average_cosine_distance_raw" in data.columns and "all_coherence" in data.columns:
        data["heuristic_cosine"] = data["coh_average_cosine_distance_raw"] * (data["all_coherence"] / 100)
    
    ### NOTE THIS IS ONLY BECAUSE WE DIDN'T UPDATE SEMANTIC DIVERSITY 
    data["semantic_diversity"] = (data["all_semantic_count_wcoh"] / 32 ) * 100
    
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
        'all_average_cosine_distance_raw',
        "semantic_diversity"
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
        'semantic_diversity'
    ]
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Create splines for smoother visualization
    ngram_spline = UnivariateSpline(data['temperature'], data['all_distinct_4_bootstrap'], s=s)
    # ast_spline = UnivariateSpline(data['temperature'], data['all_plain_subtrees_4_bootstrap'], s=s)
    abstracted_ast_spline = UnivariateSpline(data['temperature'], data['all_stripped_subtrees_4_bootstrap'], s=s)
    cosine_spline = UnivariateSpline(data['temperature'], data['all_average_cosine_distance_raw'], s=0.5)
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
                
    plt.scatter(data['temperature'], data['all_average_cosine_distance_raw'], 
                label='Average Cosine Distance', alpha=0.6, 
                color=cosine_color)
    
    plt.scatter(data['temperature'], data['all_coherence'], 
                label='Coherence', alpha=0.6, 
                color=coherence_color)
    
    # Add spline curves
    plt.plot(data['temperature'], ngram_spline(data['temperature']), color=ngram_color, linestyle='dotted')
    plt.plot(data['temperature'], abstracted_ast_spline(data['temperature']), color=abstracted_ast_color, linestyle='dotted')
    plt.plot(data['temperature'], abstracted_ast_spline(data['temperature']), color=abstracted_ast_color, linestyle='dotted')
    plt.plot(data['temperature'], cosine_spline(data['temperature']), color=cosine_color, linestyle='dotted')
    plt.plot(data['temperature'], semantic_spline(data['temperature']), color=semantic_color, linestyle='dotted')
    plt.plot(data['temperature'], coherence_spline(data['temperature']), color=coherence_color, linestyle='dotted')
    # Annotate with temperature values
    for i, txt in enumerate(data['temperature']):
        plt.annotate(f"{txt}", (data['temperature'][i], data['all_distinct_4_bootstrap'][i]), fontsize=8)
        plt.annotate(f"{txt}", (data['temperature'][i], data['all_stripped_subtrees_4_bootstrap'][i]), fontsize=8)
        plt.annotate(f"{txt}", (data['temperature'][i], data['all_average_cosine_distance_raw'][i]), fontsize=8)
        plt.annotate(f"{txt}", (data['temperature'][i], data["semantic_diversity"][i]), fontsize=8)
        plt.annotate(f"{txt}", (data['temperature'][i], data["all_coherence"][i]), fontsize=8)
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


if __name__ == "__main__":
    # Example usage:
    
    # Use hardcoded configuration from the top of the script
    file_paths = INPUT_PATHS
    output_dir = OUTPUT_DIR
    
    # Compare all models with shared scale
    compare_models(
        file_paths,
        MODELS_TO_COMPARE,
        template=TEMPLATE,
        save_dir=output_dir,
        shared_scale=True,
        bubble_size_coef=BUBBLE_SIZE_COEF,
        dpi=OUTPUT_DPI
    )
    
    # Optionally, you can also analyze a single model
    # single_model = MODELS_TO_COMPARE[0]  # Just use the first model
    # data = load_and_prepare_data(file_paths, single_model, template=TEMPLATE)
    # plot_diversity_vs_temperature(
    #     data,
    #     save_path=f"{output_dir}/{single_model.replace('/', '-')}_analysis.png",
    #     bubble_size_coef=BUBBLE_SIZE_COEF,
    #     dpi=OUTPUT_DPI
    # )
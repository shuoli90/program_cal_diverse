#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diversity Metrics Analysis Script

This script analyzes diversity metrics (coherence, semantic, lexical, syntactic)
from driver_stats.tsv files, performing statistical tests between model pairs.

It supports three types of comparisons:
1. Base vs. Instruction-tuned models (e.g., CodeLlama-7b vs CodeLlama-7b-Instruct)
2. Small vs. Large models (e.g., 7B vs 70B models)
3. Zero-shot vs. Two-shot prompting (same model with different templates)

Each comparison calculates:
- Sample size (n)
- Wilcoxon signed-rank test p-value
- Cohen's d effect size
- Mean and median differences
- Direction of effect
"""

import argparse
import json
import os
import yaml
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure pandas display options
pd.set_option('display.max_rows', None)      # Show all rows
pd.set_option('display.max_columns', 50)     # Show up to 50 columns
pd.set_option('display.width', 1000)         # Wider display
pd.set_option('display.max_colwidth', 100)   # Limit column width for readability

# Alternative settings (commented out by default)
# pd.set_option('display.max_rows', 100)     # Show limited rows
# pd.set_option('display.max_columns', None) # Show all columns
# pd.set_option('display.max_colwidth', None) # Show full column content

#------------------------------------------------------------------------------
# CONFIGURATION CONSTANTS - Edit these values to customize the analysis
#------------------------------------------------------------------------------

# Metrics to analyze - these must match column names (after renaming) in the TSV files
# - coherence: Correctness of generated code (percentage)
# - semantic_diversity: Functional differences between generations
# - lexical_diversity: N-gram diversity measurement
# - syntactic_diversity: AST-based structural diversity
# PATHS = ["/data0/shypula/prog_diversity/all_experiments/Open_Ended_Reevaluation_EAD_Open_and_Commercial_2024-08-02_16-02-07/driver_stats.tsv", 
#          "/data1/shypula/prog_diversity/all_experiments/OpenEndedCommercialV3_2024-08-09_02-28-20/driver_stats.tsv", 
#          "/data1/shypula/prog_diversity/all_experiments/Human_Directed_Reevaluation_EAD_82/driver_stats.tsv", 
#          "/data0/shypula/prog_diversity/all_experiments/Open_Ended_Reeavluation_Rebuttal_v2_2024-11-27_23-06-50/driver_stats.tsv"]



METRICS = ["coherence", "semantic_diversity", "lexical_diversity", "syntactic_diversity", "neural_diversity", 
           "coh_semantic_diversity", "coh_lexical_diversity", "coh_syntactic_diversity", "coh_neural_diversity"]

# Column name mappings (old_name: new_name)
# This allows flexibility in column naming conventions across different experiment runs
# The script will look for either the original names or the standardized names

## TODO: CHANGE THESE TO REFLECT THE NEW METRICS 
RENAME_DICT = {
    "all_coherence": "coherence",
    "all_coh_semantic_prop_of_all": "semantic_diversity",
    # "all_semantic_count_wcoh_nonempty_woutput": "semantic_diversity",
    "all_ead_4_bootstrap": "lexical_diversity", 
    "all_stripped_subtrees_4_ead_bootstrap": "syntactic_diversity",
    "all_average_cosine_distance_raw": "neural_diversity", 
    
    "all_pairwise_semantic_prop_wcoh": "coh_semantic_diversity", 
    "coh_ead_4_bootstrap": "coh_lexical_diversity",
    "coh_stripped_subtrees_4_ead_bootstrap": "coh_syntactic_diversity",
    "coh_average_cosine_distance_raw": "coh_neural_diversity"
}

# Default model order for consistent sorting in tables and visualizations
# Models will be displayed in this order regardless of their order in the data
# Models not in this list will appear after these in alphabetical order
DEFAULT_MODEL_ORDER = [
    "human_outputs/human_outputs",  # Human generations (baseline)
    
    "meta-llama/Llama-3.1-70B-Instruct",
    "allenai/Llama-3.1-Tulu-3-70B", 
    "allenai/Llama-3.1-Tulu-3-70B-DPO",
    "allenai/Llama-3.1-Tulu-3-70B-SFT",
    "meta-llama/Llama-3.1-70B",

    # Llama 3.1 8B models
    "meta-llama/Llama-3.1-8B-Instruct",
    "allenai/Llama-3.1-Tulu-3.1-8B",
    "allenai/Llama-3.1-Tulu-3-8B", 
    "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "meta-llama/Llama-3.1-8B",

    # Llama 2 70B models
    "meta-llama/Llama-2-70b-chat-hf",
    "allenai/tulu-2-70b",
    "allenai/tulu-2-dpo-70b",
    "meta-llama/Llama-2-70b-hf",

    # Llama 2 7B models
    "meta-llama/Llama-2-7b-chat-hf",
    "allenai/tulu-2-7b",
    "allenai/tulu-2-dpo-7b",
    "meta-llama/Llama-2-7b-hf"
    
    
    # # Meta Llama 3.1 models
    # "meta-llama/Meta-Llama-3.1-70B-Instruct", 
    # "meta-llama/Meta-Llama-3.1-70B",
    # # Meta Llama 3 models
    # "meta-llama/Meta-Llama-3-70B-Instruct", 
    # "meta-llama/Meta-Llama-3-70B",
    # "meta-llama/Meta-Llama-3-8B-Instruct", 
    # "meta-llama/Meta-Llama-3-8B",
    # # Meta Llama 3.1 (8B) models
    # "meta-llama/Meta-Llama-3.1-8B-Instruct", 
    # "meta-llama/Meta-Llama-3.1-8B",
    # # CodeLlama models
    # "codellama/CodeLlama-70b-Instruct-hf", 
    # "codellama/CodeLlama-70b-Python-hf",
    # "codellama/CodeLlama-34b-Instruct-hf", 
    # "codellama/CodeLlama-34b-hf",
    # "codellama/CodeLlama-7b-Instruct-hf", 
    # "codellama/CodeLlama-7b-hf",
    # # Claude models
    # "HAIKU", "SONNET",
    # # GPT models
    # "gpt-3.5-turbo-instruct", "gpt-3.5-turbo",
    # "davinci-002", "babbage-002"
]


# BASE_VS_INSTRUCT_SFT = [
#     # CodeLlama 7B
#     (("codellama/CodeLlama-7b-hf", "default"), ("codellama/CodeLlama-7b-Instruct-hf", "default")),
#     (("codellama/CodeLlama-7b-hf", "two_shot"), ("codellama/CodeLlama-7b-Instruct-hf", "two_shot")),
#     (("codellama/CodeLlama-7b-hf", "two_shot_cot"), ("codellama/CodeLlama-7b-Instruct-hf", "two_shot_cot")),
#     # CodeLlama 34B
#     (("codellama/CodeLlama-34b-hf", "default"), ("codellama/CodeLlama-34b-Instruct-hf", "default")),
#     (("codellama/CodeLlama-34b-hf", "two_shot"), ("codellama/CodeLlama-34b-Instruct-hf", "two_shot")),
#     (("codellama/CodeLlama-34b-hf", "two_shot_cot"), ("codellama/CodeLlama-34b-Instruct-hf", "two_shot_cot")),
#     # CodeLlama 70B
#     (("codellama/CodeLlama-70b-Python-hf", "default"), ("codellama/CodeLlama-70b-Instruct-hf", "default")),
#     (("codellama/CodeLlama-70b-Python-hf", "two_shot"), ("codellama/CodeLlama-70b-Instruct-hf", "two_shot")),
#     (("codellama/CodeLlama-70b-Python-hf", "two_shot_cot"), ("codellama/CodeLlama-70b-Instruct-hf", "two_shot_cot")),
# ]

# # Base models vs their RLHF (Reinforcement Learning from Human Feedback) counterparts
# # These are the Llama3 and Llama3.1 models
# BASE_VS_INSTRUCT_RLHF = [
#     # Llama 3 8B
#     (("meta-llama/Meta-Llama-3-8B", "default"), ("meta-llama/Meta-Llama-3-8B-Instruct", "default")),
#     (("meta-llama/Meta-Llama-3-8B", "two_shot"), ("meta-llama/Meta-Llama-3-8B-Instruct", "two_shot")),
#     (("meta-llama/Meta-Llama-3-8B", "two_shot_cot"), ("meta-llama/Meta-Llama-3-8B-Instruct", "two_shot_cot")),
#     # Llama 3 70B
#     (("meta-llama/Meta-Llama-3-70B", "default"), ("meta-llama/Meta-Llama-3-70B-Instruct", "default")),
#     (("meta-llama/Meta-Llama-3-70B", "two_shot"), ("meta-llama/Meta-Llama-3-70B-Instruct", "two_shot")),
#     (("meta-llama/Meta-Llama-3-70B", "two_shot_cot"), ("meta-llama/Meta-Llama-3-70B-Instruct", "two_shot_cot")),
#     # Llama 3.1 8B
#     (("meta-llama/Meta-Llama-3.1-8B", "default"), ("meta-llama/Meta-Llama-3.1-8B-Instruct", "default")),
#     (("meta-llama/Meta-Llama-3.1-8B", "two_shot"), ("meta-llama/Meta-Llama-3.1-8B-Instruct", "two_shot")),
#     (("meta-llama/Meta-Llama-3.1-8B", "two_shot_cot"), ("meta-llama/Meta-Llama-3.1-8B-Instruct", "two_shot_cot")),
#     # Llama 3.1 70B
#     (("meta-llama/Meta-Llama-3.1-70B", "default"), ("meta-llama/Meta-Llama-3.1-70B-Instruct", "default")),
#     (("meta-llama/Meta-Llama-3.1-70B", "two_shot"), ("meta-llama/Meta-Llama-3.1-70B-Instruct", "two_shot")),
#     (("meta-llama/Meta-Llama-3.1-70B", "two_shot_cot"), ("meta-llama/Meta-Llama-3.1-70B-Instruct", "two_shot_cot")),
# ]

# BASE_VS_INSTRUCT_ALL = BASE_VS_INSTRUCT_SFT + BASE_VS_INSTRUCT_RLHF



# SMALL_VS_LARGE_BASE_SFT = [
#     # Base model pairs (CodeLlama)
#     (("codellama/CodeLlama-7b-hf", "default"), ("codellama/CodeLlama-34b-hf", "default")),
#     (("codellama/CodeLlama-7b-hf", "two_shot"), ("codellama/CodeLlama-34b-hf", "two_shot")),
#     (("codellama/CodeLlama-7b-hf", "two_shot_cot"), ("codellama/CodeLlama-34b-hf", "two_shot_cot")),
#     (("codellama/CodeLlama-34b-hf", "default"), ("codellama/CodeLlama-70b-Python-hf", "default")),
#     (("codellama/CodeLlama-34b-hf", "two_shot"), ("codellama/CodeLlama-70b-Python-hf", "two_shot")),
#     (("codellama/CodeLlama-34b-hf", "two_shot_cot"), ("codellama/CodeLlama-70b-Python-hf", "two_shot_cot")),
# ]

# # Small vs Large comparisons for instruction-tuned CodeLlama models (SFT)
# SMALL_VS_LARGE_INSTRUCT_SFT = [
#     # Instruction-tuned model pairs (CodeLlama)
#     (("codellama/CodeLlama-7b-Instruct-hf", "default"), ("codellama/CodeLlama-34b-Instruct-hf", "default")),
#     (("codellama/CodeLlama-7b-Instruct-hf", "two_shot"), ("codellama/CodeLlama-34b-Instruct-hf", "two_shot")),
#     (("codellama/CodeLlama-7b-Instruct-hf", "two_shot_cot"), ("codellama/CodeLlama-34b-Instruct-hf", "two_shot_cot")),
#     (("codellama/CodeLlama-34b-Instruct-hf", "default"), ("codellama/CodeLlama-70b-Instruct-hf", "default")),
#     (("codellama/CodeLlama-34b-Instruct-hf", "two_shot"), ("codellama/CodeLlama-70b-Instruct-hf", "two_shot")),
#     (("codellama/CodeLlama-34b-Instruct-hf", "two_shot_cot"), ("codellama/CodeLlama-70b-Instruct-hf", "two_shot_cot")),
# ]

# # Small vs Large comparisons for base Llama models (RLHF)
# SMALL_VS_LARGE_BASE_RLHF = [
#     # Base model pairs (Llama3 and Llama3.1)
#     (("meta-llama/Meta-Llama-3-8B", "default"), ("meta-llama/Meta-Llama-3-70B", "default")),
#     (("meta-llama/Meta-Llama-3-8B", "two_shot"), ("meta-llama/Meta-Llama-3-70B", "two_shot")),
#     (("meta-llama/Meta-Llama-3-8B", "two_shot_cot"), ("meta-llama/Meta-Llama-3-70B", "two_shot_cot")),
#     (("meta-llama/Meta-Llama-3.1-8B", "default"), ("meta-llama/Meta-Llama-3.1-70B", "default")),
#     (("meta-llama/Meta-Llama-3.1-8B", "two_shot"), ("meta-llama/Meta-Llama-3.1-70B", "two_shot")),
#     (("meta-llama/Meta-Llama-3.1-8B", "two_shot_cot"), ("meta-llama/Meta-Llama-3.1-70B", "two_shot_cot")),
# ]

# # Small vs Large comparisons for instruction-tuned Llama models (RLHF)
# SMALL_VS_LARGE_INSTRUCT_RLHF = [
#     # Instruction-tuned model pairs (Llama3 and Llama3.1)
#     (("meta-llama/Meta-Llama-3-8B-Instruct", "default"), ("meta-llama/Meta-Llama-3-70B-Instruct", "default")),
#     (("meta-llama/Meta-Llama-3-8B-Instruct", "two_shot"), ("meta-llama/Meta-Llama-3-70B-Instruct", "two_shot")),
#     (("meta-llama/Meta-Llama-3-8B-Instruct", "two_shot_cot"), ("meta-llama/Meta-Llama-3-70B-Instruct", "two_shot_cot")),
#     (("meta-llama/Meta-Llama-3.1-8B-Instruct", "default"), ("meta-llama/Meta-Llama-3.1-70B-Instruct", "default")),
#     (("meta-llama/Meta-Llama-3.1-8B-Instruct", "two_shot"), ("meta-llama/Meta-Llama-3.1-70B-Instruct", "two_shot")),
#     (("meta-llama/Meta-Llama-3.1-8B-Instruct", "two_shot_cot"), ("meta-llama/Meta-Llama-3.1-70B-Instruct", "two_shot_cot")),
# ]

# # Combined small vs large comparisons for all SFT models (base + instruction-tuned)
# SMALL_VS_LARGE_SFT = SMALL_VS_LARGE_BASE_SFT + SMALL_VS_LARGE_INSTRUCT_SFT

# # Combined small vs large comparisons for all RLHF models (base + instruction-tuned)
# SMALL_VS_LARGE_RLHF = SMALL_VS_LARGE_BASE_RLHF + SMALL_VS_LARGE_INSTRUCT_RLHF

# # Combined small vs large comparisons for all models (base + instruction-tuned)
# SMALL_VS_LARGE_BASE = SMALL_VS_LARGE_BASE_SFT + SMALL_VS_LARGE_BASE_RLHF
# SMALL_VS_LARGE_INSTRUCT = SMALL_VS_LARGE_INSTRUCT_SFT + SMALL_VS_LARGE_INSTRUCT_RLHF


# SMALL_VS_LARGE_ALL = SMALL_VS_LARGE_BASE + SMALL_VS_LARGE_INSTRUCT


# # Zero-shot vs Two-shot for base CodeLlama models (SFT)
# ZERO_VS_TWO_SHOT_BASE_SFT = [
#     (("codellama/CodeLlama-7b-hf", "default"), ("codellama/CodeLlama-7b-hf", "two_shot")),
#     (("codellama/CodeLlama-34b-hf", "default"), ("codellama/CodeLlama-34b-hf", "two_shot")),
#     (("codellama/CodeLlama-70b-Python-hf", "default"), ("codellama/CodeLlama-70b-Python-hf", "two_shot")),
# ]

# # Zero-shot vs Two-shot for instruction-tuned CodeLlama models (SFT)
# ZERO_VS_TWO_SHOT_INSTRUCT_SFT = [
#     (("codellama/CodeLlama-7b-Instruct-hf", "default"), ("codellama/CodeLlama-7b-Instruct-hf", "two_shot")),
#     (("codellama/CodeLlama-34b-Instruct-hf", "default"), ("codellama/CodeLlama-34b-Instruct-hf", "two_shot")),
#     (("codellama/CodeLlama-70b-Instruct-hf", "default"), ("codellama/CodeLlama-70b-Instruct-hf", "two_shot")),
# ]

# # Zero-shot vs Two-shot for base Llama models (RLHF)
# ZERO_VS_TWO_SHOT_BASE_RLHF = [
#     (("meta-llama/Meta-Llama-3-8B", "default"), ("meta-llama/Meta-Llama-3-8B", "two_shot")),
#     (("meta-llama/Meta-Llama-3-70B", "default"), ("meta-llama/Meta-Llama-3-70B", "two_shot")),
#     (("meta-llama/Meta-Llama-3.1-8B", "default"), ("meta-llama/Meta-Llama-3.1-8B", "two_shot")),
#     (("meta-llama/Meta-Llama-3.1-70B", "default"), ("meta-llama/Meta-Llama-3.1-70B", "two_shot")),
# ]

# # Zero-shot vs Two-shot for instruction-tuned Llama models (RLHF)
# ZERO_VS_TWO_SHOT_INSTRUCT_RLHF = [
#     (("meta-llama/Meta-Llama-3-8B-Instruct", "default"), ("meta-llama/Meta-Llama-3-8B-Instruct", "two_shot")),
#     (("meta-llama/Meta-Llama-3-70B-Instruct", "default"), ("meta-llama/Meta-Llama-3-70B-Instruct", "two_shot")),
#     (("meta-llama/Meta-Llama-3.1-8B-Instruct", "default"), ("meta-llama/Meta-Llama-3.1-8B-Instruct", "two_shot")),
#     (("meta-llama/Meta-Llama-3.1-70B-Instruct", "default"), ("meta-llama/Meta-Llama-3.1-70B-Instruct", "two_shot")),
# ]

# ZERO_VS_TWO_SHOT_BASE = ZERO_VS_TWO_SHOT_BASE_SFT + ZERO_VS_TWO_SHOT_BASE_RLHF

# ZERO_VS_TWO_SHOT_INSTRUCT = ZERO_VS_TWO_SHOT_INSTRUCT_SFT + ZERO_VS_TWO_SHOT_INSTRUCT_RLHF

# ZERO_VS_TWO_SHOT_ALL = ZERO_VS_TWO_SHOT_BASE + ZERO_VS_TWO_SHOT_INSTRUCT

# Base models vs their SFT (Supervised Fine-Tuning) counterparts
BASE_VS_INSTRUCT_SFT = [
    # Llama 3.1 70B vs SFT
    (("meta-llama/Llama-3.1-70B", "default"), ("allenai/Llama-3.1-Tulu-3-70B-SFT", "default")),
    (("meta-llama/Llama-3.1-70B", "two_shot"), ("allenai/Llama-3.1-Tulu-3-70B-SFT", "two_shot")),
    (("meta-llama/Llama-3.1-70B", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-70B-SFT", "two_shot_cot")),
    # Llama 3.1 8B vs SFT
    (("meta-llama/Llama-3.1-8B", "default"), ("allenai/Llama-3.1-Tulu-3-8B-SFT", "default")),
    (("meta-llama/Llama-3.1-8B", "two_shot"), ("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot")),
    (("meta-llama/Llama-3.1-8B", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot_cot")),
    # Llama 2 70B vs SFT
    (("meta-llama/Llama-2-70b-hf", "default"), ("allenai/tulu-2-70b", "default")),
    (("meta-llama/Llama-2-70b-hf", "two_shot"), ("allenai/tulu-2-70b", "two_shot")),
    (("meta-llama/Llama-2-70b-hf", "two_shot_cot"), ("allenai/tulu-2-70b", "two_shot_cot")),
    # Llama 2 7B vs SFT
    (("meta-llama/Llama-2-7b-hf", "default"), ("allenai/tulu-2-7b", "default")),
    (("meta-llama/Llama-2-7b-hf", "two_shot"), ("allenai/tulu-2-7b", "two_shot")),
    (("meta-llama/Llama-2-7b-hf", "two_shot_cot"), ("allenai/tulu-2-7b", "two_shot_cot")),
]

# Base models vs their RLHF (Reinforcement Learning from Human Feedback) counterparts
BASE_VS_INSTRUCT_RLHF = [
    # Llama 3.1 70B vs RLHF
    # (("meta-llama/Llama-3.1-70B", "default"), ("meta-llama/Llama-3.1-70B-Instruct", "default")),
    # (("meta-llama/Llama-3.1-70B", "two_shot"), ("meta-llama/Llama-3.1-70B-Instruct", "two_shot")),
    # (("meta-llama/Llama-3.1-70B", "two_shot_cot"), ("meta-llama/Llama-3.1-70B-Instruct", "two_shot_cot")),
    
    # # Llama 3.1 8B vs RLHF
    # (("meta-llama/Llama-3.1-8B", "default"), ("meta-llama/Llama-3.1-8B-Instruct", "default")),
    # (("meta-llama/Llama-3.1-8B", "two_shot"), ("meta-llama/Llama-3.1-8B-Instruct", "two_shot")),
    # (("meta-llama/Llama-3.1-8B", "two_shot_cot"), ("meta-llama/Llama-3.1-8B-Instruct", "two_shot_cot")),
    
    # Llama 3.1 70B vs Tulu RLHF
    (("meta-llama/Llama-3.1-70B", "default"), ("allenai/Llama-3.1-Tulu-3-70B", "default")),
    (("meta-llama/Llama-3.1-70B", "two_shot"), ("allenai/Llama-3.1-Tulu-3-70B", "two_shot")),
    (("meta-llama/Llama-3.1-70B", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-70B", "two_shot_cot")),
    
    # Llama 3.1 8B vs Tulu RLHF variants
    ## PPO
    (("meta-llama/Llama-3.1-8B", "default"), ("allenai/Llama-3.1-Tulu-3-8B", "default")),
    (("meta-llama/Llama-3.1-8B", "two_shot"), ("allenai/Llama-3.1-Tulu-3-8B", "two_shot")),
    (("meta-llama/Llama-3.1-8B", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-8B", "two_shot_cot")),
    # ## GRPO
    # (("meta-llama/Llama-3.1-8B", "default"), ("allenai/Llama-3.1-Tulu-3.1-8B", "default")),
    # (("meta-llama/Llama-3.1-8B", "two_shot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot")),
    # (("meta-llama/Llama-3.1-8B", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot_cot")),
    # Llama 2 70B vs RLHF
    (("meta-llama/Llama-2-70b-hf", "default"), ("meta-llama/Llama-2-70b-chat-hf", "default")),
    (("meta-llama/Llama-2-70b-hf", "two_shot"), ("meta-llama/Llama-2-70b-chat-hf", "two_shot")),
    (("meta-llama/Llama-2-70b-hf", "two_shot_cot"), ("meta-llama/Llama-2-70b-chat-hf", "two_shot_cot")),
    # Llama 2 7B vs RLHF
    (("meta-llama/Llama-2-7b-hf", "default"), ("meta-llama/Llama-2-7b-chat-hf", "default")),
    (("meta-llama/Llama-2-7b-hf", "two_shot"), ("meta-llama/Llama-2-7b-chat-hf", "two_shot")),
    (("meta-llama/Llama-2-7b-hf", "two_shot_cot"), ("meta-llama/Llama-2-7b-chat-hf", "two_shot_cot")),
]


BASE_VS_INSTRUCT_GRPO = [
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "default"), ("allenai/Llama-3.1-Tulu-3.1-8B", "default")),
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot_cot")),
]

# Base models vs their DPO (Direct Preference Optimization) counterparts
BASE_VS_INSTRUCT_DPO = [
    
    # llama3.1 is DPO 
    (("meta-llama/Llama-3.1-70B", "default"), ("meta-llama/Llama-3.1-70B-Instruct", "default")),
    (("meta-llama/Llama-3.1-70B", "two_shot"), ("meta-llama/Llama-3.1-70B-Instruct", "two_shot")),
    (("meta-llama/Llama-3.1-70B", "two_shot_cot"), ("meta-llama/Llama-3.1-70B-Instruct", "two_shot_cot")),
    
    # Llama 3.1 is DPO 
    (("meta-llama/Llama-3.1-8B", "default"), ("meta-llama/Llama-3.1-8B-Instruct", "default")),
    (("meta-llama/Llama-3.1-8B", "two_shot"), ("meta-llama/Llama-3.1-8B-Instruct", "two_shot")),
    (("meta-llama/Llama-3.1-8B", "two_shot_cot"), ("meta-llama/Llama-3.1-8B-Instruct", "two_shot_cot")),
    
    # Llama 3.1 70B vs DPO
    (("meta-llama/Llama-3.1-70B", "default"), ("allenai/Llama-3.1-Tulu-3-70B-DPO", "default")),
    (("meta-llama/Llama-3.1-70B", "two_shot"), ("allenai/Llama-3.1-Tulu-3-70B-DPO", "two_shot")),
    (("meta-llama/Llama-3.1-70B", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-70B-DPO", "two_shot_cot")),
    # Llama 3.1 8B vs DPO
    (("meta-llama/Llama-3.1-8B", "default"), ("allenai/Llama-3.1-Tulu-3-8B-DPO", "default")),
    (("meta-llama/Llama-3.1-8B", "two_shot"), ("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot")),
    (("meta-llama/Llama-3.1-8B", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot_cot")),
    # Llama 2 70B vs DPO
    (("meta-llama/Llama-2-70b-hf", "default"), ("allenai/tulu-2-dpo-70b", "default")),
    (("meta-llama/Llama-2-70b-hf", "two_shot"), ("allenai/tulu-2-dpo-70b", "two_shot")),
    (("meta-llama/Llama-2-70b-hf", "two_shot_cot"), ("allenai/tulu-2-dpo-70b", "two_shot_cot")),
    # Llama 2 7B vs DPO
    (("meta-llama/Llama-2-7b-hf", "default"), ("allenai/tulu-2-dpo-7b", "default")),
    (("meta-llama/Llama-2-7b-hf", "two_shot"), ("allenai/tulu-2-dpo-7b", "two_shot")),
    (("meta-llama/Llama-2-7b-hf", "two_shot_cot"), ("allenai/tulu-2-dpo-7b", "two_shot_cot")),
]

# Combine base vs preference-optimized models (RLHF + DPO)
BASE_VS_INSTRUCT_PREFERENCE = BASE_VS_INSTRUCT_RLHF + BASE_VS_INSTRUCT_DPO + BASE_VS_INSTRUCT_GRPO

BASE_VS_INSTRUCT_RL = BASE_VS_INSTRUCT_RLHF + BASE_VS_INSTRUCT_GRPO

# Combined all base vs instruction models
BASE_VS_INSTRUCT_ALL = BASE_VS_INSTRUCT_SFT + BASE_VS_INSTRUCT_RLHF + BASE_VS_INSTRUCT_DPO

# Small vs Large comparisons for base models
SMALL_VS_LARGE_BASE = [
    # Llama 3.1 base
    (("meta-llama/Llama-3.1-8B", "default"), ("meta-llama/Llama-3.1-70B", "default")),
    (("meta-llama/Llama-3.1-8B", "two_shot"), ("meta-llama/Llama-3.1-70B", "two_shot")),
    (("meta-llama/Llama-3.1-8B", "two_shot_cot"), ("meta-llama/Llama-3.1-70B", "two_shot_cot")),
    # Llama 2 base
    (("meta-llama/Llama-2-7b-hf", "default"), ("meta-llama/Llama-2-70b-hf", "default")),
    (("meta-llama/Llama-2-7b-hf", "two_shot"), ("meta-llama/Llama-2-70b-hf", "two_shot")),
    (("meta-llama/Llama-2-7b-hf", "two_shot_cot"), ("meta-llama/Llama-2-70b-hf", "two_shot_cot")),
]

# Small vs Large comparisons for RLHF instruction-tuned models
SMALL_VS_LARGE_INSTRUCT_RLHF = [
    # # Llama 3.1 Instruct
    # (("meta-llama/Llama-3.1-8B-Instruct", "default"), ("meta-llama/Llama-3.1-70B-Instruct", "default")),
    # (("meta-llama/Llama-3.1-8B-Instruct", "two_shot"), ("meta-llama/Llama-3.1-70B-Instruct", "two_shot")),
    # (("meta-llama/Llama-3.1-8B-Instruct", "two_shot_cot"), ("meta-llama/Llama-3.1-70B-Instruct", "two_shot_cot")),
    # Llama 3.1 Tulu RLHF
    (("allenai/Llama-3.1-Tulu-3-8B", "default"), ("allenai/Llama-3.1-Tulu-3-70B", "default")),
    (("allenai/Llama-3.1-Tulu-3-8B", "two_shot"), ("allenai/Llama-3.1-Tulu-3-70B", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-8B", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-70B", "two_shot_cot")),
    # Llama 2 chat
    (("meta-llama/Llama-2-7b-chat-hf", "default"), ("meta-llama/Llama-2-70b-chat-hf", "default")),
    (("meta-llama/Llama-2-7b-chat-hf", "two_shot"), ("meta-llama/Llama-2-70b-chat-hf", "two_shot")),
    (("meta-llama/Llama-2-7b-chat-hf", "two_shot_cot"), ("meta-llama/Llama-2-70b-chat-hf", "two_shot_cot")),
]

# Small vs Large comparisons for SFT models
SMALL_VS_LARGE_INSTRUCT_SFT = [
    # Llama 3.1 Tulu SFT
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "default"), ("allenai/Llama-3.1-Tulu-3-70B-SFT", "default")),
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot"), ("allenai/Llama-3.1-Tulu-3-70B-SFT", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-70B-SFT", "two_shot_cot")),
    # Llama 2 Tulu
    (("allenai/tulu-2-7b", "default"), ("allenai/tulu-2-70b", "default")),
    (("allenai/tulu-2-7b", "two_shot"), ("allenai/tulu-2-70b", "two_shot")),
    (("allenai/tulu-2-7b", "two_shot_cot"), ("allenai/tulu-2-70b", "two_shot_cot")),
]

# Small vs Large comparisons for DPO models
SMALL_VS_LARGE_INSTRUCT_DPO = [
    # Llama 3.1 Instruct
    (("meta-llama/Llama-3.1-8B-Instruct", "default"), ("meta-llama/Llama-3.1-70B-Instruct", "default")),
    (("meta-llama/Llama-3.1-8B-Instruct", "two_shot"), ("meta-llama/Llama-3.1-70B-Instruct", "two_shot")),
    (("meta-llama/Llama-3.1-8B-Instruct", "two_shot_cot"), ("meta-llama/Llama-3.1-70B-Instruct", "two_shot_cot")),
    
    # Llama 3.1 Tulu DPO
    (("allenai/Llama-3.1-Tulu-3-8B-DPO", "default"), ("allenai/Llama-3.1-Tulu-3-70B-DPO", "default")),
    (("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot"), ("allenai/Llama-3.1-Tulu-3-70B-DPO", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-70B-DPO", "two_shot_cot")),
    # Llama 2 Tulu DPO
    (("allenai/tulu-2-dpo-7b", "default"), ("allenai/tulu-2-dpo-70b", "default")),
    (("allenai/tulu-2-dpo-7b", "two_shot"), ("allenai/tulu-2-dpo-70b", "two_shot")),
    (("allenai/tulu-2-dpo-7b", "two_shot_cot"), ("allenai/tulu-2-dpo-70b", "two_shot_cot")),
]

# Combined small vs large comparisons for all instruction models
SMALL_VS_LARGE_INSTRUCT = SMALL_VS_LARGE_INSTRUCT_RLHF + SMALL_VS_LARGE_INSTRUCT_SFT + SMALL_VS_LARGE_INSTRUCT_DPO

SMALL_VS_LARGE_PREFERENCE = SMALL_VS_LARGE_INSTRUCT_RLHF + SMALL_VS_LARGE_INSTRUCT_DPO 

# Combined small vs large comparisons for all models
SMALL_VS_LARGE_ALL = SMALL_VS_LARGE_BASE + SMALL_VS_LARGE_INSTRUCT

# Zero-shot vs Two-shot for base models
ZERO_VS_TWO_SHOT_BASE = [
    # Llama 3.1 base
    (("meta-llama/Llama-3.1-8B", "default"), ("meta-llama/Llama-3.1-8B", "two_shot")),
    (("meta-llama/Llama-3.1-70B", "default"), ("meta-llama/Llama-3.1-70B", "two_shot")),
    # Llama 2 base
    (("meta-llama/Llama-2-7b-hf", "default"), ("meta-llama/Llama-2-7b-hf", "two_shot")),
    (("meta-llama/Llama-2-70b-hf", "default"), ("meta-llama/Llama-2-70b-hf", "two_shot")),
]

# Zero-shot vs Two-shot for RLHF instruction-tuned models
ZERO_VS_TWO_SHOT_INSTRUCT_RLHF = [
    # Llama 3.1 Instruct
    # (("meta-llama/Llama-3.1-8B-Instruct", "default"), ("meta-llama/Llama-3.1-8B-Instruct", "two_shot")),
    # (("meta-llama/Llama-3.1-70B-Instruct", "default"), ("meta-llama/Llama-3.1-70B-Instruct", "two_shot")),
    # Llama 3.1 Tulu RLHF variants
    (("allenai/Llama-3.1-Tulu-3-8B", "default"), ("allenai/Llama-3.1-Tulu-3-8B", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-70B", "default"), ("allenai/Llama-3.1-Tulu-3-70B", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3.1-8B", "default"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot")),
    # Llama 2 chat
    (("meta-llama/Llama-2-7b-chat-hf", "default"), ("meta-llama/Llama-2-7b-chat-hf", "two_shot")),
    (("meta-llama/Llama-2-70b-chat-hf", "default"), ("meta-llama/Llama-2-70b-chat-hf", "two_shot")),
]



# Zero-shot vs Two-shot for SFT models
ZERO_VS_TWO_SHOT_INSTRUCT_SFT = [
    # Llama 3.1 Tulu SFT
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "default"), ("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-70B-SFT", "default"), ("allenai/Llama-3.1-Tulu-3-70B-SFT", "two_shot")),
    # Llama 2 Tulu
    (("allenai/tulu-2-7b", "default"), ("allenai/tulu-2-7b", "two_shot")),
    (("allenai/tulu-2-70b", "default"), ("allenai/tulu-2-70b", "two_shot")),
]

# Zero-shot vs Two-shot for DPO models
ZERO_VS_TWO_SHOT_INSTRUCT_DPO = [
    
    # Llama 3.1 Instruct
    (("meta-llama/Llama-3.1-8B-Instruct", "default"), ("meta-llama/Llama-3.1-8B-Instruct", "two_shot")),
    (("meta-llama/Llama-3.1-70B-Instruct", "default"), ("meta-llama/Llama-3.1-70B-Instruct", "two_shot")),
    
    # Llama 3.1 Tulu DPO
    (("allenai/Llama-3.1-Tulu-3-8B-DPO", "default"), ("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-70B-DPO", "default"), ("allenai/Llama-3.1-Tulu-3-70B-DPO", "two_shot")),
    # Llama 2 Tulu DPO
    (("allenai/tulu-2-dpo-7b", "default"), ("allenai/tulu-2-dpo-7b", "two_shot")),
    (("allenai/tulu-2-dpo-70b", "default"), ("allenai/tulu-2-dpo-70b", "two_shot")),
]

# Combined zero-shot vs two-shot for all instruction models
ZERO_VS_TWO_SHOT_INSTRUCT = ZERO_VS_TWO_SHOT_INSTRUCT_RLHF + ZERO_VS_TWO_SHOT_INSTRUCT_SFT + ZERO_VS_TWO_SHOT_INSTRUCT_DPO

# Combined zero-shot vs two-shot for all models
ZERO_VS_TWO_SHOT_ALL = ZERO_VS_TWO_SHOT_BASE + ZERO_VS_TWO_SHOT_INSTRUCT

# Two-shot vs Two-shot-CoT comparisons
TWO_SHOT_VS_TWO_SHOT_COT_ALL = [
    # Llama 3.1 models
    (("meta-llama/Llama-3.1-8B", "two_shot"), ("meta-llama/Llama-3.1-8B", "two_shot_cot")),
    (("meta-llama/Llama-3.1-70B", "two_shot"), ("meta-llama/Llama-3.1-70B", "two_shot_cot")),
    (("meta-llama/Llama-3.1-8B-Instruct", "two_shot"), ("meta-llama/Llama-3.1-8B-Instruct", "two_shot_cot")),
    (("meta-llama/Llama-3.1-70B-Instruct", "two_shot"), ("meta-llama/Llama-3.1-70B-Instruct", "two_shot_cot")),
    # Llama 3.1 Tulu models
    (("allenai/Llama-3.1-Tulu-3-8B", "two_shot"), ("allenai/Llama-3.1-Tulu-3-8B", "two_shot_cot")),
    (("allenai/Llama-3.1-Tulu-3-70B", "two_shot"), ("allenai/Llama-3.1-Tulu-3-70B", "two_shot_cot")),
    (("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot_cot")),
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot"), ("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot_cot")),
    (("allenai/Llama-3.1-Tulu-3-70B-SFT", "two_shot"), ("allenai/Llama-3.1-Tulu-3-70B-SFT", "two_shot_cot")),
    (("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot"), ("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot_cot")),
    (("allenai/Llama-3.1-Tulu-3-70B-DPO", "two_shot"), ("allenai/Llama-3.1-Tulu-3-70B-DPO", "two_shot_cot")),
    # Llama 2 models
    (("meta-llama/Llama-2-7b-hf", "two_shot"), ("meta-llama/Llama-2-7b-hf", "two_shot_cot")),
    (("meta-llama/Llama-2-70b-hf", "two_shot"), ("meta-llama/Llama-2-70b-hf", "two_shot_cot")),
    (("meta-llama/Llama-2-7b-chat-hf", "two_shot"), ("meta-llama/Llama-2-7b-chat-hf", "two_shot_cot")),
    (("meta-llama/Llama-2-70b-chat-hf", "two_shot"), ("meta-llama/Llama-2-70b-chat-hf", "two_shot_cot")),
    # Llama 2 Tulu models
    (("allenai/tulu-2-7b", "two_shot"), ("allenai/tulu-2-7b", "two_shot_cot")),
    (("allenai/tulu-2-70b", "two_shot"), ("allenai/tulu-2-70b", "two_shot_cot")),
    (("allenai/tulu-2-dpo-7b", "two_shot"), ("allenai/tulu-2-dpo-7b", "two_shot_cot")),
    (("allenai/tulu-2-dpo-70b", "two_shot"), ("allenai/tulu-2-dpo-70b", "two_shot_cot")),
]

# SFT vs DPO comparisons (Tulu models only)
SFT_VS_DPO = [
    # Llama 3.1 Tulu 8B SFT vs DPO
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "default"), ("allenai/Llama-3.1-Tulu-3-8B-DPO", "default")),
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot"), ("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot_cot")),
    
    # Llama 3.1 Tulu 70B SFT vs DPO
    (("allenai/Llama-3.1-Tulu-3-70B-SFT", "default"), ("allenai/Llama-3.1-Tulu-3-70B-DPO", "default")),
    (("allenai/Llama-3.1-Tulu-3-70B-SFT", "two_shot"), ("allenai/Llama-3.1-Tulu-3-70B-DPO", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-70B-SFT", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-70B-DPO", "two_shot_cot")),
    
    # Tulu-2 SFT vs DPO
    (("allenai/tulu-2-7b", "default"), ("allenai/tulu-2-dpo-7b", "default")),
    (("allenai/tulu-2-7b", "two_shot"), ("allenai/tulu-2-dpo-7b", "two_shot")),
    (("allenai/tulu-2-7b", "two_shot_cot"), ("allenai/tulu-2-dpo-7b", "two_shot_cot")),
    
    # Tulu-2 SFT vs DPO 
    (("allenai/tulu-2-70b", "default"), ("allenai/tulu-2-dpo-70b", "default")),
    (("allenai/tulu-2-70b", "two_shot"), ("allenai/tulu-2-dpo-70b", "two_shot")),
    (("allenai/tulu-2-70b", "two_shot_cot"), ("allenai/tulu-2-dpo-70b", "two_shot_cot")),    
    
]

# SFT vs RLHF comparisons (Tulu models only)
SFT_VS_RLHF = [
    # Llama 3.1 Tulu 8B SFT vs RLHF (Tulu-3)
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "default"), ("allenai/Llama-3.1-Tulu-3-8B", "default")),
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot"), ("allenai/Llama-3.1-Tulu-3-8B", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-8B", "two_shot_cot")),
    
    # Llama 3.1 Tulu 8B SFT vs RLHF (Tulu-3.1)
    # (("allenai/Llama-3.1-Tulu-3-8B-SFT", "default"), ("allenai/Llama-3.1-Tulu-3.1-8B", "default")),
    # (("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot")),
    # (("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot_cot")),
    
    # Llama 3.1 Tulu 70B SFT vs RLHF
    (("allenai/Llama-3.1-Tulu-3-70B-SFT", "default"), ("allenai/Llama-3.1-Tulu-3-70B", "default")),
    (("allenai/Llama-3.1-Tulu-3-70B-SFT", "two_shot"), ("allenai/Llama-3.1-Tulu-3-70B", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-70B-SFT", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-70B", "two_shot_cot")),
]

SFT_VS_GRPO = [
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "default"), ("allenai/Llama-3.1-Tulu-3.1-8B", "default")),
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-8B-SFT", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot_cot")),
]

DPO_VS_GRPO = [
    (("allenai/Llama-3.1-Tulu-3-8B-DPO", "default"), ("allenai/Llama-3.1-Tulu-3.1-8B", "default")),
    (("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot_cot")),
]

SFT_VS_PREFERENCE = SFT_VS_RLHF + SFT_VS_DPO + SFT_VS_GRPO

SFT_VS_RL = SFT_VS_RLHF + SFT_VS_GRPO

RLHF_VS_GRPO = [
    (("allenai/Llama-3.1-Tulu-3-8B", "default"), ("allenai/Llama-3.1-Tulu-3.1-8B", "default")),
    (("allenai/Llama-3.1-Tulu-3-8B", "two_shot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-8B", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot_cot")),
]

# Combined SFT vs preference-optimized models (RLHF + DPO)
SFT_VS_PREFERENCE = SFT_VS_RLHF + SFT_VS_DPO + SFT_VS_GRPO

DPO_VS_RLHF = [
    # Llama 3.1 Tulu 8B DPO vs RLHF (Tulu-3)
    (("allenai/Llama-3.1-Tulu-3-8B-DPO", "default"), ("allenai/Llama-3.1-Tulu-3-8B", "default")),
    (("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot"), ("allenai/Llama-3.1-Tulu-3-8B", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-8B", "two_shot_cot")),
    
    
    
    # # Llama 3.1 Tulu 8B DPO vs RLHF (Tulu-3.1)
    # (("allenai/Llama-3.1-Tulu-3-8B-DPO", "default"), ("allenai/Llama-3.1-Tulu-3.1-8B", "default")),
    # (("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot")),
    # (("allenai/Llama-3.1-Tulu-3-8B-DPO", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3.1-8B", "two_shot_cot")),
    
    # Llama 3.1 Tulu 70B DPO vs RLHF
    (("allenai/Llama-3.1-Tulu-3-70B-DPO", "default"), ("allenai/Llama-3.1-Tulu-3-70B", "default")),
    (("allenai/Llama-3.1-Tulu-3-70B-DPO", "two_shot"), ("allenai/Llama-3.1-Tulu-3-70B", "two_shot")),
    (("allenai/Llama-3.1-Tulu-3-70B-DPO", "two_shot_cot"), ("allenai/Llama-3.1-Tulu-3-70B", "two_shot_cot")),
    
    
]

DPO_VS_RL = DPO_VS_RLHF + DPO_VS_GRPO



# Define explicit model and template pairs for comparison
# Format: ((model1, template1), (model2, template2))
# Each pair represents a specific comparison to be made between model+template combinations
# MODEL_TEMPLATE_PAIRS = {
#     # Base models vs their Instruction-tuned counterparts
#     # These comparisons assess the impact of instruction tuning on diversity metrics
#     "base_vs_instruct": BASE_VS_INSTRUCT_ALL,
#     "base_vs_instruct_sft": BASE_VS_INSTRUCT_SFT,
#     "base_vs_instruct_rlhf": BASE_VS_INSTRUCT_RLHF,
    
#     # Small vs Large models (comparing across model sizes)
#     # These comparisons assess the impact of model scale on diversity metrics
#     "small_vs_large": SMALL_VS_LARGE_ALL,
#     "small_vs_large_sft": SMALL_VS_LARGE_SFT,
#     "small_vs_large_rlhf": SMALL_VS_LARGE_RLHF,
#     "small_vs_large_base": SMALL_VS_LARGE_BASE,
#     "small_vs_large_instruct": SMALL_VS_LARGE_INSTRUCT,
    
#     # Zero-shot vs Two-shot prompting
#     # These comparisons assess the impact of few-shot examples on diversity metrics
#     "zero_vs_two_shot": ZERO_VS_TWO_SHOT_ALL,
#     "zero_vs_two_shot_base": ZERO_VS_TWO_SHOT_BASE,
#     "zero_vs_two_shot_instruct": ZERO_VS_TWO_SHOT_INSTRUCT,
# }

MODEL_TEMPLATE_PAIRS = {
    # Base models vs their Instruction-tuned counterparts
    # These comparisons assess the impact of instruction tuning on diversity metrics
    "base_vs_instruct": BASE_VS_INSTRUCT_ALL,
    "base_vs_instruct_sft": BASE_VS_INSTRUCT_SFT,
    "base_vs_instruct_rlhf": BASE_VS_INSTRUCT_RLHF,
    "base_vs_instruct_dpo": BASE_VS_INSTRUCT_DPO,
    "base_vs_instruct_preference": BASE_VS_INSTRUCT_PREFERENCE,
    "base_vs_instruct_grpo": BASE_VS_INSTRUCT_GRPO,
    "base_vs_instruct_rl": BASE_VS_INSTRUCT_RL,
    # Small vs Large models (comparing across model sizes)
    # These comparisons assess the impact of model scale on diversity metrics
    "small_vs_large": SMALL_VS_LARGE_ALL,
    "small_vs_large_base": SMALL_VS_LARGE_BASE,
    "small_vs_large_instruct": SMALL_VS_LARGE_INSTRUCT,
    "small_vs_large_instruct_rlhf": SMALL_VS_LARGE_INSTRUCT_RLHF,
    "small_vs_large_instruct_sft": SMALL_VS_LARGE_INSTRUCT_SFT,
    "small_vs_large_instruct_dpo": SMALL_VS_LARGE_INSTRUCT_DPO,
    "small_vs_large_preference": SMALL_VS_LARGE_PREFERENCE,
    # Zero-shot vs Two-shot prompting
    # These comparisons assess the impact of few-shot examples on diversity metrics
    "zero_vs_two_shot": ZERO_VS_TWO_SHOT_ALL,
    "zero_vs_two_shot_base": ZERO_VS_TWO_SHOT_BASE,
    "zero_vs_two_shot_instruct": ZERO_VS_TWO_SHOT_INSTRUCT,
    "zero_vs_two_shot_instruct_rlhf": ZERO_VS_TWO_SHOT_INSTRUCT_RLHF,
    "zero_vs_two_shot_instruct_sft": ZERO_VS_TWO_SHOT_INSTRUCT_SFT,
    "zero_vs_two_shot_instruct_dpo": ZERO_VS_TWO_SHOT_INSTRUCT_DPO,
    
    # Two-shot vs Two-shot-CoT prompting
    # These comparisons assess the impact of chain-of-thought reasoning on diversity metrics
    "two_shot_vs_two_shot_cot": TWO_SHOT_VS_TWO_SHOT_COT_ALL,
    
    # Comparisons between different fine-tuning methods
    # These assess the impact of different preference optimization techniques
    "sft_vs_dpo": SFT_VS_DPO,
    "sft_vs_rlhf": SFT_VS_RLHF,
    "sft_vs_preference": SFT_VS_PREFERENCE,
    "sft_vs_rl": SFT_VS_RL,
    "dpo_vs_rlhf": DPO_VS_RLHF,
    "sft_vs_grpo": SFT_VS_GRPO,
    "dpo_vs_grpo": DPO_VS_GRPO,
    "rlhf_vs_grpo": RLHF_VS_GRPO,
    "dpo_vs_rl": DPO_VS_RL,
}

def check_for_missing_model_prompt_pairs(data: pd.DataFrame) -> None:
    
    all_model_prompt_pairs = set(data[["model", "template"]].apply(tuple, axis=1))
    all_desired_model_prompt_pairs = set()
    for model_pair in MODEL_TEMPLATE_PAIRS.values():
        for pair in model_pair:
            all_desired_model_prompt_pairs.add(tuple(pair[0]))
            all_desired_model_prompt_pairs.add(tuple(pair[1]))
    
    missing_pairs = all_desired_model_prompt_pairs - all_model_prompt_pairs
    if missing_pairs:
        print("Missing model-prompt pairs:")
        for pair in missing_pairs:
            print(pair)


def load_and_process_data(tsv_paths: List[str], model_order: List[str]) -> pd.DataFrame:
    """
    Load data from TSV files and process it for analysis.
    
    Args:
        tsv_paths: List of paths to driver_stats.tsv files
        model_order: List of models in desired sorting order
        
    Returns:
        Processed DataFrame
    """
    # Load and combine all TSV files
    dfs = []
    for path in tsv_paths:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            exit(1)
        try:
            data = pd.read_csv(path, sep='\t')
            print(f"Loaded {path}: {data.shape[0]} rows, {data.shape[1]} columns")
            dfs.append(data)
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
    
    if not dfs:
        raise ValueError("No valid data files found")
    
    # use only the examples where temperature == 1.0 and top_p == 1.0 
    total_rows = sum(len(df) for df in dfs)
    
    # import pdb; pdb.set_trace()
    
    new_dfs = []
    for df in dfs:
        df = df[df["temperature"] == 1.0]
        df = df[df["top_p"] == 1.0]
        new_dfs.append(df)
        
    dfs = new_dfs
        
    print(f"Filtered {total_rows - sum(len(df) for df in dfs)} rows")
    
    # Check column consistency across files
    all_columns = set()
    for i, df in enumerate(dfs):
        all_columns.update(df.columns)
    
    for i, df in enumerate(dfs):
        missing = all_columns - set(df.columns)
        if missing:
            print(f"File {i} ({tsv_paths[i]}) is missing columns: {missing}")
    
    # Combine all dataframes
    data = pd.concat(dfs, ignore_index=True)
    print(f"Combined data shape: {data.shape}")
    
    # Check for NaN values in key columns before processing
    nan_check_before = {col: data[col].isna().sum() for col in data.columns if data[col].isna().sum() > 0}
    if nan_check_before:
        print("NaN values before processing:")
        for col, count in sorted(nan_check_before.items(), key=lambda x: x[1], reverse=True):
            print(f"  {col}: {count} NaNs ({count/len(data)*100:.2f}%)")
    
    # Remove rows with errors
    if "all_coherence" in data.columns:
        orig_len = len(data)
        data = data[~data["all_coherence"].apply(
            lambda row: any(word in str(row) for word in ["ERROR", "error"])
        )]
        filtered_len = len(data)
        if orig_len > filtered_len:
            print(f"Removed {orig_len - filtered_len} rows with errors in all_coherence")
    
    # # Calculate coherence from raw values if needed
    # try:
    #     # First check if the component columns exist and have non-NaN values
    #     component_cols = ["all_semantic_count_wcoh_nonempty_woutput", 
    #                       "all_semantic_proportion_wcoh_nonempty_woutput"]
        
    #     for col in component_cols:
    #         if col in data.columns:
    #             nan_count = data[col].isna().sum()
    #             zero_count = (data[col] == 0).sum()
    #             if nan_count > 0:
    #                 print(f"Warning: {col} has {nan_count} NaN values")
    #             if zero_count > 0:
    #                 print(f"Warning: {col} has {zero_count} zero values")
        
    #     # Calculate coherence, avoiding division by zero
    #     if all(col in data.columns for col in component_cols):
    #         data["All Coherence (NonEmpty w/ Output)"] = np.where(
    #             data["all_semantic_proportion_wcoh_nonempty_woutput"] > 0,
    #             (data["all_semantic_count_wcoh_nonempty_woutput"] / 
    #             data["all_semantic_proportion_wcoh_nonempty_woutput"]) * 100,
    #             0  # Default value when denominator is zero
    #         )
    #         print("Calculated coherence from raw values")
    # except Exception as e:
    #     print(f"Error calculating coherence: {str(e)}")
    
    # Apply renaming from the global RENAME_DICT, handling potential missing columns
    for old_col, new_col in RENAME_DICT.items():
        if old_col in data.columns:
            data[new_col] = data[old_col]
            nan_count = data[new_col].isna().sum()
            if nan_count > 0:
                print(f"Warning: {new_col} has {nan_count} NaN values after renaming from {old_col}")
                # raise ValueError(f"NaN values in {new_col} after renaming from {old_col}")
    
    # Remove rows with NaN in the model column
    orig_len = len(data)
    data = data.dropna(subset=["model"])
    filtered_len = len(data)
    if orig_len > filtered_len:
        print(f"Removed {orig_len - filtered_len} rows with NaN model values")
    
    # Check for NaN values in the metrics columns
    nan_in_metrics = {metric: data[metric].isna().sum() for metric in METRICS 
                     if metric in data.columns and data[metric].isna().sum() > 0}
    
    # if nan_in_metrics:
        # print("Warning: NaN values in metrics columns:")
        # for metric, count in nan_in_metrics.items():
        #     print(f"  {metric}: {count} NaNs ({count/len(data)*100:.2f}%)")
        
        # Print examples of rows with NaN metrics
        # print("\nExample rows with NaN metrics:")
        # for metric in nan_in_metrics:
        #     nan_rows = data[data[metric].isna()].head(3)
        #     if not nan_rows.empty:
        #         print(f"\nRows with NaN in {metric}:")
        #         print(nan_rows[["model", "template"] + [m for m in METRICS if m in data.columns]])
    
    # Ensure all required columns exist
    required_columns = ["model", "template"] + list(set(METRICS) & set(data.columns))
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print("Available columns:", list(data.columns))
        if "model" in missing_columns or "template" in missing_columns:
            raise ValueError(f"Critical columns missing: {missing_columns}")
    
    # Add debugging information
    print(f"Data shape after processing: {data.shape}")
    print(f"Unique models: {data['model'].nunique()}")
    print(f"Models with NaN values: {data['model'].isna().sum()}")
    print(f"Available metrics: {[m for m in METRICS if m in data.columns]}")
    
    # Convert model column to categorical with the specified order
    if model_order:
        # Check for models in data but not in model_order
        models_not_ordered = set(data["model"].unique()) - set(model_order)
        if models_not_ordered:
            print(f"Warning: Models not in model_order: {models_not_ordered}")
        
        data["model"] = pd.Categorical(data["model"], categories=model_order, ordered=True)
        data = data.sort_values(by=["model", "template"])
        
    check_for_missing_model_prompt_pairs(data)
    
    return data


#------------------------------------------------------------------------------
# Statistical Analysis Functions 
#------------------------------------------------------------------------------

def cohens_d(d1: np.ndarray, d2: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate Cohen's d effect size for paired samples.
    
    Cohen's d measures the standardized difference between two means.
    Common interpretations: 0.2=small effect, 0.5=medium effect, 0.8=large effect
    
    Args:
        d1: First array of values
        d2: Second array of values
        
    Returns:
        Tuple of (cohen's d, mean difference, pooled standard deviation)
    """
    # Remove NaN values in paired manner
    mask = ~np.isnan(d1) & ~np.isnan(d2)
    d1, d2 = d1[mask], d2[mask]
    
    # Calculate differences and statistics
    diff = d1.mean() - d2.mean()
    
    n1, n2 = len(d1), len(d2)
    
    # Safety check for empty arrays
    if n1 <= 1 or n2 <= 1:
        return 0.0, diff, 0.0
    
    s1 = np.sum((d1 - d1.mean()) ** 2) / (n1 - 1)
    s2 = np.sum((d2 - d2.mean()) ** 2) / (n2 - 1)
    
    # Pooled standard deviation with improved handling
    if (n1 + n2 - 2) <= 0:
        return 0.0, diff, 0.0
    
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    
    # Prevent division by zero
    if pooled_std <= 0:
        return 0.0, diff, 0.0
    
    # Cohen's d is mean difference divided by pooled standard deviation
    d_value = diff / pooled_std
    
    return d_value, diff, pooled_std


def perform_statistical_test(group1: np.ndarray, group2: np.ndarray, 
                            group1_name: str, group2_name: str) -> Dict[str, Any]:
    """
    Perform statistical analysis comparing two groups.
    
    Uses Wilcoxon signed-rank test (non-parametric paired test) and Cohen's d for effect size.
    
    Args:
        group1: First group values
        group2: Second group values
        group1_name: Name of first group (e.g., "base", "small")
        group2_name: Name of second group (e.g., "instruct", "large")
        
    Returns:
        Dictionary of statistical results including sample size, p-value, and effect sizes
    """
    # Remove NaN values in paired manner
    group1 = np.array(group1, dtype=float)
    group2 = np.array(group2, dtype=float)
    try: 
        mask = ~np.isnan(group1) & ~np.isnan(group2)
    except:
        import pdb; pdb.set_trace()
        mask = ~pd.isna(group1) & ~pd.isna(group2)
    g1, g2 = group1[mask], group2[mask]
    
    if len(g1) == 0 or len(g2) == 0:
        return {
            "n": 0,
            "wilcoxon_statistic": None,
            "wilcoxon_p": None,
            "cohens_d": None,
            "mean_diff": None,
            "median_diff": None,
            "direction": "insufficient_data",
            "significant": False,
            "positive_diffs": 0,
            "negative_diffs": 0
        }
    
    # Calculate differences
    differences = g2 - g1
    
    # Count positive and negative differences
    pos_diff = np.sum(differences > 0)
    neg_diff = np.sum(differences < 0)
    
    # Calculate mean and median differences
    mean_diff = differences.mean()
    median_diff = np.median(differences)
    
    # Determine direction
    if mean_diff > 0:
        direction = f"{group2_name}_higher"
    elif mean_diff < 0:
        direction = f"{group1_name}_higher"
    else:
        direction = "no_difference"
    
    # Calculate Cohen's d
    d_value, _, _ = cohens_d(g1, g2)
    
    # Wilcoxon signed-rank test
    try:
        if np.all(differences == 0):
            wilcoxon_statistic, p_value = None, 1.0
        else:
            wilcoxon_statistic, p_value = stats.wilcoxon(g1, g2, alternative='two-sided')
    except ValueError:
        # Handle cases with insufficient non-zero differences
        wilcoxon_statistic, p_value = None, None
    
    return {
        "n": len(g1),
        "wilcoxon_statistic": float(wilcoxon_statistic) if wilcoxon_statistic is not None else None,
        "wilcoxon_p": float(p_value) if p_value is not None else None,
        "cohens_d": float(d_value),
        "mean_diff": float(mean_diff),
        "median_diff": float(median_diff),
        "direction": direction,
        "significant": bool(p_value is not None and p_value < 0.05),
        "positive_diffs": int(pos_diff),
        "negative_diffs": int(neg_diff)
    }

def analyze_model_template_pairs(data: pd.DataFrame, 
                              pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]], 
                              metrics: List[str],
                              group1_name: str,
                              group2_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Analyze pairs of (model, template) combinations for each specified metric.
    
    This function works by:
    1. For each metric (e.g., coherence, semantic_diversity), it collects values for all pairs
    2. For each (model1,template1), (model2,template2) pair, it extracts their metric values
    3. It then performs statistical tests on the collected values
    4. Results include both summary statistics and detailed pair-by-pair information
    
    Args:
        data: Processed DataFrame with model, template, and metric columns
        pairs: List of ((model1, template1), (model2, template2)) pairs to compare
        metrics: List of metrics to analyze (e.g., coherence, semantic_diversity)
        group1_name: Name for first group in pair (e.g., "base", "small", "zero_shot")
        group2_name: Name for second group in pair (e.g., "instruct", "large", "two_shot")
        
    Returns:
        Dictionary of results by metric
    """
    results = {}
    
    # For each metric
    for metric in metrics:
        group1_values = []
        group2_values = []
        pair_info = []
        
        # For each model-template pair
        for (model1, template1), (model2, template2) in pairs:
            # Get data for each (model, template) combination
            model1_data = data[(data["model"] == model1) & (data["template"] == template1)]
            model2_data = data[(data["model"] == model2) & (data["template"] == template2)]
            
            
            # Skip if either model+template doesn't have data
            if model1_data.empty or model2_data.empty:
                print(f"Warning: Missing data for metric {metric} pair ({model1}, {template1}) vs ({model2}, {template2})")
                # import pdb; pdb.set_trace()
                continue
            
            # Extract values
            try:
                value1 = model1_data[metric].values[0]
                value2 = model2_data[metric].values[0]
                
                # make float
                value1 = float(value1)
                value2 = float(value2)
                
                # Check for NaN values
                # if np.isnan(value1) or np.isnan(value2):
                if pd.isna(value1) or pd.isna(value2):
                    print(f"Warning: NaN value for ({model1}, {template1}) vs ({model2}, {template2})")
                    continue
                    
                group1_values.append(value1)
                group2_values.append(value2)
                pair_info.append((model1, template1, model2, template2, value1, value2))
                
            except (IndexError, KeyError) as e:
                print(f"Error extracting values for ({model1}, {template1}) vs ({model2}, {template2}): {e}")
        
        # Perform statistical tests if we have data
        if group1_values and group2_values:
            stat_results = perform_statistical_test(
                np.array(group1_values),
                np.array(group2_values),
                group1_name,
                group2_name
            )
            
            # Add detailed information about each pair
            stat_results["pairs"] = [
                {
                    f"{group1_name}_model": p[0],
                    f"{group1_name}_template": p[1],
                    f"{group2_name}_model": p[2],
                    f"{group2_name}_template": p[3],
                    f"{group1_name}_value": float(p[4]),
                    f"{group2_name}_value": float(p[5]),
                    "difference": float(p[5] - p[4])
                }
                for p in pair_info
            ]
            
            results[metric] = stat_results
        else:
            print(f"Warning: No valid pairs found for metric {metric}")
            results[metric] = {
                "n": 0,
                "error": "No valid pairs found"
            }
    
    return results

def save_json(results: Dict[str, Any], output_file: str) -> None:
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

def json_to_csv(results: Dict[str, Any], output_file: str) -> None:
    """Convert hierarchical JSON results to flat CSV format."""
    rows = []
    
    # Flatten the hierarchical structure
    for comparison_type, metrics in results.items():
        for metric_name, stats in metrics.items():
            if isinstance(stats, dict):  # Skip if stats is not a dictionary
                row = {
                    "comparison_type": comparison_type,
                    "metric": metric_name
                }
                # Add all statistics as columns
                for stat_name, stat_value in stats.items():
                    if stat_name != "pairs":  # Skip the detailed pairs info
                        row[stat_name] = stat_value
                
                rows.append(row)
    
    # Convert to DataFrame and save as CSV
    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f"CSV saved to {output_file}")

def save_yaml(results: Dict[str, Any], output_file: str) -> None:
    """Save results to YAML file."""
    with open(output_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    print(f"YAML saved to {output_file}")
    
def json_to_yaml(results: Dict[str, Any], output_file: str) -> None:
    """Save results to YAML file."""
    with open(output_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    print(f"YAML saved to {output_file}")


def main():
    """
    Main function to run the analysis.
    
    This orchestrates the entire analysis workflow:
    1. Parse command line arguments
    2. Load and process the data from TSV files
    3. Run comparisons (base vs. instruct, small vs. large, zero-shot vs. two-shot)
    4. Save results in the requested formats
    """
    parser = argparse.ArgumentParser(description='Analyze diversity metrics from TSV files')
    parser.add_argument('--tsv_paths', nargs='+', required=False, 
                        help='Paths to driver_stats.tsv files')
    parser.add_argument('--output_prefix', default='diversity_analysis',
                        help='Prefix for output files')
    parser.add_argument('--output_format', choices=['json', 'csv', 'yaml', 'all'], 
                        default='all', help='Output format')
    parser.add_argument('--pairs_file', help='JSON file defining model pairs to compare')
    parser.add_argument('--model_order_file', help='JSON file with model sorting order')
    args = parser.parse_args()
    
    if args.tsv_paths is None:
        args.tsv_paths = PATHS
    
    # Load model order if provided, otherwise use default
    model_order = DEFAULT_MODEL_ORDER
    if args.model_order_file:
        try:
            with open(args.model_order_file, 'r') as f:
                model_order = json.load(f)
            print(f"Loaded custom model order from {args.model_order_file}")
        except Exception as e:
            print(f"Error loading model order, using default: {e}")
    
    # Load and process data
    data = load_and_process_data(args.tsv_paths, model_order)
    # data.to_csv(f"{args.output_prefix}_raw_data.csv")
    data[["model", "template"] + METRICS].to_csv(f"{args.output_prefix}_raw_data.csv")
    
    # Load model-template pairs from file or use defaults
    model_template_pairs = MODEL_TEMPLATE_PAIRS
    if args.pairs_file:
        try:
            with open(args.pairs_file, 'r') as f:
                model_template_pairs = json.load(f)
            print(f"Loaded custom model pairs from {args.pairs_file}")
        except Exception as e:
            print(f"Error loading model pairs, using default: {e}")
    
    # Run analyses - using the same function for all comparisons
    print("\nRunning analyses...")
    results = {}
    # import pdb; pdb.set_trace()
    for comparison_type, pairs in model_template_pairs.items():
        # Extract the comparison labels from the comparison type (e.g. "base_vs_instruct" -> "base", "instruct")
        label1, label2 = comparison_type.split("_vs_")
        results[comparison_type] = analyze_model_template_pairs(
            data, pairs, METRICS, label1, label2
        )
    
    # Save results in requested formats
    print("\nSaving results...")
    if args.output_format in ['json', 'all']:
        save_json(results, f"{args.output_prefix}.json")

    if args.output_format in ['csv', 'all']:
        json_to_csv(results, f"{args.output_prefix}.csv")
        
        # Load the CSV we just created
        post_train_df = pd.read_csv(f"{args.output_prefix}.csv")
        
        # Apply the same filtering as you did with the tuning data
        comparison_types = ["base_vs_instruct", "base_vs_instruct_sft", "base_vs_instruct_dpo", "base_vs_instruct_rl", "base_vs_instruct_preference", "sft_vs_dpo", "sft_vs_rl", "sft_vs_preference", "dpo_vs_rl"]
        post_train_filtered = post_train_df[post_train_df["comparison_type"].isin(comparison_types)]
        post_train_sub = post_train_filtered[["comparison_type", "metric", "n", "wilcoxon_p", "cohens_d", "direction"]]
        # Save the filtered post-training results
        post_train_sub.to_csv(f"{args.output_prefix}_post-train.csv")
        
        small_vs_large_types = ["small_vs_large", "small_vs_large_base", "small_vs_large_instruct", "small_vs_large_instruct_rlhf", "small_vs_large_instruct_sft", "small_vs_large_instruct_dpo", "small_vs_large_preference"]
        small_vs_large_filtered = post_train_df[post_train_df["comparison_type"].isin(small_vs_large_types)]
        small_vs_large_sub = small_vs_large_filtered[["comparison_type", "metric", "n", "wilcoxon_p", "cohens_d", "direction"]]
        
        small_vs_large_sub.to_csv(f"{args.output_prefix}_small_vs_large.csv")
    if args.output_format in ['yaml', 'all']:
        json_to_yaml(results, f"{args.output_prefix}.yaml")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
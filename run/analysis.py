import os
import sys
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_curve

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis of data')
    parser.add_argument('--root', type=str, default='../collected',
                        help='root directory of collected data')
    parser.add_argument('--save', type=str, default='../tmp')
    parser.add_argument('--folder', type=str, default='codenet',
                        help='folder name of data')
    parser.add_argument('--split', type=str, default='val',
                        help='split of data')
    parser.add_argument('--model', type=str,
                        default='gpt-3.5-turbo',
                        help='model to use')
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    # Read in data
    file_name = "_".join([args.model, args.folder, args.split, 'diversity', 'results']) + '.jsonl'
    path = os.path.join(args.root, file_name)
    df = pd.read_json(path, lines=True, orient='records')

    df_results_stats = df[['coherence', 'semantic_count', 'n_outputs', 'n_coherent', 'distinct_1', 'distinct_2', 'distinct_3', 'corpus_self_bleu']]
    described = df_results_stats.describe()
    print('Model:', args.model)
    print(described)

    # plot boxplot of diversity metrics
    plt.figure(figsize=(10, 5))
    metrics = ['semantic_count', 'distinct_1', 'distinct_2', 'distinct_3', 'corpus_self_bleu']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        df.boxplot(column=metric)
        plt.title(metric)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save, f'{args.model}_diversity_metrics.png'))
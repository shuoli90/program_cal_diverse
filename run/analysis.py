import os
import sys
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    file_name = "_".join([args.model, str(args.temperature), args.folder, args.split]) + '.jsonl'
    path = os.path.join(args.root, file_name)
    df = pd.read_json(path, lines=True, orient='records')

    correctness_scores = np.stack(df['accuracy']).flatten()
    indicators = ['ecc_confidence', 'degree_confidence', 'nll']
    for indicator in indicators:
        confidences = np.stack(df[indicator])[:, 0, :].flatten()

        fig, ax = plt.subplots()
        threshold = 0.5
        y_true = correctness_scores >= threshold
        # plot roc curve
        fpr, tpr, _ = roc_curve(y_true, confidences)
        ax.plot(fpr, tpr, label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC curve with ' + indicator)
        ax.legend()
        ax.grid()
        dir = os.path.join(args.save, f'roc_curve_{args.model}_{args.folder}_{args.split}_{args.temperature}_{indicator}.png')
        plt.savefig(dir)
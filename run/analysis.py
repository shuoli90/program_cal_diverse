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
    file_name = "_".join([args.model, str(args.temperature), args.folder, args.split]) + '.jsonl'
    path = os.path.join(args.root, file_name)
    df = pd.read_json(path, lines=True, orient='records')
    # replace column 'nll' with "ll"
    df['ll'] = df['nll']

    semantic_nlls = []
    highest_semantic = []
    for index, row in df.iterrows():
        programs = row['programs']
        program_2_semantic_sting = row['program_2_semantic_string']
        nlls = row['nll']
        correctness_scores = row['accuracy']
        semantic_nll = defaultdict(list)
        for program, nll, correctness in zip(programs, nlls, correctness_scores):
            semantic = program_2_semantic_sting[program]
            likelihood = np.exp(-np.sum(nll))
            semantic_nll[semantic].append([likelihood, correctness])
        # find the highest semantic
        highest = 0.0
        highest_semantic_tmp = ''
        for semantic in semantic_nll:
            likelihoods = [x[0] for x in semantic_nll[semantic]] 
            likelihoods = np.sum(likelihoods)
            if likelihoods > highest:
                highest = likelihoods
                highest_semantic_tmp = semantic
                correctness_highest = semantic_nll[semantic][0][1]
        semantic_nlls.append(semantic_nll)
        highest_semantic.append([highest_semantic_tmp, correctness_highest, highest])
    
    syntax_nlls = []
    highest_syntax_semantic = []
    for index, row in df.iterrows():
        syntax_nll = {}
        programs = row['programs']
        program_2_semantic_sting = row['program_2_semantic_string']
        for program, nll, correctness in zip(programs, row['nll'], row['accuracy']):
            if program not in syntax_nlls:
                likelihood = np.exp(-np.sum(nll))
                syntax_nll[program] = [likelihood, correctness]
        highest = 0.0
        highest_syntax_tmp = ''
        for program in syntax_nll:
            likelihood = syntax_nll[program][0]
            if likelihood > highest:
                highest = likelihood
                highest_syntax_tmp = program
                correctness_highest = syntax_nll[program][1]
        highest_syntax_semantic.append([program_2_semantic_sting[highest_syntax_tmp], correctness_highest, highest])
        syntax_nlls.append([syntax_nll, correctness_highest])
    
    isSame = []
    semantic_corrects = []
    syntax_corrects = []
    for semantic, syntax_semantic in zip(highest_semantic, highest_syntax_semantic):
        isSame.append(semantic[0] == syntax_semantic[0])
        semantic_corrects.append(semantic[1])
        syntax_corrects.append(syntax_semantic[1])

    # correctness_scores = np.stack(df['accuracy']).flatten()
    # indicators = ['ecc_confidence', 'degree_confidence', 'll']
    # for indicator in indicators:
    #     confidences = np.stack(df[indicator])[:, 0, :].flatten()

    #     fig, ax = plt.subplots()
    #     threshold = 0.5
    #     y_true = correctness_scores >= threshold
    #     # plot roc curve
    #     fpr, tpr, _ = roc_curve(y_true, confidences)
    #     ax.plot(fpr, tpr, label='ROC curve')
    #     ax.plot([0, 1], [0, 1], 'k--', label='Random')
    #     ax.set_xlabel('False Positive Rate')
    #     ax.set_ylabel('True Positive Rate')
    #     ax.set_title('ROC curve with ' + indicator)
    #     ax.legend()
    #     ax.grid()
    #     dir = os.path.join(args.save, f'roc_curve_{args.model}_{args.folder}_{args.split}_{args.temperature}_{indicator}.png')
    #     plt.savefig(dir)
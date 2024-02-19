import numpy as np
from typing import List

def cluster(outputs, **kwargs):
    results = np.unique(outputs, axis=0, return_counts=True)
    return results

def get_sim_mat(outputs, **kwargs):
    sim_mat = np.zeros((len(outputs), len(outputs)))
    for i, output in enumerate(outputs):
        for j, output_ in enumerate(outputs):
            sim_mat[i, j] = output == output_
    return sim_mat

# This is the code from: https://github.com/zlin7/UQ-NLG

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans


def get_affinity_mat(sim_mat, mode='disagreement'):
    if mode == 'jaccard':
        return sim_mat >= 0.5
    sim_mat = (sim_mat + sim_mat.permute(1,0,2))/2
    if mode == 'disagreement':
        W = sim_mat.argmax(-1) != 0
    elif mode == 'agreement':
        W = sim_mat.argmax(-1) == 2
    elif mode == 'entailment':
        W = sim_mat[:, :, 2]
    elif mode == 'contradiction':
        W = 1-sim_mat[:, :, 0]
    else:
        raise NotImplementedError()
    W = W.cpu().numpy()
    W[np.arange(len(W)), np.arange(len(W))] = 1
    W = W.astype(np.float32)
    return W

def get_D_mat(W):
    # compute the degreee matrix from the weighted adjacency matrix
    D = np.diag(np.sum(W, axis=1))
    return D

def get_L_mat(W, symmetric=True):
    # compute the degreee matrix from the weighted adjacency matrix
    D = np.diag(np.sum(W, axis=1))
    # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
    if symmetric:
        L = np.linalg.inv(np.sqrt(D)) @ (D - W) @ np.linalg.inv(np.sqrt(D))
    else:
        raise NotImplementedError()
        # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
        L = np.linalg.inv(D) @ (D - W)
    return L.copy()

def get_eig(L, thres=None, eps=None):
    # This function assumes L is symmetric
    # compute the eigenvalues and eigenvectors of the laplacian matrix
    if eps is not None:
        L = (1-eps) * L + eps * np.eye(len(L))
    eigvals, eigvecs = np.linalg.eigh(L)

    if thres is not None:
        keep_mask = eigvals < thres
        eigvals, eigvecs = eigvals[keep_mask], eigvecs[:, keep_mask]
    return eigvals, eigvecs

def find_equidist(P, eps=1e-4):
    from scipy.linalg import eig
    P = P / P.sum(1)[:, None]
    P = (1-eps) * P + eps * np.eye(len(P))
    assert np.abs(P.sum(1)-1).max() < 1e-3
    w, vl, _ = eig(P, left=True)
    #assert np.max(np.abs(w.imag)) < 1e-5
    w = w.real
    idx = w.argsort()
    w = w[idx]
    vl = vl[:, idx]
    assert np.max(vl[:, -1].imag) < 1e-5
    return vl[:, -1].real / vl[:, -1].real.sum()

class SpetralClustering:
    def __init__(self,
                 eigv_threshold=None,) -> None:
        self.eigv_threshold = eigv_threshold
        self.rs = 0

    def get_laplacian(self, W):
        L = get_L_mat(W, symmetric=True)
        return L

    def get_eigvs(self, sim_mat):
        L = self.get_laplacian(sim_mat)
        return (1-get_eig(L)[0])

    def __call__(self, sim_mat, cluster=None):
        if cluster is None: cluster = self.cluster
        L = self.get_laplacian(sim_mat)
        if not cluster:
            return (1-get_eig(L)[0]).clip(0 if self.adjust else -1).sum()
        eigvals, eigvecs = get_eig(L, thres=self.eigv_threshold)
        k = eigvecs.shape[1]
        self.rs += 1
        kmeans = KMeans(n_clusters=k, random_state=self.rs, n_init='auto').fit(eigvecs)
        return kmeans.labels_

    def clustered_entropy(self, sim_mat):
        from scipy.stats import entropy
        labels = self(sim_mat, cluster=True)
        P = torch.softmax(sim_mat, dim=-1)[:, :, 2].cpu().numpy()
        pi = find_equidist(P)
        clustered_pi = pd.Series(pi).groupby(labels).sum().values
        return entropy(clustered_pi)

    def eig_entropy(self, W):
        L = get_L_mat(W, symmetric=True)
        eigs = get_eig(L, eps=1e-4)[0] / W.shape[0]
        return np.exp(- (eigs * np.nan_to_num(np.log(eigs))).sum())

    def proj(self, W):
        L = get_L_mat(W, symmetric=True)
        eigvals, eigvecs = get_eig(L, thres=self.eigv_threshold)
        return eigvecs

    def kmeans(self, eigvecs):
        k = eigvecs.shape[1]
        self.rs += 1
        kmeans = KMeans(n_clusters=k, random_state=self.rs, n_init='auto').fit(eigvecs)
        return kmeans.labels_

def umap_visualization(eigvecs, labels):
    # perform umap visualization on the eigenvectors
    import umap
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(eigvecs)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels)
    return embedding

import numpy as np

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

def proj(sim_mat, eigv_threshold=0.1):
    L = get_L_mat(sim_mat, symmetric=True)
    eigvals, eigvecs = get_eig(L, thres=eigv_threshold)
    return eigvecs

class Eccentricity():
    def __init__(self, eig_threshold=0.1):
        self.eig_threshold = eig_threshold
    
    def __call__(self, sim_mats, **kwargs):
        '''
        Input:
            batch_prompts: a batch of prompts[prompt_1, ..., prompt_B]
            batch_responses: a batch of sequences [[r_1^1, ..., r_{n_1}^1], ..., [r_1^1, ..., r_{n_B}^B]]
        Output:
            batch_U: a batch of uncertainties [U^1, ..., U^B]
            batch_Cs: a batch of confidence sequences [[C_1^1, ..., C_{n_1}^1], ..., [C_1^B, ..., C_{n_B}^B]]
        '''
        batch_projected = [proj(sim_mat, eigv_threshold=self.eig_threshold) for sim_mat in sim_mats]
        batch_Cs = [-np.linalg.norm(projected-projected.mean(0)[None, :],2,axis=1) for projected in batch_projected]
        batch_U = [np.linalg.norm(projected-projected.mean(0)[None, :], 2) for projected in batch_projected]
        return batch_U, batch_Cs
    
class Degree():
    def __init__(self, eigv_threshold=0.1):
        self.eigv_threshold = eigv_threshold
    
    def __call__(self, sim_mats, **kwargs):
        '''
        Input:
            batch_prompts: a batch of prompts [p^1, ..., p^B]
            batch_responses: a batch of sequences [[r_1^1, ..., r_{n_1}^1], ..., [r_1^1, ..., r_{n_B}^B]]
        Output:
            batch_U: a batch of uncertainties [U^1, ..., U^B]
            batch_Cs: a batch of confidence sequences [[C_1^1, ..., C_{n_1}^1], ..., [C_1^B, ..., C_{n_B}^B]]
        '''
        batch_Cs = [np.mean(W, axis=1) for W in sim_mats]
        batch_U = [1/W.shape[0]-np.sum(W)/W.shape[0]**2 for W in sim_mats]
        return batch_U, batch_Cs

class SpectralEigv():
    def __init__(self, eigv_threshold=0.1, **kwargs):
        self.eigv_threshold = eigv_threshold

    def __call__(self, sim_mats, **kwargs):
        return [1-get_eig(sim_mat, thres=self.eigv_threshold)[0].sum() for sim_mat in sim_mats] 
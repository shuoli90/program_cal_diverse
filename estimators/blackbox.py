import re
import numpy as np
from utils import clustering as pc

def spectral_projected(batch_sim_mat, threshold=0.5):
    # sim_mats: list of similarity matrices using semantic similarity model or jacard similarity
    clusterer = pc.SpetralClustering(eigv_threshold=threshold)
    return [clusterer.proj(sim_mat) for sim_mat in batch_sim_mat]

class VerbalizedConfidence():
    def __init__(self, pipe=None):
        self.pipe = pipe
        self.tokenizer = self.pipe.tokenizer
        self.description1 = "Given the task description:{}"
        self.description2 = "\nProvide a numeric confidence that indicates your certainty about this generated program . \
                            For instance, if your confidence level is 80%, it means you are 80% certain that this answer is correct and there is a 20% chance that it is incorrect. \
                            Use the following format to provide your confidence: Confidence: [Your confidence, a numerical number in the range of 0-100]%."

    def extract_confidence(self, s):
        pattern = r'Confidence:\s*(\d+)%'
        match = re.findall(pattern, s)
        if match:
            conf = int(match[0])/100
            return conf
        else:
            breakpoint()
            raise ValueError("No formatted verbalized confidence available!")

    def compute_scores(self, batch_prompt, programs, **kwargs):
        '''
        Input:
            batch_prompt: a batch of prompt [p^1, ..., p^B]
            batch_responses: a batch of sequences [[r_1^1, ..., r_{n_1}^1], ..., [r_1^1, ..., r_{n_B}^B]]
        Output:
            batch_Cs: a batch of confidence sequences [[C_1^1, ..., C_{n_1}^1], ..., [C_1^B, ..., C_{n_B}^B]]
        '''
        Cs = []
        for prompt, program in zip(batch_prompt, programs):
            combo_text = self.description1+prompt+self.description2+program
            cur_length = len(self.tokenizer(combo_text)['input_ids'])
            verbal_conf = self.pipe.generate(combo_text, max_length=cur_length+10, return_full_text=False)[0]['generated_text']
            Cs.append(self.extract_confidence(verbal_conf))
        return Cs

class Eccentricity():
    def __init__(self, device='cuda'):
    
    def compute_scores(self, batch_prompts, batch_responses, **kwargs):
        '''
        Input:
            batch_prompts: a batch of prompts[prompt_1, ..., prompt_B]
            batch_responses: a batch of sequences [[r_1^1, ..., r_{n_1}^1], ..., [r_1^1, ..., r_{n_B}^B]]
        Output:
            batch_U: a batch of uncertainties [U^1, ..., U^B]
            batch_Cs: a batch of confidence sequences [[C_1^1, ..., C_{n_1}^1], ..., [C_1^B, ..., C_{n_B}^B]]
        '''
        batch_sim_mats = [pc.get_sim_mat(responses) for responses in batch_responses]
        batch_projected = spectral_projected(batch_sim_mats, threshold=0.1)
        batch_Cs = [-np.linalg.norm(projected-projected.mean(0)[None, :],2,axis=1) for projected in batch_projected]
        batch_U = [np.linalg.norm(projected-projected.mean(0)[None, :],2).clip(-1, 1) for projected in batch_projected]
        return batch_U, batch_Cs
    
class Degree():
    def __init__(self, device='cuda'):
    
    def compute_scores(self, batch_prompts, batch_responses, **kwargs):
        '''
        Input:
            batch_prompts: a batch of prompts [p^1, ..., p^B]
            batch_responses: a batch of sequences [[r_1^1, ..., r_{n_1}^1], ..., [r_1^1, ..., r_{n_B}^B]]
        Output:
            batch_U: a batch of uncertainties [U^1, ..., U^B]
            batch_Cs: a batch of confidence sequences [[C_1^1, ..., C_{n_1}^1], ..., [C_1^B, ..., C_{n_B}^B]]
        '''
        batch_W = [pc.get_sim_mat(responses) for responses in batch_responses]
        batch_Cs = [np.mean(W, axis=1) for W in batch_W]
        batch_U = [1/W.shape[0]-np.sum(W)/W.shape[0]**2 for W in batch_W]
        return batch_U, batch_Cs

class SpectralEigv():
    def __init__(self, device='cuda'):

    def compute_scores(self, batch_prompts, batch_responses, **kwargs):
        sim_mats = [pc.get_sim_mat(responses) for responses in batch_responses]
        clusterer = pc.SpetralClustering(eigv_threshold=None,
                                         temperature=self.temperature)
        return [clusterer.get_eigvs(_).clip(0 if self.adjust else -1).sum() for _ in sim_mats]
import torch
from collections import defaultdict

CONTRADICT, NEUTRAL, AGREE = 0, 1, 2
llh_shift = torch.tensor(5.0)

@torch.no_grad()
def get_neg_loglikelihoods(model, tokenizer, messages):
    # https://github.com/lorenzkuhn/semantic_uncertainty
    device = model.device
    result = []
    for sample in messages:
        result_dict = {}
        prompt = sample['prompt']
        generations = sample['generations'].to(device)
        id_ = sample['id']

        average_neg_log_likelihoods = torch.zeros((generations.shape[0],))
        average_unconditioned_neg_log_likelihoods = torch.zeros((generations.shape[0],))
        neg_log_likelihoods = torch.zeros((generations.shape[0],))
        neg_unconditioned_log_likelihoods = torch.zeros((generations.shape[0],))
        pointwise_mutual_information = torch.zeros((generations.shape[0],))
        sequence_embeddings = []
        for generation_index in range(generations.shape[0]):
            prompt = prompt[prompt != tokenizer.pad_token_id]
            generation = generations[generation_index][generations[generation_index] != tokenizer.pad_token_id]
            generation_only = generation.clone()
            unconditioned_model_output = model(torch.reshape(generation_only, (1, -1)),
                                                labels=generation_only,
                                                output_hidden_states=True)

            # concatenate the prompt and the generation tokens
            generation = torch.cat((prompt, generation[1:]))
            target_ids = generation.clone()
            target_ids[:len(prompt)] = -100
            model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=True)  
            hidden_states = model_output['hidden_states']
            average_neg_log_likelihood = model_output['loss']
            average_unconditioned_neg_log_likelihood = unconditioned_model_output['loss']
            average_neg_log_likelihoods[generation_index] = average_neg_log_likelihood
            average_unconditioned_neg_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood
            # neg_log_likelihoods[generation_index] = average_neg_log_likelihood * (len(generation) - len(prompt))
            neg_log_likelihoods[generation_index] = average_neg_log_likelihood * len(generation_only)
            neg_unconditioned_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood * len(generation_only)
            pointwise_mutual_information[generation_index] = -neg_log_likelihoods[
                generation_index] + neg_unconditioned_log_likelihoods[generation_index]

            average_of_last_layer_token_embeddings = torch.mean(hidden_states[-1], dim=1)
            sequence_embeddings.append(average_of_last_layer_token_embeddings)

        sequence_embeddings = torch.stack(sequence_embeddings)
        result_dict['prompt'] = prompt
        result_dict['generations'] = generations
        result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods
        result_dict['neg_log_likelihoods'] = neg_log_likelihoods
        result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods
        result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods
        result_dict['pointwise_mutual_information'] = pointwise_mutual_information
        result_dict['id'] = id_
        result.append(result_dict)

    return result

# whitebox methods
def _logmeanexp(x, dim, ignore_negative_inf=False):
    if ignore_negative_inf:
        cnt = (x > -torch.inf).sum(dim)
    else:
        cnt = torch.tensor(x.shape[dim])
    return torch.logsumexp(x, dim=dim) - torch.log(cnt)
    
class WhiteBox():

    def __init__(self):
        return NotImplementedError

    def compute_scores(self):
        return NotImplementedError


class SemanticEntropy(WhiteBox):

    def __init__(self, pipe, device='cuda'):
        self.device = device if device is not None else torch.device('cpu')
        self.mem = defaultdict(dict)
        self.model = pipe.model
        self.tokenizer = pipe.tokenizer
    
    def compute_scores(self, prompt, programs, outputs, **kwargs):
        '''
        Input:
            batch_prompt: one prompt
            programs: programs [prog^1, ..., prog^B]
            outputs: outputs [out^1, ..., out^B]
        Output:
            batch_entropy: one entropy value
        '''
        
        # group programs by outputs
        outputs = torch.tensor(outputs)
        programs = torch.tensor(programs)
        unique_outputs, _ = torch.unique(outputs, return_counts=True, dim=0)

        # compute the negative log likelihoods for each program
        messages = {
            'prompt': torch.tensor(self.tokenizer.encode(prompt)).to(self.device),
            'generations': torch.tensor(self.tokenizer(programs, padding='longest')['input_ids']).to(self.device),
            'id': 0}
        neg_log_likelihoods = get_neg_loglikelihoods(self.model, self.tokenizer, messages)['neg_log_likelihoods']
        nll_groups = [neg_log_likelihoods[outputs == u] for u in unique_outputs]
        aggregated_log_likelihoods = []
        for nll_group in nll_groups:
            aggregated_log_likelihoods.append(torch.logsumexp(nll_group, 0))
        aggregated_log_likelihoods = torch.tensor(aggregated_log_likelihoods)
        entropy = - torch.sum(aggregated_log_likelihoods, dim=0) / torch.tensor(aggregated_log_likelihoods.shape[0])
        return entropy            

class GenerationProbability(WhiteBox):
    def __init__(self, pipe):
        self.model = pipe.model
        self.tokenizer = pipe.tokenizer
        self.device = self.model.device
    
    def compute_scores(self, prompt, programs, **kwargs):
        '''
        Input:
            prompt: one prompt p
            programs: generated programs [prog^1, ..., prog^B]
        Output:
            batch_GPs: neg_loglikelihoods [GP^1, ..., GP^B]
        '''
        messages = {
            'prompt': torch.tensor(self.tokenizer.encode(prompt)).to(self.device),
            'generations': torch.tensor(self.tokenizer(programs, padding='longest')['input_ids']).to(self.device),
            'id': 0}
        batch_GPs = get_neg_loglikelihoods(self.model, self.tokenizer, messages)
        return batch_GPs
    

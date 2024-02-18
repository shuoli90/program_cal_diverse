import re

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
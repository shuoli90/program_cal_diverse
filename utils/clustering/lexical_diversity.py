import tokenize
from io import BytesIO
import tokenize
from io import BytesIO
from transformers import GPT2Tokenizer
import joblib
from joblib import Parallel, delayed
import contextlib
from nltk import ngrams 
from typing import List, Callable
from sacrebleu.metrics import BLEU
import re
from transformers import AutoTokenizer
from dataclasses import dataclass
from tqdm import tqdm
from parso.python.tokenize import tokenize as parso_tokenize

bleu = BLEU(tokenize=None)

newline_pattern = re.compile(r'\n')

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
_codebert_tokenzier = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def codebert_tokenizer(code_str, remove_comments = False):
    if remove_comments:
        code_str = re.sub(r"#.*", "", code_str)
    encoded_input = _codebert_tokenzier(code_str)
    tokens = _codebert_tokenzier.convert_ids_to_tokens(encoded_input['input_ids'])
    return tokens
    

def get_relevant_tokens_lexer(code_str, remove_comments = False):
    # Convert the string to a bytes-like object
    bytes_io = BytesIO(code_str.encode('utf-8'))
    
    # Use the tokenize module to tokenize the code
    tokens = tokenize.tokenize(bytes_io.readline)
    
    # Define the irrelevant token types
    relevant_tokens = _get_relevant_tokens(tokens, remove_comments)
    
    return relevant_tokens

def get_relevant_tokens_parso(code_str, remove_comments = False):
    """
    The python native tokenizer does not handle errors well, so we use the parso tokenizer instead which supports error handling
    """
    tokens = parso_tokenize(code_str, version_info=(3, 12))
    relevant_tokens = _get_relevant_tokens(tokens, remove_comments)
    return relevant_tokens

def _get_relevant_tokens(tokens, remove_comments = False):
    irrelevant_types = {
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        # tokenize.NEWLINE,
        tokenize.INDENT, # indents are usually 'obvious' ie an if statement will have an indent, a for statement will have an indent, etc; the dedent is more interesting
        tokenize.NL,
        # tokenize.COMMENT,
    }
    if remove_comments:
        irrelevant_types.add(tokenize.COMMENT)
    
    # Extract the relevant tokens, ignore irrelevant ones, and exclude the token type for string representation
    # relevant_tokens = [token.string for token in tokens if token.type not in irrelevant_types or token.type == tokenize.DEDENT]
    relevant_tokens = [] 
    for token in tokens:
        if token.type not in irrelevant_types:
            if token.type == tokenize.STRING or token.type == tokenize.COMMENT:
                relevant_tokens.extend(token.string.split(" "))
            elif token.type == tokenize.DEDENT:
                relevant_tokens.append("DEDENT")
            else: 
                relevant_tokens.append(token.string)
    return relevant_tokens


    



def get_relevant_tokens_tokenizer(code_str, tokenizer = codebert_tokenizer, remove_comments = False):
    # get the tokens from the tokenizer
    if remove_comments:
        code_str = re.sub(r"#.*", "", code_str)
    tokens = tokenizer.tokenize(code_str)
    return tokens

# # Example usage:
# code = """class AdventureGame:
#     ## the init
#     def __init__(self):
#         self.rooms = {'Hall': {'South': 'Kitchen', 'item': 'Key'},
#                       'Kitchen': {'North': 'Hall', 'item': 'Monster'}}
#         self.inventory = []
#         self.current_room = 'Hall'
#     # the move
#     def move(self, direction):
#         if direction in self.rooms[self.current_room]:
#             self.current_room = self.rooms[self.current_room][direction]
#             print(f"You are in the {self.current_room}.")
#         else:
#             print("You can't go that way.")
#     # the get item
#     def get_item(self):
#         item = self.rooms[self.current_room].get('item', '')
#         if item:
#             print(f"You have found a {item}!")
#             self.inventory.append(item)
#             del self.rooms[self.current_room]['item']  # Remove the item from the room
#         else:
#             print("There's nothing here.")"""

# relevant_tokens = get_relevant_tokens_lexer(code)
# print(relevant_tokens)

# relevant_tokens = get_relevant_tokens_tokenizer(code)
# print(relevant_tokens)




def tokenize_for_self_bleu(code_str, ftokenizer: Callable[[str], List[str]]) -> List[str]:
    tokens = ftokenizer(code_str)
    tokens = [newline_pattern.sub("NEWLINE", token) for token in tokens]
    return " ".join(tokens)

# \mathrm{DP}(Y)=\frac{1}{|Y|(|Y|-1)} \sum_{y \in Y} \sum_{y^{\prime} \in Y, y^{\prime} \neq y} 1-\Delta\left(y, y^{\prime}\right)

def self_bleu_metric(src: str, tgt: str, ftokenizer: Callable[[str], List[str]]) -> float:
    src_tokens = tokenize_for_self_bleu(src, ftokenizer)
    tgt_tokens = tokenize_for_self_bleu(tgt, ftokenizer)
    return 100 - bleu.sentence_score(src_tokens, [tgt_tokens]).score

# https://aclanthology.org/P19-1177.pdf
def iterative_corpus_self_bleu(sentences: List[str], ftokenizer: Callable[[str], List[str]], normalize: bool = True) -> float:
    total_self_bleu = 0
    n = len(sentences)
    for i in range(n):
        for j in range(i+1, n):
            total_self_bleu += self_bleu_metric(sentences[i], sentences[j], ftokenizer)
    return (total_self_bleu / (n * (n-1) / 2)) if normalize else total_self_bleu


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
        
        
def parallel_corpus_self_bleu(sentences: List[str], ftokenizer: Callable[[str], List[str]], n_jobs: int = -1, normalize: bool = True) -> float:
    total_self_bleu = 0
    n = len(sentences)
    total_pairs = n * (n-1) / 2
    # with tqdm_joblib(tqdm(desc="Generating test cases", total=len(problem_ids))) as progress_bar:
    with tqdm_joblib(tqdm(desc="Calculating self-bleu", total=total_pairs)) as progress_bar:
        total_self_bleu = sum(Parallel(n_jobs=n_jobs)(delayed(self_bleu_metric)(sentences[i], sentences[j], ftokenizer) for i in range(n) for j in range(i+1, n)))
    return (total_self_bleu / total_pairs) if normalize else total_self_bleu
    

# https://aclanthology.org/N16-1014.pdf
def distinct_n(corpus: List[str], n: int, ftokenizer: Callable[[str], List[str]], remove_comments: bool = False) -> float:
    ngrams_list = [list(ngrams(ftokenizer(seq, remove_comments), n)) for seq in corpus]
    ngrams_set = set()
    for ngrams_seq in ngrams_list:
        ngrams_set.update(ngrams_seq)
    return len(ngrams_set) / sum(map(len, ngrams_list)) if sum(map(len, ngrams_list)) > 0 else 0.0






    

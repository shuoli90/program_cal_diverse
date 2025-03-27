import os
import dotenv
from openai import OpenAI
from llm_code_eval import evaluate
import random
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def compute_ice(problem_desc, gen1, gen2, model="gpt-4o-mini"):
    # randomly choose one of the two generations as the reference
    gen_pair = [gen1, gen2]
    ref_idx = random.sample([0, 1], 1)[0]
    output_prog = gen_pair[1 - ref_idx]
    ref_prog = gen_pair[ref_idx]
    raw_score = evaluate(
        problem_desc,
        output_prog,
        reference=ref_prog,
        task="code-gen",
        aspect="functional correctness",
        model="gpt-4o-mini",
        cot=False
    )
    return 1 - (raw_score / 4), output_prog, ref_prog

def get_embedding(model, tokenizer, gen, batch_size=8, max_tokens=1024):
    with torch.no_grad():
        inputs = tokenizer(gen, return_tensors="pt", truncation=True, max_length=max_tokens).to("cuda")
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states[-1]
        embedding = hidden_states.mean(dim=1)
    return embedding.squeeze(0)

def get_embedding_batch(model, tokenizer, gens, batch_size=8, max_tokens=1024):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(gens), batch_size):
            batch = gens[i:i+batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to("cuda")
            outputs = model(**encoded)
            hidden_states = outputs.hidden_states[-1]
            embeddings = hidden_states.mean(dim=1)
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

def compute_embedding_similarity(gen1_emb, gen2_emb):
    sim_score = cosine_similarity(gen1_emb.unsqueeze(0), gen2_emb.unsqueeze(0)).item()
    return sim_score

if __name__ == '__main__':
    test_problem_desc = "Given a list of integers, return the sum of all the integers."
    test_gen1 = "def f(lst):\n    return sum(lst)"
    test_gen2 = "def f(lst):\n    return sum(lst) + 1"
    test_gen3 = "def f(lst):\n    return 1"
    test_gen4 = "def f(lst):\n    total = 0\n    for num in lst:\n        total += num\n    return total"
    test_gen5 = "def f(lst):\n    result = 1\n    for num in lst:\n        result *= num\n    return result"
    test_gen6 = "def f(lst):\n    return lst[::-1]"
    
    # === Test get_embedding_batch ===
    model_name = "meta-llama/CodeLlama-7b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to("cuda")
    
    test_gens = [test_gen1, test_gen2, test_gen3, test_gen4, test_gen5, test_gen6]
    test_embeddings = get_embedding_batch(model, tokenizer, test_gens)
    print(test_embeddings.shape)  # torch.Size([len(test_gens), hidden_size])
    breakpoint()
    
    # === Test compute_ice ===
    print("gen1 vs gen2:", compute_ice(test_problem_desc, test_gen1, test_gen2))
    print("gen2 vs gen1:", compute_ice(test_problem_desc, test_gen2, test_gen1))
    print()
    print("gen2 vs gen3:", compute_ice(test_problem_desc, test_gen2, test_gen3))
    print("gen3 vs gen2:", compute_ice(test_problem_desc, test_gen3, test_gen2))
    print()
    print("gen2 vs gen2:", compute_ice(test_problem_desc, test_gen2, test_gen2))
    print("gen3 vs gen3:", compute_ice(test_problem_desc, test_gen3, test_gen3))
    
    # === Test compute_embedding ===    
    print("gen1 vs gen2:", compute_embedding_similarity(model, tokenizer, test_gen1, test_gen2))
    print("gen2 vs gen1:", compute_embedding_similarity(model, tokenizer, test_gen2, test_gen1))
    print()
    print("gen1 vs gen3:", compute_embedding_similarity(model, tokenizer, test_gen1, test_gen3))
    print("gen3 vs gen1:", compute_embedding_similarity(model, tokenizer, test_gen3, test_gen1))
    print()
    print("gen1 vs gen4:", compute_embedding_similarity(model, tokenizer, test_gen1, test_gen4))
    print("gen4 vs gen1:", compute_embedding_similarity(model, tokenizer, test_gen1, test_gen1))
    print()
    print("gen1 vs gen5:", compute_embedding_similarity(model, tokenizer, test_gen1, test_gen5))
    print("gen5 vs gen1:", compute_embedding_similarity(model, tokenizer, test_gen5, test_gen1))
    print()
    print("gen1 vs gen6:", compute_embedding_similarity(model, tokenizer, test_gen1, test_gen6))
    print("gen6 vs gen1:", compute_embedding_similarity(model, tokenizer, test_gen6, test_gen1))
    print()
    print("gen1 vs gen1:", compute_embedding_similarity(model, tokenizer, test_gen1, test_gen1))
    print("gen5 vs gen5:", compute_embedding_similarity(model, tokenizer, test_gen5, test_gen5))
    print("gen6 vs gen6:", compute_embedding_similarity(model, tokenizer, test_gen6, test_gen6))
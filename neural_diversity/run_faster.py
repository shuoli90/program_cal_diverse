import os
import random
import pandas as pd
import math
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM

from neural_metrics import compute_ice, compute_embedding_similarity, get_embedding, get_embedding_batch

DATADIR = [
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetTemperatureSweep_2025-03-11_12-22-31", 
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetTemperatureSweep_2025-03-10_16-48-50", 
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetTemperatureSweep_2025-03-05_14-06-22"
]
# 6 models
MODELS_TO_COMPARE = [
    # "meta-llama-Llama-3.1-8B",
    # "meta-llama-Llama-3.1-8B-Instruct",
    "meta-llama-Llama-3.1-70B", 
    "meta-llama-Llama-3.1-70B-Instruct", 
    "allenai-Llama-3.1-Tulu-3-8B-SFT", 
    "allenai-Llama-3.1-Tulu-3-70B-SFT"
]
# 8 temps
TEMP = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]

NUM_SAMPLES = 15
SEED = 2025
METRIC = "embedding"
MAX_WORKERS = 100

if METRIC == "ice":
    MODEL = "gpt-4o-mini"
elif METRIC == "embedding":
    model_name = "meta-llama/CodeLlama-7b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    MODEL = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to("cuda")
    MODEL.eval()
    BATCH_SIZE = 4
    MAX_TOKENS = 1024
    MAX_WORKERS = 1
else:
    raise ValueError(f"[Error] wrong metric: {METRIC}")

def mapping_model_temp_to_path():
    mappings = {}
    for model in MODELS_TO_COMPARE:
        for temp in TEMP:
            prefix = f"{model}_temp_{temp}"
            matched_dirs = []
            for dir in DATADIR:
                subdirs = [
                    os.path.join(dir, d) 
                    for d in os.listdir(dir)
                    if os.path.isdir(os.path.join(dir, d)) and d.startswith(prefix)
                ]
                matched_dirs.extend(subdirs)
            if len(matched_dirs) == 0:
                raise ValueError(f"[Error] {model} temp_{temp}")
            mappings[(model, temp)] = matched_dirs
    return mappings

def mapping_problem_id_to_desc():
    filepath = "/home/bzhang16/program_cal_diverse/data/open_ended_extended/dataset_extended_final.jsonl"
    mappings = {}
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line)
            mappings[data["problem_id"]] = data["description_string"]
    return mappings

def compute_div(sampled_gen_paths, metric="ice", problem_desc=""):
    gens = []
    for gen_path in sampled_gen_paths:
        with open(os.path.join(gen_path, "gen.txt"), "r") as f:
            gen = f.read().strip()
            gens.append(gen)
    if len(gens) == 0:
        return None, None
    gen_pairs = [(gens[i], gens[j]) for i in range(len(gens)) for j in range(i+1, len(gens))]
    if len(gen_pairs) != math.comb(len(gens), 2):
        raise ValueError(f"[Error] wrong number of pairs")
    if metric == "ice":
        scores = []
        scores_detail = []
        for gen1, gen2 in gen_pairs:
            score, output_prog, ref_prog = compute_ice(problem_desc, gen1, gen2, model=MODEL)
            scores.append(score)
            scores_detail.append({
                "output_prog": output_prog,
                "ref_prog": ref_prog,
                "score": score
            })
    elif metric == "embedding":
        # gens = [
        #     get_embedding(
        #         MODEL, 
        #         tokenizer, 
        #         gen, 
        #         batch_size=BATCH_SIZE, 
        #         max_tokens=MAX_TOKENS
        #     ) 
        #     for gen in gens
        # ]
        gen_embs = get_embedding_batch(
            MODEL, 
            tokenizer, 
            gens, 
            batch_size=BATCH_SIZE, 
            max_tokens=MAX_TOKENS
        )
        gen_embs_pairs = [(gen_embs[i], gen_embs[j]) for i in range(len(gen_embs)) for j in range(i+1, len(gen_embs))]
        if len(gen_embs_pairs) != math.comb(len(gens), 2):
            raise ValueError(f"[Error] wrong number of pairs")
        scores = []
        scores_detail = []
        for (gen1, gen2), (gen1_emb, gen2_emb) in zip(gen_pairs, gen_embs_pairs):
            score = compute_embedding_similarity(gen1_emb, gen2_emb)
            scores.append(score)
            scores_detail.append({
                "gen1": gen1,
                "gen2": gen2,
                "score": score
            })
        
    if len(scores) == 0:
        return None, None
    div = sum(scores) / len(scores)
    return div, scores_detail

def compute_worker(problem_path, model, temp, temp_dir, problem_desc_lookup, tracker_str):
    problem_id = os.path.basename(problem_path).split("_")[1]
    print(f"{tracker_str} {problem_id}")
    
    gen_paths = [
        os.path.join(problem_path, gen_dir)
        for gen_dir in os.listdir(problem_path)
        if gen_dir.startswith("generation_")
    ]
    random.seed(SEED)
    sampled_gen_paths = random.sample(gen_paths, min(NUM_SAMPLES, len(gen_paths)))
    problem_desc = problem_desc_lookup[problem_id]
    div, scores_detail = compute_div(sampled_gen_paths, metric=METRIC, problem_desc=problem_desc)
    if div is None:
        return None, None
    div_detail = [
        {
            "model": model,
            "temperature": temp,
            "problem_id": problem_id,
            **detail
        }
        for detail in scores_detail
    ]
    with open(f"{temp_dir}/{problem_id}.jsonl", "w", encoding="utf-8") as f:
        for entry in div_detail:
            f.write(json.dumps(entry) + "\n")
    return div, div_detail

if __name__ == '__main__':
    path_lookup = mapping_model_temp_to_path()
    problem_desc_lookup = mapping_problem_id_to_desc()
    
    output_dir = f"{os.getcwd()}/neural_diversity/{METRIC}_results"
    if not os.path.exists(output_dir):
        raise ValueError(f"[Error] {output_dir} does not exist")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    all_scores = []
    all_detailed_scores = []
    for i, ((model, temp), dirpath) in enumerate(path_lookup.items(), 1):
        model_dir = f"{output_dir}/{model}"
        os.makedirs(model_dir, exist_ok=True)
        temp_dir = f"{model_dir}/temp_{temp}"
        os.makedirs(temp_dir, exist_ok=True)
        tracker_str = f"[{i}/{len(MODELS_TO_COMPARE) * len(TEMP)}] {timestamp}"
        print(f"{tracker_str} {model} (temp={temp}) ...")
        dirpath = dirpath[-1]
        
        # 108 problem_p* -> 32 generation_<0-31> -> gen.txt
        problem_paths = [
            os.path.join(dirpath, problem_dir)
            for problem_dir in os.listdir(dirpath)
            if problem_dir.startswith("problem_p")
        ]
        # if len(problem_paths) != 108:
        #     raise ValueError(f"[Error] {dirpath}")
        divs = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(compute_worker, problem_path, model, temp, temp_dir, problem_desc_lookup, tracker_str)
                for problem_path in problem_paths
            ]
            for future in futures:
                div, div_detail = future.result()
                if div is not None:
                    divs.append(div)
                    all_detailed_scores.extend(div_detail)
        if len(divs) > 0:
            avg_div = sum(divs) / len(divs)
            score = {
                "model": model,
                "temperature": temp,
                METRIC: avg_div
            }
            all_scores.append(score)
    
    df = pd.DataFrame(all_scores)
    df.to_csv(f"{output_dir}/ice_scores_n{NUM_SAMPLES}_{timestamp}.csv", index=False)

    with open(f"{output_dir}/ice_scores_detailed_n{NUM_SAMPLES}_{timestamp}.jsonl", "w", encoding="utf-8") as f:
        for entry in all_detailed_scores:
            f.write(json.dumps(entry) + "\n")
            
    print("✅ Done.")
import os 
import glob 
import json 
from typing import List, Dict
import warnings

directories = paths = ["/data0/shypula/prog_diversity/all_experiments/Open_Ended_Reevaluation_EAD_Open_and_Commercial_2024-08-02_16-02-07/", 
                        "/data1/shypula/prog_diversity/all_experiments/OpenEndedCommercialV3_2024-08-09_02-28-20/"]


key_columns = ["all_semantic_count_wcoh_nonempty_woutput", "all_average_cosine_distance_programs", "all_average_cosine_distance_raw", 
               "all_average_cosine_distance_programs_zero_null", 
               "all_ead_4_bootstrap", "all_stripped_subtrees_4_bootstrap", 
               "generations", "extracted_programs", "experiment_name", "problem_id"]


def read_tsv_file(file_path):
    """
    Read a TSV file and return all key-value pairs as a dictionary.
    """
    result_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split('\t')
            result_dict[key] = value
    return result_dict

# def make_semantic_strings(output_records: List[Dict]):
#     program_2_semantic_string = {}
#     semantic_strings_2_programs = defaultdict(list)
#     for output_record in output_records:
#         semantic_string = ""
#         for testcase_id, testcase_input in output_record["testcase_inputs"].items():
#             testcase_output = output_record["testcase_outputs"][testcase_id]
#             semantic_string += f"testcase_input: {testcase_input}, output: {testcase_output}\n"
#         program_2_semantic_string[output_record["code"]] = semantic_string
#         semantic_strings_2_programs[semantic_string].append(output_record["code"])
#     return program_2_semantic_string, semantic_strings_2_programs

def output_record_to_semantic_string(output_record: Dict):
    semantic_string = ""
    all_keys = list(output_record["testcase_outputs"].keys())
    all_keys.sort()
    for testcase_id in all_keys:
        testcase_input = output_record["testcase_inputs"][testcase_id]
        testcase_output = output_record["testcase_outputs"][testcase_id]
        semantic_string += f"testcase_input: {testcase_input}, output: {testcase_output}\n"
    return semantic_string


experiment_dirs = []
for directory in directories:
    ds = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    ds = [os.path.join(directory, d) for d in ds]
    experiment_dirs.extend(ds)

all_results = []
all_pairs = []

from tqdm import tqdm
import re 
import json
    
# pbar = tqdm(total=len(experiment_dirs))
# if not os.path.exists("/data0/shypula/prog_diversity/codebertscore_vs_semantic_analysis/all_pairs.json"):

number_experiment_x_problem = 0
for i, directory in enumerate(experiment_dirs):
    for problem_dir in glob.glob(os.path.join(directory, "p*")):
        number_experiment_x_problem += 1
print(f"Number of experiments x problems: {number_experiment_x_problem}")
pbar = tqdm(total=number_experiment_x_problem)
for i, directory in enumerate(experiment_dirs):
    # results_tsvs = glob.glob(os.path.join(directory, "p*", "result.tsv"))
    # if i < 5:
    #     print(f"Found {len(results_tsvs)} result.tsv files in {directory}")
    
    # for tsv_path in results_tsvs:
    #     result = read_tsv_file(tsv_path)
    #     all_results.append(result)
    experiment_name = os.path.basename(directory)
    problem_dirs = glob.glob(os.path.join(directory, "p*"))
    for problem_dir in problem_dirs:
        problem_id = os.path.basename(problem_dir)
        generation_dirs = glob.glob(os.path.join(problem_dir, "generation_*"))
        # assert len(generation_dirs) == 100, f"Expected 100 generation directories, but found {len(generation_dirs)} in {problem_dir}"
        # if len(generation_dirs) != 100:
        #     warnings.warn(f"Expected 100 generation directories, but found {len(generation_dirs)} in {problem_dir}")
        generation_tuples = []
        for gen_dir in generation_dirs:
            # generation_38_coh_float 
            gen_id = re.search(r"generation_(\d+)_", gen_dir).group(1)
            if int(gen_id) < 0 or int(gen_id) >= 100:
                continue
            extracted_program_path = os.path.join(gen_dir, "prog.txt")
            # assert os.path.exists(extracted_program_path), f"Expected extracted program at {extracted_program_path}"
            if not os.path.exists(extracted_program_path):
                # warnings.warn(f"Expected extracted program at {extracted_program_path}")
                continue
            with open(extracted_program_path, 'r') as f:
                extracted_program = f.read().strip()
            if extracted_program == "":
                continue
            generation_result_path = os.path.join(gen_dir, "output_record.json")
            with open(generation_result_path, 'r') as f:
                generation_result = json.load(f)
            semantic_string = output_record_to_semantic_string(generation_result)
            generation_tuples.append((gen_id, extracted_program, semantic_string))
        for (i, prog_i, sem_str_i) in generation_tuples:
            for (j, prog_j, sem_str_j) in generation_tuples:
                if i != j: 
                    all_pairs.append({
                        "experiment_name": experiment_name,
                        "problem_id": problem_id,
                        "gen_i": i,
                        "gen_j": j,
                        "prog_i": prog_i,
                        "prog_j": prog_j,
                        "sem_str_i": sem_str_i,
                        "sem_str_j": sem_str_j
                    })
        pbar.update(1)
        pbar.set_description(f"Currently have {len(all_pairs)} pairs")
            
equal_pairs = [p for p in all_pairs if p["sem_str_i"] == p["sem_str_j"]]
not_equal_pairs = [p for p in all_pairs if p["sem_str_i"] != p["sem_str_j"]]
output_dir = "/data0/shypula/prog_diversity/codebertscore_vs_semantic_analysis"

print(f"Found {len(all_pairs)} pairs, {len(equal_pairs)} equal pairs, and {len(not_equal_pairs)} not equal pairs")
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# with open(os.path.join(output_dir, "all_pairs.json"), 'w') as f:
#     json.dump(all_pairs, f)
#     print(f"Saved all pairs to {os.path.join(output_dir, 'all_pairs.json')}")
# with open(os.path.join(output_dir, "equal_pairs.json"), 'w') as f:
#     json.dump(equal_pairs, f)
#     print(f"Saved equal pairs to {os.path.join(output_dir, 'equal_pairs.json')}")
# with open(os.path.join(output_dir, "not_equal_pairs.json"), 'w') as f:
#     json.dump(not_equal_pairs, f)
#     print(f"Saved not equal pairs to {os.path.join(output_dir, 'not_equal_pairs.json')}")
        
        
# else: 
#     with open("/data0/shypula/prog_diversity/codebertscore_vs_semantic_analysis/all_pairs.json", 'r') as f:
#         all_pairs = json.load(f)
#     with open("/data0/shypula/prog_diversity/codebertscore_vs_semantic_analysis/equal_pairs.json", 'r') as f:
#         equal_pairs = json.load(f)
#     with open("/data0/shypula/prog_diversity/codebertscore_vs_semantic_analysis/not_equal_pairs.json", 'r') as f:
#         not_equal_pairs = json.load(f)
#     print(f"Loaded {len(all_pairs)} pairs, {len(equal_pairs)} equal pairs, and {len(not_equal_pairs)} not equal pairs")
        
import random 

random.shuffle(equal_pairs)
random.shuffle(not_equal_pairs)

with open(os.path.join(output_dir, "equal_pairs_shuffled.json"), 'w') as f:
    # sample 5000 pairs or all pairs if less than 5000
    json.dump(equal_pairs[:5000], f)
    print(f"Saved shuffled equal pairs to {os.path.join(output_dir, 'equal_pairs_shuffled.json')}")
with open(os.path.join(output_dir, "not_equal_pairs_shuffled.json"), 'w') as f:
    # sample 5000 pairs or all pairs if less than 5000
    json.dump(not_equal_pairs[:5000], f)
    print(f"Saved shuffled not equal pairs to {os.path.join(output_dir, 'not_equal_pairs_shuffled.json')}")


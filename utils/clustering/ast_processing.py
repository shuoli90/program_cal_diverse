

import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from obfuscation.bobskater_obfuscator import obfuscateString

import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
import contextlib
import traceback
import ast
from astor import to_source
import copy
import ast 
from ast import NodeTransformer

import ast
import copy
from copy import deepcopy
import numpy as np 
import random 
from random import choices


import logging

logging.basicConfig(level=logging.INFO)




# def obfuscateString(s, *args, **kwargs):
#     # Parse string for AST
#     sAst = ast.parse(s)
#     # Walk the AST once total to get all the scope information
#     ftnv = FrameTrackingNodeVisitor()
#     ftnv.visit(sAst)
#     logging.getLogger(__name__).debug(ftnv.getRootFrame())
#     # Walk the AST a second time to obfuscate identifiers with
#     # queriable scope info
#     transformer = ObfuscationTransformer(ftnv.getRootFrame(), *args, **kwargs)
#     sAst = transformer.visit(sAst)
#     # Unparse AST into source code
#     return astunparse.unparse(sAst), transformer.names_generator.get_dictionary()

# from tree_sitter import Language, Parser

## let's use the bobskater obfuscator to obfuscate

## then we'll do AST subtrees in 3 ways
### 1. Traditional, no-obfuscation, including all terminals (incl fname + variable names)
### 2. Obfuscation, but we call the same function as 1, before we use the bobskater
### 3. Obfuscation, we use the NodeTransformer to modify the tree to canonicalize all ID's to 'x' and Constants to '1'

### Capabilities: Return a list of all subtrees 



class StripIDValueVisitor(ast.NodeTransformer):
    def visit(self, node):
        """Visit a node and transform specific attributes."""
        # Node is modified only if it has certain attributes
        if hasattr(node, 'id') and isinstance(node.id, str):
            node.id = 'x'
        if hasattr(node, 'name') and isinstance(node.name, str):
            node.name = 'x'
        if hasattr(node, 'arg') and isinstance(node.arg, str):
            node.arg = 'x'
        if hasattr(node, 'value') and isinstance(node.value, (int, float, str)):
            # Set to '1' or 1 based on original type
            node.value = type(node.value)('s') if isinstance(node.value, str) else 1
        
        # Recursively handle all child nodes
        self.generic_visit(node)  # Very important!
        return node


def strip_id_value(node):
    v = StripIDValueVisitor()
    return v.visit(node)


class DeleteNodeAtHeight(NodeTransformer): 
    def __init__(self, max_height, verbose=False):
        self.max_height = max_height
        self.verbose = verbose
        self.current_height = 0  # Adding an instance variable to track current height
    
    def generic_visit(self, node):
        """Override generic_visit to manage the height increment."""
        if self.current_height == self.max_height:
            if self.verbose:
                print(f"Removing node: {ast.dump(node)}")
            return None  # Remove the node by returning None
        self.current_height += 1
        # Continue with the original generic_visit, which will recursively visit children
        result = super().generic_visit(node)
        self.current_height -= 1
        return result

def find_ast_height(node):
    if not isinstance(node, ast.AST):
        return 0
    max_height = 0
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                max_height = max(max_height, find_ast_height(item))
        elif isinstance(value, ast.AST):
            max_height = max(max_height, find_ast_height(value))
    return 1 + max_height


class AstTruncator: 
    """wrapper class to hold all H - 2 ast's for a given ast.
    We have the original AST, and then we have the H - 1, H - 2, ... 2 asts. 
    If we used H = 1, it would only be a module which is not very interesting.
    """
    def __init__(self, ast, verbose=False):
        self.verbose = verbose
        self.ast = ast
        self.max_height = find_ast_height(self.ast)
        self.all_sub_asts = []
        if self.verbose:
            print(f"Max height of AST: {self.max_height}")
            print("making all sub asts")
        for i in range(2, self.max_height+1):
            if self.verbose:
                print("-"*40)
                print(f"Generating sub AST at height {i}")
            copy_ast = deepcopy(self.ast)
            self.all_sub_asts.append(DeleteNodeAtHeight(i, verbose=self.verbose).visit(copy_ast))
        assert set([find_ast_height(ast) for ast in self.all_sub_asts]) == set(range(2, self.max_height+1))
        # reverse order 
        self.all_sub_asts = self.all_sub_asts[::-1]
        if self.verbose:
            print("done making all sub asts")
                                     
    
    def get_sub_ast(self, height):
        if height < 2 or height >= self.max_height:
            raise ValueError(f"Height must be between 2 and {self.max_height - 1}")
        return self.all_sub_asts[height - 2]
    
    
    def __iter__(self):
        return iter(self.all_sub_asts)
    

class AstSubTree:
    def __init__(self, root, height, derivation: str =None):
        self.root = root
        self.height = height
        self.str = ast.dump(root) 
        self.derivation = derivation
        
    def __str__(self):
        if self.derivation is not None:
            return f"Height: {self.height}, Derivation: {self.derivation}\nAST: {ast.dump(self.root, indent=4)}"
        else: 
            return f"Height: {self.height}\nAST: {ast.dump(self.root, indent=4)}"
    
    def as_string(self):
        return self.str


  
def bottom_up_subtrees_for_ast(node, derivation_path=[], visited=set(), verbose=False):
    """
    Given an AST node, bottom-up, return all sub-trees with the existing terminals. Keep a visited set to avoid re-visiting the same paths
    A driver function for all_subtrees calls this at each step, removing some terminals, thus allowing us to get all subtrees, but we need to track visited paths 
    to avoid re-visiting the same paths
    """
    subtrees = []
    max_height = 0 
    current_derivation_list = derivation_path + [type(node).__name__]
    

    any_children_new = False
    children_subtrees = []
    for i, child in enumerate(ast.iter_child_nodes(node)):
        child_derivation = current_derivation_list + [f"child_{i}"]
        child_subtrees, child_max_height, child_is_new, child_visited_set = bottom_up_subtrees_for_ast(child, child_derivation, visited)
        # sanity check here 
        if not child_is_new: 
            assert len(child_subtrees) == 1
            assert visited == child_visited_set
        
        visited.update(child_visited_set)
        max_height = max(max_height, child_max_height)
        if child_is_new:
            children_subtrees.extend(child_subtrees)
        any_children_new = any_children_new or child_is_new # if any children (recursively) is novel / changed, we'll add all subtrees received 
    
    max_height += 1
    
    this_node = [AstSubTree(node, max_height, "/".join(current_derivation_list))]
    
    # check if this is a terminal 
    if max_height == 1: 
        current_derivation = "/".join(current_derivation_list)
        # if current_derivation in visited: 
        is_new = current_derivation not in visited # we've been down this path before
        if verbose:
            print(f"Processing {current_derivation}, is_new: {is_new}")
            print("Visited so far:", visited)
        visited.add(current_derivation)
        return this_node, max_height, is_new, visited

    # we propagate this node up as well as all children subtrees returned 
    # if the node's subtree is new (any of its children is new), we add the node to the children subtrees (if any) and return the combined list
    # else, we only propagate this node up, and do not add any children subtrees
    
    is_new = any_children_new
    subtrees = children_subtrees + this_node if is_new else this_node 
    # if we are the root node and not is_new, we should return [] 
    # this is not necessary, because at each iteration in all_subtrees, the root node will always be modified as we will always truncate 
    # subtrees = [] if is_root and not is_new else subtrees
    
    
    return subtrees, max_height, is_new, visited

def all_subtrees(node, verbose=False):
    """
    Given an AST node, return all subtrees of the node. We first make all subtrees of the truncated tree (remove children at each level)
    """
    all_subtrees = []
    visited = set()
    for truncated_tree in AstTruncator(node, verbose=verbose):
        if verbose:
            height = find_ast_height(truncated_tree)
            print("-"*40)
            print(f"Height of truncated tree: {height}")
        new_subtrees, _, _, visited = bottom_up_subtrees_for_ast(truncated_tree, [], visited, verbose=verbose)
        all_subtrees.extend(new_subtrees)
    return all_subtrees


def subtrees_from_code(source_code, obfuscate=False, strip_all=False, verbose=False):
    if obfuscate:
        try: 
            source_code, _ = obfuscateString(source_code)
        except Exception as e:
            # every now and then there is a bizzare error, so we'll skip it
            source_code = ""
    tree = ast.parse(source_code)
    if strip_all:
        tree = strip_id_value(tree)
    subtrees = all_subtrees(tree, verbose=verbose)
    return subtrees  # Return the list of subtrees for the entire AST


class AllSubtreeAnalysis: 
    def __init__(self, source_code, verbose=False):
        self.source_code = source_code
        try: 
            self.orig_ast = ast.parse(source_code)
            self.plain_subtrees = subtrees_from_code(source_code, verbose=verbose)
            self.strip_subtrees = subtrees_from_code(source_code, strip_all=True, verbose=verbose)
            self.obf_subtrees = subtrees_from_code(source_code, obfuscate=True, verbose=verbose)
        except SyntaxError as e:
            self.orig_ast = []
            self.plain_subtrees = []
            self.strip_subtrees = []
            self.obf_subtrees = []
            if verbose:
                logging.info(f"Syntax Error processing source code: {source_code}")
                print(e)
            raise e
        except Exception as e:
            self.orig_ast = []
            self.plain_subtrees = []
            self.strip_subtrees = []
            self.obf_subtrees = []
            logging.critical(f"Error processing source code")
            err_msg = traceback.format_exc()
            logging.critical(err_msg)
            raise e
            
     
    @staticmethod
    def filter_below_height(subtrees, height = None):
        if height is None:
            return subtrees
        return [subtree for subtree in subtrees if subtree.height == height]
    
    @staticmethod
    def subtrees_as_string(subtrees): 
        return [subtree.as_string() for subtree in subtrees]
    
    def get_plain_subtrees(self, max_height = None):
        subtrees = self.filter_below_height(self.plain_subtrees, max_height)
        return self.subtrees_as_string(subtrees)

    
    def get_stripped_subtrees(self, max_height = None):
        subtrees = self.filter_below_height(self.strip_subtrees, max_height)
        return self.subtrees_as_string(subtrees)
    
    def get_obfuscated_subtrees(self, max_height = None):
        subtrees = self.filter_below_height(self.obf_subtrees, max_height)
        return self.subtrees_as_string(subtrees)
    
    def get_subtrees(self, typ: str, max_height = None):
        if typ == "plain":
            return self.get_plain_subtrees(max_height)
        elif typ == "stripped":
            return self.get_stripped_subtrees(max_height)
        elif typ == "obfuscated":
            return self.get_obfuscated_subtrees(max_height)
        else:
            raise ValueError("Invalid type, must be one of 'plain', 'stripped', or 'obfuscated'")
    
    

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


# with tqdm_joblib(tqdm(desc="Processing Programs", total=len(programs))) as progress_bar:
#         output_records = Parallel(n_jobs=n_jobs, backend='threading')(delayed(instrument_code_docker)(
#             program, testcases, outputs, tcgen_image, client, n_test_cases=n_test_cases, verbose_docker=verbose_docker, open_ended=open_ended
#         ) for program in programs)



def parallel_subtree_analysis(source_codes, n_jobs = -1, heights=[3,4,5,6], verbose=False, do_bootstrap=False, do_jaccard=False, iterations=100, subsample_size=2, jaccard_iterations=-1):
    assert len(heights) > 0, "Must provide at least one height to analyze"
    assert min(heights) >= 1, "Height must be at least 1"
    assert max(heights) <= 10, "Height must be at most 10"
    assert all(isinstance(height, int) for height in heights), "All heights must be integers"
    assert not (do_bootstrap and do_jaccard), "Cannot do both bootstrapping and Jaccard distance calculation"

    # _subtree_analysis = lambda source_code: AllSubtreeAnalysis(source_code)
    def _subtree_analysis(source_code):
        try: 
            return AllSubtreeAnalysis(source_code)
        except SyntaxError as e:
            if verbose:
                logging.info(f"Syntax Error processing source code: {source_code}")
                logging.info(traceback.format_exc())
            return None
        except Exception as e:
            logging.critical(f"Error processing source code: {source_code}")
            logging.critical(traceback.format_exc())
            return None
        
    with tqdm_joblib(tqdm(desc="Creating all subtrees", total=len(source_codes))) as progress_bar:
        results = Parallel(n_jobs=n_jobs)(delayed(_subtree_analysis)(source_code) for source_code in source_codes)
    result_dict = {
        "plain_subtrees": {}, 
        "stripped_subtrees": {},
        "obfuscated_subtrees": {}, 
    }
    def prop_distinct(subtree_analysis_list, height, typ: str):
        # get subtrees and flatten
        # if None, there likely was an error processing the source code
        subtrees = [subtree for subtree_analysis in subtree_analysis_list if subtree_analysis is not None for subtree in subtree_analysis.get_subtrees(typ, height)]
        # get proportion of distinct subtrees
        
        prop_distinct_plain = len(set(subtrees)) / len(subtrees) if len(subtrees) > 1 else np.nan
        return prop_distinct_plain
    
    def jaccard_distance(subtree_analysis_1, subtree_analysis_2, height, typ: str):
        # get subtrees and flatten
        # if None, there likely was an error processing the source code
        subtrees_1 = [subtree for subtree in subtree_analysis_1.get_subtrees(typ, height)]
        subtrees_2 = [subtree for subtree in subtree_analysis_2.get_subtrees(typ, height)]
        # get proportion of distinct subtrees
        n_intersect = len(set(subtrees_1).intersection(set(subtrees_2)))
        n_union = len(set(subtrees_1).union(set(subtrees_2)))
        jaccard_sim = n_intersect / n_union if n_union > 0 else np.nan
        return 1 - jaccard_sim
    
    # def prop_distinct(subtree_analysis_list, height, typ: str):
    #     # Flatten the list of subtrees while filtering out None values
    #     subtrees = [subtree for subtree_analysis in subtree_analysis_list if subtree_analysis is not None
    #                 for subtree in subtree_analysis.get_subtrees(typ, height)]
    #     # Calculate proportion of distinct subtrees
    #     return len(set(subtrees)) / len(subtrees) if len(subtrees) > 1 else np.nan
    
    def prop_distinct_bootstrapped(subtree_analysis_list, height, typ: str, iterations=100, subsample_size=2):
        import random 
        from random import choices
        random.seed(42)
        
        subtree_analysis_list = [subtree_analysis for subtree_analysis in subtree_analysis_list if subtree_analysis is not None]
        if len(subtree_analysis_list) < subsample_size:
            logging.warning(f"Subsample size is greater than the number of valid subtree analyses; returning NaN; out of {len(subtree_analysis_list)} there were {len(subtree_analysis_list)} non-empty analyses")
            return np.nan
        
        distinct_props = []
        for _ in range(iterations):
            # Sample with replacement from the subtree analysis list
            sampled_subtree_analysis = choices(subtree_analysis_list, k=subsample_size)
            # Calculate the proportion of distinct subtrees for the sample
            prop_distinct_value = prop_distinct(sampled_subtree_analysis, height, typ)
            distinct_props.append(prop_distinct_value)
    
        # Return the average of the calculated proportions, ignoring NaN values
        return np.nanmean(distinct_props)
    
    def jaccard_distance_driver(subtree_analysis_list, height, typ: str, iterations=-1):
        subtree_analysis_list = [subtree_analysis for subtree_analysis in subtree_analysis_list if subtree_analysis is not None]
        if len(subtree_analysis_list) < 2:
            logging.warning(f"Cannot calculate Jaccard distance with fewer than 2 valid subtree analyses; returning NaN; out of {len(subtree_analysis_list)} there were {len(subtree_analysis_list)} non-empty analyses")
            return np.nan
        jaccard_dists = []
        if iterations > 0:
            random.seed(42)
            for _ in range(iterations):
                # Sample with replacement from the subtree analysis list
                sampled_subtree_analysis = choices(subtree_analysis_list, k=2)
                # Calculate the Jaccard distance for the sample
                jaccard_dist = jaccard_distance(sampled_subtree_analysis[0], sampled_subtree_analysis[1], height, typ)
                jaccard_dists.append(jaccard_dist)
        else:
            for i in range(len(subtree_analysis_list)):
                for j in range(i+1, len(subtree_analysis_list)):
                    jaccard_dist = jaccard_distance(subtree_analysis_list[i], subtree_analysis_list[j], height, typ)
                    jaccard_dists.append(jaccard_dist)
        return np.nanmean(jaccard_dists)
     
    
    if do_bootstrap:
        for height in tqdm(heights, desc="Calculating Proportion of Distinct Subtrees"):
            result_dict["plain_subtrees"][height] = prop_distinct_bootstrapped(results, height, "plain", iterations, subsample_size)
            result_dict["stripped_subtrees"][height] = prop_distinct_bootstrapped(results, height, "stripped", iterations, subsample_size)
            result_dict["obfuscated_subtrees"][height] = prop_distinct_bootstrapped(results, height, "obfuscated", iterations, subsample_size)
            
    elif do_jaccard:
        for height in tqdm(heights, desc="Calculating Jaccard Distance"):
            result_dict["plain_subtrees"][height] = jaccard_distance_driver(results, height, "plain", jaccard_iterations)
            result_dict["stripped_subtrees"][height] = jaccard_distance_driver(results, height, "stripped", jaccard_iterations)
            result_dict["obfuscated_subtrees"][height] = jaccard_distance_driver(results, height, "obfuscated", jaccard_iterations)
            
    else: 
        for height in tqdm(heights, desc="Calculating Proportion of Distinct Subtrees"):
            result_dict["plain_subtrees"][height] = prop_distinct(results, height, "plain") 
            result_dict["stripped_subtrees"][height] = prop_distinct(results, height, "stripped")
            result_dict["obfuscated_subtrees"][height] = prop_distinct(results, height, "obfuscated")
        
    return result_dict


    
    
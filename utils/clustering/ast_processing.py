

import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from obfuscation.bobskater_obfuscator import obfuscateString


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

import ast
from astor import to_source
import copy

import ast
import copy


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


class AstSubTree:
    def __init__(self, root, height):
        self.root = root
        self.height = height
        self.str = ast.dump(root) 
        
    def __str__(self):
        return ast.dump(self.root, indent=4)
    
    def as_string(self):
        return self.str

def all_subtrees(node):
    """Analyze the subtree to compute the height and collect all subtrees,
       returning the list of subtrees including the current node with its height."""
    subtrees = []
    max_height = 0

    # Recursively compute the height of each child and collect subtrees
    for child in ast.iter_child_nodes(node):
        children_subtrees, child_height = all_subtrees(child)
        subtrees.extend(children_subtrees)
        max_height = max(max_height, child_height)
    
    # Increment to include the current node in the height calculation
    max_height += 1

    # Append the current node's subtree after processing all children
    subtrees.append(AstSubTree(node, max_height))
    return subtrees, max_height

def subtrees_from_code(source_code, obfuscate=False, strip_all=False):
    if obfuscate:
        source_code, _ = obfuscateString(source_code)
    tree = ast.parse(source_code)
    if strip_all:
        tree = strip_id_value(tree)
    subtrees, _ = all_subtrees(tree)
    return subtrees  # Return the list of subtrees for the entire AST


class AllSubtreeAnalysis: 
    def __init__(self, source_code):
        self.source_code = source_code
        self.orig_ast = ast.parse(source_code)
        self.plain_subtrees = subtrees_from_code(source_code)
        self.strip_subtrees = subtrees_from_code(source_code, strip_all=True)
        self.obf_subtrees = subtrees_from_code(source_code, obfuscate=True)
     
    @staticmethod
    def filter_below_height(subtrees, height = None):
        if height is None:
            return subtrees
        return [subtree for subtree in subtrees if subtree.height <= height]
    
    @staticmethid 
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
        reutrn self.subtrees_as_string(subtrees)
    
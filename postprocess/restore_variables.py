from typing import List, Tuple
import re
from penman import Graph
from postprocess.postprocess_utils import split_str_triple, get_initial_letter, name_new_variable
from AMR_utils import get_default_amr
from AMR_utils import noop_penman_decode
from AMRSimplifier import AMRSimplifier
# from postprocess.str_2_triple import amr_str_to_list, restore_vars
from postprocess.wikify import wikify_simple
import penman
from settings import PROJECT_DIR
# from etc import visualize_graph

# A source cannot have multiple target nodes with the same relation type
unique_roles = [":wiki", ":name", ":degree",
                ":ARG0", ":ARG1", ":ARG2", ":ARG3", ":ARG4", ":ARG5", ":ARG6", ":ARG7", ":ARG8", ":ARG9", ":ARG10",
                ":country", ":polarity",
                ":op1", ":op2", ":op3", ":op4", ":op5", ":op6", ":op7", ":op8", ":op9", ":op10"]

def starts_and_ends_with_quote(word):
    pattern = r'^".*?"$'
    return bool(re.match(pattern, word))


def is_attr_value(rel: str, tgt: str) -> bool:
    only_numbers = re.compile(r'^\d+$')
    if rel == ":mode" and tgt in ["imperative", "interrogative", "expressive"]:
        return True
    if rel == ":polarity" and tgt in ["+", "-"]:   # exception amr-unkown
        return True
    if rel == ":wiki":
        return True
    if rel == ":value" and tgt.lstrip('-+').isdigit():
        return True
    if starts_and_ends_with_quote(tgt):
        return True
    if re.match(only_numbers, tgt):
        return True

    return False

def create_instance_triples(var_2_instance: dict) -> List[Tuple[str]]:
    instance_triples = []
    for var, instance in var_2_instance.items():
        instance_triples.append((var, ":instance", instance))
    return instance_triples


def filter_attribute_instances(triples: List[Tuple[str]]) -> List[Tuple[str]]:
    # remove the attribute instance triples
    # e.g. polarity ( - , :instance , - )
    filtered = []
    for src, inst_rel, tgt in triples:
        if src == "-" or tgt == "-":
            continue
        else:
            filtered.append((src, inst_rel, tgt))
    return filtered


def get_variable(concept, var_2_instance, counter, force_new_var=False) -> str:
    # If force_new_var is True, always return a new variable
    if force_new_var:
        return name_new_variable(concept, counter)

    # Check if the token is already associated with a variable
    if concept in var_2_instance.values():
        for var, instance in reversed(var_2_instance.items()): # reversed to suppose co-referring nodes are closely located
            if instance == concept:
                return var

    # If not found, create a new variable
    return name_new_variable(concept, counter)


def get_variable_strict(curr_concept, var_2_instance, counter, curr_rel, triples, force_new_var=False) -> str:
    # If force_new_var is True, always return a new variable

    if force_new_var:
        return name_new_variable(curr_concept, counter)

    # Check if the token is already associated with a variable
    if curr_concept in var_2_instance.values():
        for var, concept in reversed(var_2_instance.items()): # reversed to suppose co-referring nodes are closely located
            if concept == curr_concept:
                if not has_redundant_role(var, curr_rel, triples):
                    return var

    # If not found, create a new variable
    return name_new_variable(curr_concept, counter)

def restore_single_instance_var(triple):
    assert len(triple) == 1
    src, rel, tgt = triple[0]
    var = src[0]
    return [(var, rel, tgt)]


def normalize(triple):
    src, rel, tgt = triple
    if rel.endswith("-of"):
        return tgt, rel[:-3], src
    return triple

def is_disconnected(src:str, triples: list) -> bool:
    variables = [triple[0] for triple in triples] + [triple[2] for triple in triples]
    if src not in variables:
        return True
    return False

def has_cycle(current_triple: tuple, triples:tuple)-> bool:
    # it only checks cycled neighbor nodes (not a sub graph)
    # it only checks cycles with the same relation (e.g. A :ARG0 C == C :ARG0-of A)
    curr_triple = normalize(current_triple)
    reversed_triple = (curr_triple[2], curr_triple[1], curr_triple[0])
    for triple in triples:
        if normalize(triple)[0]== reversed_triple[0] and normalize(triple)[2]== reversed_triple[2]:
            return True
    return False


def correct_prev_error(curr_src, curr_src_var, var_2_node, new_triples):

    # this is called when a graph is disconnected (src node is not in the graph)
    # and the previous graph has an error in the variable assignment
    # we fix it by reassigning the variable to the correct node (priory to the close node from the current node)
    # warning! because of this, the order of the triples is important (the graph should follow the original graph in DFS order)

    triples_reverse_order = list(reversed(new_triples))
    for i, triple in enumerate(triples_reverse_order):
        if triple[2] in var_2_node and var_2_node[triple[2]] == curr_src:
            triples_reverse_order[i] = (triple[0], triple[1], curr_src_var)
            return list(reversed(triples_reverse_order))

    print("something is wrong.. couldn't find the correct node to connect")
    return new_triples

def has_redundant_role(curr_src, curr_rel, all_triples):
    if curr_rel not in unique_roles:
        return False
    for triple in all_triples:
        if triple[0] == curr_src and triple[1] == curr_rel:
            return True

def get_src_variable(src, last_tgt, last_src, var2node, var_num_counter, force_new_variable=False):
    is_second_run = force_new_variable # if force_new_variable is True, it's the second run to re-attribute the variable
    if src == last_tgt[1]:  # in triplet, tgt node is repeated (if not a leaf) in the next source triplet
        src_v = last_tgt[0]

    else:
        is_name_reentrancy = (last_src[1] == "name")  # if previous src is a name concept, it's most likely a reentrancy
        if not is_second_run:
            force_new_variable = (
                        src == "name" and not is_name_reentrancy)  # new variable for name concept (except for reentrancy)

        src_v = get_variable(src, var2node, var_num_counter, force_new_variable)
        if src_v not in var2node:
            var2node[src_v] = src


    last_src = (src_v, src)

    return src_v, last_src


def get_src_variable_strict(src, last_tgt, last_src, var2node, curr_rel, triples, var_num_counter):
    if src == last_tgt[1]:  # in triplet, tgt node is repeated (if not a leaf) in the next source triplet
        src_v = last_tgt[0]

    else:
        is_name_reentrancy = (last_src[1] == "name")  # if previous src is a name concept, it's most likely a reentrancy
        force_new_variable = (src == "name" and not is_name_reentrancy)  # new variable for name concept (except for reentrancy)
        src_v = get_variable_strict(src, var2node, var_num_counter, curr_rel, triples, force_new_variable)
        if src_v not in var2node:
            var2node[src_v] = src

    last_src = (src_v, src)

    return src_v, last_src



def get_tgt_variable(rel, tgt, var2node, var_num_counter, force_new_variable=False):
    tgt_v = tgt

    if force_new_variable:
        tgt_v = get_variable(tgt, var2node, var_num_counter, force_new_variable)
        var2node[tgt_v] = tgt
        last_tgt = (tgt_v, tgt)
        return tgt_v, last_tgt

    # create a new variable if it's not an attribute value
    if not is_attr_value(rel, tgt) :  # tgt may not be an instance (e.g. +, 01324, imperative, '"Korea"', etc.)
        tgt_v = get_variable(tgt, var2node, var_num_counter, tgt == "name")
        var2node[tgt_v] = tgt
        last_tgt = (tgt_v, tgt)

    # if it's an attribute value, initialize the last_tgt to None
    else:
        last_tgt = (None, None)

    return tgt_v, last_tgt

def restore_vars(triples: List[Tuple[str]]) -> List[Tuple[str]]:
    # given (src, tgt, rel) triples
    # consider every same node as re-entrancy / co-reference except for named entities (name)

    new_triples = []
    var2node = {}
    var_num_counter = {}
    last_src = (None, None) # necessary for variable naming
    last_tgt = (None, None) # necessary for variable naming

    for i, triple in enumerate(triples):
        src, rel, tgt = triple
        src_v, last_src = get_src_variable(src, last_tgt, last_src, var2node, var_num_counter)
        tgt_v, last_tgt = get_tgt_variable(rel, tgt, var2node, var_num_counter)
        new_triples.append((src_v, rel, tgt_v))

    # lastly, add the instance triples
    instance_triples = create_instance_triples(var2node)
    new_triples.extend(instance_triples)

    return new_triples

def restore_vars_with_constraint_strict(triples: List[Tuple[str]]) -> List[Tuple[str]]:
    # strict constraint to restore variables
    # do not allow

    new_triples = []
    var2node = {}
    var_num_counter = {}
    last_src = (None, None)  # necessary for variable naming
    last_tgt = (None, None)  # necessary for variable naming

    for i, triple in enumerate(triples):
        src, rel, tgt = triple

        src_v, last_src = get_src_variable_strict(src, last_tgt, last_src, var2node, rel, new_triples, var_num_counter) # do not allow redundant roles
        tgt_v, last_tgt = get_tgt_variable(rel, tgt, var2node, var_num_counter)

        # if cycle is detected, reassign the target node variable
        if has_cycle((src_v, rel, tgt_v), new_triples):
            tgt_v, last_tgt = get_tgt_variable(rel, tgt, var2node, var_num_counter, force_new_variable=True)

        # it is possible that it still has a redundant role, reassign the source node variable
        if has_redundant_role(src_v, rel, new_triples):
            src_v, last_src = get_src_variable(src, last_tgt, last_src, var2node, var_num_counter, force_new_variable=True)

        # as a final check, if current source variable is not connected to any of the existing nodes, reassign its variable
        if i!= 0 and is_disconnected(src_v, new_triples):
            new_triples = correct_prev_error(src, src_v, var2node, new_triples)

        new_triples.append((src_v, rel, tgt_v))

    # lastly, add the instance triples
    instance_triples = create_instance_triples(var2node)
    new_triples.extend(instance_triples)

    return new_triples

def linearize_and_backlinearize(amr_orig) -> Tuple[Graph, Graph]:
    # amr = to_single_line_amr(amr_orig)
    g = noop_penman_decode(amr_orig)
    simplifier = AMRSimplifier(g)
    amr = simplifier.simplify()
    triples = split_str_triple(amr)

    # TODO: code below can be replaced with prediction_to_penman

    amr = wikify_simple(triples)
    if len(amr) == 1 and amr[0][1] == ":instance":
        amr = restore_single_instance_var(amr)
    else:
        amr = restore_vars_with_constraint_strict(amr)
    g2 = Graph(amr)
    try:
        penman.encode(g2)
        return g, g2
    # # print except type
    except Exception as e:
        print(e)
        # visualize_graph.visualize_from_triplets(g.edges())
        return g, get_default_amr(return_type="graph")



if __name__ == '__main__':

    # count invalid graphs
    from pathlib import Path
    from settings import DATA_DIR
    from tqdm import tqdm
    import subprocess
    from etc.cd import cd
    from copy import deepcopy

    split = "dev"
    filename= "restore_var_predicate_new_variable_combi_wo_wiki"
    linearized_wo_variable = DATA_DIR / "amr" / "fr" / split / "fr-amr.wo_var.tri"
    pm_file = DATA_DIR / "amr" / "fr" / split / "fr-amr.pm"
    destin =  Path(str(linearized_wo_variable).replace(".wo_var.tri", f".{filename}.pm"))
    not_handled = Path(str(linearized_wo_variable).replace(".wo_var.tri", f".{filename}.failed.pm"))
    with open(pm_file) as f:
        pms = f.read().split("\n\n")


    # with open(linearized_wo_variable, "r") as f, open(destin, "w") as f2, open(not_handled, "w") as f3:
    # # with open(linearized_wo_variable, "r") as f, open(destin, "w") as f2:
    #     graphs = []
    #     amrs = f.read().split("\n")
    #
    #     for i, a in enumerate(tqdm(amrs)):
    #         if a!= "" :
    #             triples = split_str_triple(a)
    #             triples = wikify(triples)
    #             if len(triples) == 1 and triples[0][1] == ":instance":
    #                 amr = restore_single_instance_var(triples)
    #             else:
    #                 # amr = restore_vars(triples, allow_redundant_roles=False)
    #                 amr = restore_vars_with_constraint_strict(triples)
    #             g2 = Graph(amr)
    #             try:
    #                 encoded = penman.encode(g2)
    #             ## print except type
    #             except Exception as e:
    #                 graph = restore_vars(triples)
    #                 graph = Graph(graph)
    #                 encoded = penman.encode(graph)
    #                 original = pms[i]
    #                 f3.write(original + "\n\n")
    #
    #             f2.write(encoded + "\n\n")


    with open(linearized_wo_variable, "r") as f, open(destin, "w") as f2:
        graphs = []
        amrs = f.read().split("\n")

        for i, a in enumerate(tqdm(amrs)):
            if a!= "" :
                triples = split_str_triple(a)
                # triples = wikify(triples)
                original_triples = deepcopy(triples)
                if len(triples) == 1 and triples[0][1] == ":instance":
                    amr = restore_single_instance_var(triples)
                else:
                    # amr = restore_vars(triples, allow_redundant_roles=False)
                    amr = restore_vars_with_constraint_strict(triples)
                g2 = Graph(amr)
                try:
                    encoded = penman.encode(g2)
                ## print except type
                except Exception as e:
                    try:
                        print("restoring without contraints")
                        amr = restore_vars(original_triples)
                        g2 = Graph(amr)
                        encoded = penman.encode(g2)
                    except Exception as e:
                        graph = get_default_amr(return_type="graph")
                        encoded = penman.encode(graph)

                f2.write(encoded + "\n\n")

    with cd(PROJECT_DIR):
        subprocess.check_call(
            ["python", "smatch_modified/compute_smatch.py", "-f", destin, pm_file, "-r", "5", "--significant", "3"])

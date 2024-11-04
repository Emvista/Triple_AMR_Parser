from postprocess.wikify import wikify_simple
import penman
from postprocess.postprocess_utils import split_str_triple, remove_space_between_quotes
from AMR_utils import get_default_amr, safe_encoding
from penman import Graph
import networkx as nx
from networkx import DiGraph
from penman.graph import Instance
from typing import List
from postprocess.restore_variables import restore_vars_with_constraint_strict, restore_single_instance_var, restore_vars
import re
from postprocess.restore_variables import starts_and_ends_with_quote
from AMR.smatch.amr import AMR as AMRGraph
# from etc.visualize_graph import visualize_from_graph
import uuid
from AMR_utils import get_default_amr, to_single_line_amr
import logging

logfmt = '%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(format=logfmt, datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()

def undo_ne_recategorize(triples: list):
    # undo the named entity recategorization
    new_triples = []
    entire_variables = set([triple[0] for triple in triples])
    variable_start_with_n = [var for var in entire_variables if re.match(r'n\d?', var)] # e.g. n, n2, n3
    n_idx = len(variable_start_with_n) + 1

    for triple in triples:
        src, rel, tgt = triple
        # if the relation is :name
        if rel.strip() == ":name":
            # split the tgt into names and create a new node for each name with 'opx' relation
            names = tgt.replace('"', "").split("_")

            src_var = f"n{n_idx}" if n_idx > 0 else "n" # if there are multiple names, create a new variable for each name
            for op_id, name in enumerate(names, 1):
                new_triples.append((src_var, f":op{op_id}", f'"{name.strip()}"'))

            # add new instance triple for the variable
            new_triples.append((src_var, ":instance", "name"))
            n_idx += 1
            tgt = src_var

        new_triples.append((src, rel, tgt))
    return new_triples

def attach_instances(sub_graph:DiGraph, instance_edges:list[Instance])->Graph:
    # Extracting relational edges from the graph
    relational_edges = [(src, attribute["label"], tgt) for src, tgt, attribute in sub_graph.edges(data=True)]

    # Extracting instance edges that start from a node present in the graph
    subgraph_instance_edges = [edge for edge in instance_edges if edge[0] in sub_graph.nodes()]

    # Combine relational edges and instance edges
    total_edges = relational_edges + subgraph_instance_edges

    # Create a penman graph
    sub_graph_pm = Graph(total_edges)
    return sub_graph_pm


def get_largest_subgraph(disconnected_graph: Graph)-> DiGraph:
    # get the largest spanning subgraph from a disconnected graph
    logger.info("Possibly disconnected graph, getting the largest subgraph")
    directed_graph = DiGraph()
    directed_graph.add_nodes_from(disconnected_graph.variables())  # not sure if this is necessary since the nodes are already added when adding edges
    directed_graph.add_edges_from((src, tgt, {'label': rel}) for src, rel, tgt in disconnected_graph.edges())

    connected_components = list(nx.weakly_connected_components(directed_graph))
    largest_component = max(connected_components, key=len)
    largest_subgraph = directed_graph.subgraph(largest_component)

    return largest_subgraph


def add_variables_to_triples(triples: list):

    if len(triples) == 1 and triples[0][1] == ":instance":
        triples_with_vars = restore_single_instance_var(triples)
        return triples_with_vars

    else:
        return restore_vars_with_constraint_strict(triples)


def recover_and_encode(triples) -> str:
    """
    Encode a Penman graph trying to recover from encoding errors if possible

    :param triples: List of Penman triples
    :return: Encoded Penman graph if successful, otherwise default AMR
    """
    graph = penman.Graph(triples)
    # safe_encoding returns None if encoding fails, otherwise returns the encoded graph
    encoded_graph = safe_encoding(graph)

    if encoded_graph:
        return encoded_graph

    # if encoding fails, try to get the largest subgraph and encode it instead
    else:
        sub_graph = get_largest_subgraph(graph)
        sub_graph_with_instances = attach_instances(sub_graph, graph.instances())
        encoded_graph = safe_encoding(sub_graph_with_instances)

        if encoded_graph:
            return encoded_graph

    # If both encodings fail, return a default AMR
    return get_default_amr(return_type="pm_1line")


def update_prev_tgt(prev_tgt, curr_src, curr_tgt, curr_rel):
    if curr_rel.startswith(":op") and curr_src=="name":
        return prev_tgt
    else:
        return curr_tgt


def is_loosely_attribute(value):
    # to loosely check if this is an attribute value or not ..
    # this is less risky when we do it loosely to not skip non attribute values
    attribute_values = ["imperative", "interrogative", "expressive", "+", "-"]
    # number + (special characters) + (number)  e.g. 3, 4, 3:45, 3-4
    only_numbers = re.compile(r'^\d*[-!$%^&*()_+|~=`{}\[\]:";\'<>?,./]?\d*$')

    if value in attribute_values or only_numbers.match(value):
        return True
    if starts_and_ends_with_quote(value):
        return True
    if re.match(only_numbers, value):
        return True

    return False

def filter_erroneous_instance_triples(instance_triples):
    # filter when there are more than one instance for the same variables
    vars = set()
    after_filter = []
    for instance_triple in instance_triples:
        if instance_triple[0] not in vars:
            vars.add(instance_triple[0])
            after_filter.append(instance_triple)
        else:
            logger.info(f"Filtering erroneous instance triple: {instance_triple}")
    return after_filter

def filter_erroneous_rel_triples(vars_with_instance, rel_triples):

    after_filter = []
    for rel_triple in rel_triples:
        if not (rel_triple[0] in vars_with_instance) and not is_loosely_attribute(rel_triple[0]):
            logger.info(f"Skipping erroneous rel triple: {rel_triple}")
            continue
        if not (rel_triple[2] in vars_with_instance) and not is_loosely_attribute(rel_triple[2]):
            logger.info(f"Skipping erroneous rel triple: {rel_triple}")
            continue

        after_filter.append(rel_triple)
    return after_filter

def skip_erroneous_triples(triples:List):
    # filter out erroneous triples which are not possible to recover (during parsing for smatch evaluation)

    # step 1: remove triples which include two instances having the same variable
    instance_triples = [triple for triple in triples if triple[1] == ":instance"]
    instance_triples = filter_erroneous_instance_triples(instance_triples)
    vars_with_instance = set([instance_triple[0] for instance_triple in instance_triples])

    # step 2: remove rel triples which include variables without instance definition
    rel_triples = [triple for triple in triples if triple[1] != ":instance"]
    rel_triples = filter_erroneous_rel_triples(vars_with_instance, rel_triples)

    return instance_triples + rel_triples

def pass_sanity_check(amr_penman_multiline):
    amr_singlle_line = to_single_line_amr(amr_penman_multiline)
    try:
        parsed = AMRGraph.parse_AMR_line(amr_singlle_line)
        if parsed is None:
            logger.info(f"Invalid AMR graph: {amr_singlle_line}")
            return False
    except:
        logger.info(f"Invalid AMR graph: {amr_singlle_line}")
        return False

    return True


def prediction_to_penman(triple_in_str:str, restore_wiky=False, do_variable_restore=False, do_invrole_restore=False) -> str:
    # should return a multiline penman string

    # 0) temporary - this should be obsolete when triplet special tokens are modified (e.g. amr -> amr_tri)
    triple_in_str = triple_in_str.replace(" -unknown", " amr-unknown")

    # 0.1) remove spaces between quotes (if any)
    triple_in_str = remove_space_between_quotes(triple_in_str)

    # 1) split the linearized amr into triples
    triples = split_str_triple(triple_in_str)
    if not triples: # if the linearized amr is empty
        return get_default_amr(return_type="pm_1line")

    # 2) wikify the triples (if no_wikify is False)
    if restore_wiky:
        triples = wikify_simple(triples)

    # 3) restore variables (if do_variable_restore is True)
    if do_variable_restore:
        try:
            triples = add_variables_to_triples(triples)
        except Exception as e:
            logger.info("Error in restoring variables: ", e)
            triples = get_default_amr(return_type="list")

    # 4) undo ne recategorize
    triples = undo_ne_recategorize(triples)

    # 5) remove triples which include variables without instace definition
    # such error happens when model generates triples with skipping (var :instance concept) triples
    if not do_variable_restore:
        triples = skip_erroneous_triples(triples)

    # 6) linearize the triples while recovering from encoding errors
    amr_penman = recover_and_encode(triples)

    # 7) Final sanity check and if it fails, return default AMR
    if not pass_sanity_check(amr_penman):
        amr_penman = get_default_amr(return_type="pm")

    return amr_penman


if __name__ == '__main__':
    from settings import DATA_DIR, PROJECT_DIR
    from tqdm import tqdm
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, help="input file path (linearized tri or predictions.txt)")
    parser.add_argument("--without_variables", action="store_true", help="if the file does not contain variables or not, restore it if it does not contain variables")
    parser.add_argument("--restore_wiki", action="store_true", help="restore wiki tokens in the input file")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        linearized_amr = f.read().strip().split("\n")

    restored = args.input_file.parent / "restored" / (args.input_file.name + ".wiki.restore.pm")
    restored.parent.mkdir(parents=True, exist_ok=True)

    with open(restored, "w") as f:
        for amr in tqdm(linearized_amr):
            restored = prediction_to_penman(amr, do_variable_restore=args.without_variables, do_invrole_restore=False, restore_wiky=args.restore_wiki)
            f.write(restored + "\n\n")


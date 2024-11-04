from typing import List, Tuple
from AMR_utils import to_single_line_amr
import penman
from postprocess.postprocess_utils import name_new_variable
from penman.models import amr as amr_model
from penman.models import noop as noop_model
from preprocess.preprocess_utils import ne_recategorize, add_emptyspace, add_space_between_quotes
from pathlib import Path

'''
This script is used to linearized penman to penman while: 
1) reordering variables to match the order of occurrence
2) recategorize named entities 
'''

def replace_old_var_names(triples: List[Tuple[str]], old_var_to_new_var: dict) -> List[Tuple[str]]:
    new_triples = []
    for triple in triples:
        src, rel, tgt = triple
        if src in old_var_to_new_var:
            src = old_var_to_new_var[src]
        if tgt in old_var_to_new_var:
            tgt = old_var_to_new_var[tgt]
        new_triples.append((src, rel, tgt))
    return new_triples

def rename_variables(triples: List[Tuple[str]]) -> List[Tuple[str]]:
    # instead of random order, e.g. t3 and then t
    # the occuring var should be t and then t2 from left to right
    instance_triples = [triple for triple in triples if triple[1] == ":instance"]
    counter = {}
    old_var_to_new_var = {}
    for triple in instance_triples:
        old_var, role, instance = triple
        new_var = name_new_variable(instance, counter)

        if old_var != new_var:
            old_var_to_new_var[old_var] =  new_var

    triples = replace_old_var_names(triples, old_var_to_new_var)
    return triples


def pm_to_triple_wo_wiki(single_line_amr:str, no_inverse_role=False) -> List[Tuple[str]]:
    # return a list of triples
    # with the first triple being the instance triple
    # and the rest being the other triples
    # filter wiki triples out
    parse_model = amr_model if no_inverse_role else noop_model
    graph = penman.decode(single_line_amr, model=parse_model.model)
    total_triples = graph.triples # returns dfs order triples
    instance_triples = []
    other_triples = []

    for triple in total_triples:
        # instance triples
        if triple[1] == ":instance":
            instance_triples.append(triple)

        else:
            if ":wiki" not in triple[1]:
                other_triples.append(triple)

    return instance_triples + other_triples # reordered triples starting from instances

def penman_preprocess(amr: str, no_inverse_role=False) :
    amr = to_single_line_amr(amr)
    amr = ne_recategorize(amr)
    triples = pm_to_triple_wo_wiki(amr, no_inverse_role)
    triples = rename_variables(triples) # rename variables to follow the order of occurence
    graph = penman.Graph(triples) # create a graph from the triples
    return penman.encode(graph)


def main(input_path, output_path, no_inverse_role=False):
    with open(input_path, "r") as f, open(output_path, "w") as f2:
        amr = f.read().split("\n\n")
        for i, a in enumerate(tqdm(amr)):
            if a!= "" :
                graph = penman_preprocess(a, no_inverse_role)
                single_line_graph = to_single_line_amr(graph)
                space_between_parenthesis = add_emptyspace(single_line_graph)
                space_between_quotes = add_space_between_quotes(space_between_parenthesis)
                f2.write(space_between_quotes + "\n")

if __name__ == '__main__':
    import argparse
    from tqdm import tqdm

    args = argparse.ArgumentParser()
    args.add_argument("--input_path", type=str, help="input path to pm file")

    args = args.parse_args()
    output_path = Path(args.input_path).with_suffix(".penman")
    main(args.input_path, output_path,)



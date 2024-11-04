from typing import List, Tuple, Union
from penman import Graph
from penman.models import noop
import penman
import re
from smatch_modified.compute_smatch import get_amr_match, compute_f
from penman.exceptions import LayoutError

def safe_encoding(graph:Graph)->Union[str, None]:
    # try to encode the graph, if it fails, return None
    try:
        return penman.encode(graph)
    except (LayoutError, AttributeError) as e :
        print(e)
        return None

def safe_decoding(penman_str) -> Union[Graph, None]:
    # try to decode the amr string, if it fails, return None
    try:
        return penman.decode(penman_str)
    except:
        print("Error decoding the AMR string: {}".format(penman_str))
        return None

def noop_penman_decode(penman_str: str, model=noop.model) -> Graph:
    # decode the penman string to a graph without normalizing (e.g. :ARG1-of -> :ARG1)
    return penman.decode(penman_str, model=model)

def triples_to_str(triples: List[Tuple[str]]) -> str:
    # convert a list of triples to a string
    formatted_strings = [f"( {src}, {rel}, {tgt} )" for src, rel, tgt in triples]
    return " ".join(formatted_strings)

def compute_smatch(amr1_str: str, amr2_str: str) -> float:
    # compute the smatch_modified score between two amr strings
    best_match_num, test_triple_num, gold_triple_num = get_amr_match(amr1_str, amr2_str)
    return compute_f(best_match_num, test_triple_num, gold_triple_num)

def get_default_amr(return_type="str", silent=False) -> Union[str, List, Graph]:
    # TODO: fix the default amr type and string
    if not silent:
        print("=== get default graph ====")
    if return_type.lower() == "list":
        return [('w', ':instance', 'want-01'), ('w', ':ARG0', 'b'), ('b', ':instance', 'boy'), ('w', ':ARG1', 'g'), ('g', ':instance', 'go-02'), ('g', ':ARG0', 'b')]

    elif return_type.lower() == "graph":
        return penman.decode('(w / want-01 :ARG0 (b / boy) :ARG1 (g / go-02 :ARG0 b))')

    elif return_type.lower() == "pm_1line":
        return '(w / want-01 :ARG0 (b / boy) :ARG1 (g / go-02 :ARG0 b))'

    elif return_type.lower() == "pm":
        return '''
        (w / want-01 
            :ARG0 (b / boy) 
            :ARG1 (g / go-02 
                :ARG0 b))
                '''.strip()


    else:
        raise ValueError("return_type must be either 'list', 'str', or 'graph'")

def to_single_line_amr(penman_str: str) -> str:
    # remove newlines and extra spaces
    # from the penman string
    unformatted = penman_str.replace("\n", " ")
    unformatted = re.sub(r"\s+", " ", unformatted)
    return unformatted



if __name__ == '__main__':
    from settings import DATA_DIR
    pm_file = DATA_DIR / "amr" / "fr" / "silvertest" / "fix" / "fr-amr.pm"
    destin_file = DATA_DIR / "amr" / "fr" / "silvertest" / "fix" / "fr-amr.amr"
    with open(pm_file, "r") as f:
        entire_doc = f.read()
        pms = entire_doc.split("\n\n")
        single_lines = []

        for pm in pms:
            single_line_amr = to_single_line_amr(pm)
            single_lines.append(single_line_amr)

    with open(destin_file, 'w') as f:
        f.write("\n".join(single_lines))





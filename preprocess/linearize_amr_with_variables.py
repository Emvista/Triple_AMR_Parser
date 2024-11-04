from typing import List, Tuple
from AMR_utils import to_single_line_amr
import penman
from postprocess.postprocess_utils import concat_triples_to_str, name_new_variable
from penman.models import amr as amr_model
from penman.models import noop as noop_model
from preprocess.preprocess_utils import ne_recategorize, add_space_between_quotes

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
    # the occuring var should be t and then t2
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


def pm_str_to_list(single_line_amr:str, no_inverse_role=False) -> List[Tuple[str]]:
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

        # other triples but wiki
        else:
            if triple[1] != ":wiki":
                other_triples.append(triple)

    return instance_triples + other_triples # reordered triples starting from instances

def penman_to_linearized(amr: str, no_inverse_role=False) -> str :
    amr = to_single_line_amr(amr)
    amr = ne_recategorize(amr)
    triples = pm_str_to_list(amr, no_inverse_role)
    triples = rename_variables(triples) # rename variables to follow the order of occurence
    linearized_amr = concat_triples_to_str(triples)
    return linearized_amr


def main(input_path, output_path, no_inverse_role=False):
    with open(input_path, "r") as f, open(output_path, "w") as f2:
        amr = f.read().split("\n\n")
        for i, a in enumerate(tqdm(amr)):
            if a!= "" :
                a = to_single_line_amr(a)
                # ne_recategorize
                linearized = penman_to_linearized(a, no_inverse_role)
                space_added_between_quotes = add_space_between_quotes(linearized)
                f2.write(space_added_between_quotes + "\n")

if __name__ == '__main__':
    import argparse
    from settings import get_data_path
    from AMR_utils import noop_penman_decode
    from tqdm import tqdm

    # args = argparse.ArgumentParser()
    # args.add_argument("--input_path", type=str, help="input path to pm file")
    # args.add_argument("--output_path", type=str, default=None, help="output path to save the linearized amr")
    # args.add_argument("--no_inverse_role", action="store_true", help="no inverse roles in the AMR simplification")
    # args = args.parse_args()
    #
    # if args.output_path is None:
    #     if args.no_inverse_role:
    #         args.output_path = args.input_path.replace(".pm", ".wo_invrole.tri")
    #     else:
    #         args.output_path = args.input_path.replace(".pm", ".tri")
    #
    # main(args.input_path, args.output_path, args.no_inverse_role)


# single amr test case
    amr = """
        (s / seem-01 :polarity -
          :ARG1 (s2 / see-01
                :ARG0 w
                :ARG1 (p / person
                      :mod (a / any)
                      :mod (n / nutter)
                      :ARG0-of (d / dig-01)
                      :ARG0-of (a2 / acknowledge-01
                            :ARG1 (t / thing
                                  :ARG1-of (t2 / true-01
                                        :location (i / it))))))
          :ARG2 (w / we)
          :time (e / ever))
      """
    orig_g = noop_penman_decode(amr)
    linearized = penman_to_linearized(amr)
    # penman_str = linearized_to_penman(linearized)
    print(linearized)




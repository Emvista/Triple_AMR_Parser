from AMR_utils import noop_penman_decode, to_single_line_amr
from AMRSimplifier import AMRSimplifier
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from penman.models.amr import model as amr_model
from penman.models.noop import model as noop_model

from preprocess.preprocess_utils import ne_recategorize, add_space_between_quotes

def main(input_file, output_file, no_inverse_role=False):
    decoding_model = noop_model
    if no_inverse_role:
        decoding_model = amr_model


    with open(input_file, "r") as f, open(output_file, "w") as f2:
        amrs = f.read().strip().split("\n\n")

        for i, amr in tqdm(enumerate(amrs)):
            amr = to_single_line_amr(amr)
            amr = ne_recategorize(amr)
            g= noop_penman_decode(amr, decoding_model)
            simplifier = AMRSimplifier(g)
            simple_amr = simplifier.simplify()
            space_added_between_quotes = add_space_between_quotes(simple_amr)
            f2.write(space_added_between_quotes + "\n")


if __name__ == '__main__':
    argparse = ArgumentParser()
    argparse.add_argument("--input_path", type=str, help="Path to the pm file")
    argparse.add_argument("--output_path", type=str, default=None, help="Path to the output file")
    argparse.add_argument("--no_inverse_role", action="store_true", help="no inverse roles in the AMR simplification")

    args = argparse.parse_args()

    assert Path(args.input_path).suffix == ".pm", "Input file must be a .pm file"

    if args.output_path is None:
        if args.no_inverse_role:
            args.output_path = args.input_path.replace(".pm", ".wo_var.wo_invrole.tri")
        else:
            args.output_path = args.input_path.replace(".pm", ".wo_var.tri")

    main(args.input_path, args.output_path, args.no_inverse_role)

    ## single amr test
#     amr = """
# (o / obligate-01 :ARG1 (w / we) :ARG2 (p / prevent-01 :ARG0 w :ARG1 (p2 / person :ARG0-of (r / run-02) :mod (d / dog) :ARG0-of (b / betray-01)) :ARG2 (c3 / collude-01 :ARG0 p2 :ARG1 (i / imperialism :mod (w4 / world-region :wiki "Western_world" :name (n3 / name :op1 "West"))))) :ARG1-of (c4 / cause-01 :ARG0 (c2 / chaos :location (a2 / and :op1 (w2 / world-region :wiki "East_Africa" :name (n / name :op1 "East" :op2 "Africa")) :op2 (w3 / world-region :wiki "North_Africa" :name (n2 / name :op1 "North" :op2 "Africa"))))) :mod (a / also))
# """
#     amr = to_single_line_amr(amr)
#     amr = ner_recategorize(amr)
#     g = noop_penman_decode(amr, None)
#     simplifier = AMRSimplifier(g)
#     simple_amr = simplifier.simplify()
#     print(simple_amr)
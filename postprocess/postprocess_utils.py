"""
This script is inspired by https://github.com/RikVN/AMR.git
"""
from typing import List, Tuple, Union, Dict
import logging
import re
from AMR_utils import get_default_amr

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
def concat_triples_to_str(triples) -> str:
    linearized_amr = ""
    for i, triple in enumerate(triples):
        src, rel, tgt = triple
        triple_str = " ".join([src, rel.lstrip(":"), tgt])
        linearized_amr += triple_str

        if i != len(triples) - 1:
            linearized_amr += " | "

    return linearized_amr


def remove_space_between_quotes(txt: str) -> str:
    new_txt = re.sub(r'\"\s*([^\"\s]+?)\s*\"', r'"\1"', txt)
    return new_txt

def is_triple(triple: str) -> bool:
    # there are cases where there are spaces within the quotes
    # e.g. s value " Auf Wiedersehen "
    if len(split_empty_space(triple)) == 3:
        return True
    return False

def fix_error_triple(triple: str) -> str:
    # to fix cases such as 'a3 instance and 339.12'
    triple_li = split_empty_space(triple)
    if len(triple_li) > 3:
        return " ".join(triple_li[:3]) # take the first 3 elements of the triple

    # to fix cases such as 'd instancedimension-01' or overlook-01 ARG1dimension
    else:
        concat_chars = " ".join(triple_li)
        new_split = re.split(r'(ARG\d|instance)', concat_chars)
        return " ".join(new_split[:3])

def split_empty_space(triple: str) -> List[str]:
    # split by space but keep the quoted strings together
    triples = re.findall(r'\"[^\"]+\"|\S+', triple)
    return triples

def split_str_triple(linearized_amr:str) -> List[Tuple[str]]:
    # return a list of triples by splitting the linearized amr string by "|"
    triples_str = linearized_amr.split("|")
    triple_list = []

    for triple_str in triples_str:
        triple_str = triple_str.strip()
        if not is_triple(triple_str):
            triple_str = fix_error_triple(triple_str)
            # try to fix it by taking the first 3 elements
            # if not fixed, ignore the triple and continue
            if not is_triple(triple_str):
                logging.info(f"Invalid triple: {triple_str}")
                continue
        src, rel, tgt = split_empty_space(triple_str)
        triple_list.append((src, f":{rel}", tgt))

    return triple_list


def replace_quotes(triples_str: str) -> Tuple[str, Dict] :
    pattern = r'"([^"]+)"' # match everything inside the parenthesis
    matches = re.findall(pattern, triples_str)

    if matches:
        matches = {i: f'"{match}"' for i, match in enumerate(matches)}

        for i, match in matches.items():
            triples_str = triples_str.replace(match, f"__masked__{i}")

        return triples_str, matches

    return triples_str, {}

def restore_quotes(t: str, quotes: Dict) -> str:
    if t.startswith("__masked__"):
        idx = t.split("__masked__")[1]
        return quotes[int(idx)] # restore the original string
    return t

def amr_str_to_list(triples_str: str) -> List[Tuple[str]]:
    processed_str, quotes = replace_quotes(triples_str)
    tuple_list = []
    pattern = r'\(([^()]+)\)' # match everything inside the parenthesis
    matches = re.findall(pattern, processed_str)

    for match in matches:
        src, rel, tgt = match.split(',')
        src = restore_quotes(src.strip(), quotes)
        tgt = restore_quotes(tgt.strip(), quotes)
        tuple_list.append((src, rel.strip(), tgt))

    return tuple_list

def get_initial_letter(instance: str) -> str:
    instance = instance.strip().lower()
    if instance.startswith('"'): # wiki instance
        return instance[1]
    else:
        return instance[0]


def name_new_variable(node: str, counter: dict = None) -> (str, dict):

    if counter is None: counter = {}

    first_letter = get_initial_letter(node)
    if first_letter in counter:
        counter[first_letter] += 1
        var = first_letter + str(counter[first_letter])
    else:
        counter[first_letter] = 1
        var = first_letter

    return var


# def undo_var_restore(penman_str):
#     added_vars = re.compile(r'(v+\d[^\s]+?\s/\s)[^\s]+?/')
#
#     for match in added_vars.finditer(penman_str):
#         var = match.group(1)
#         penman_str = penman_str.replace(var, '')
#
#     return penman_str

def add_space(penman_str):
    penman_str = re.sub(r'\(([^\s|(])', r' ( \1', penman_str)   # add space after (
    penman_str = re.sub(r'([^\s|)])\)', r'\1 ) ', penman_str)   # add space before )
    penman_str = re.sub(r'/([^\s])', r' / \1', penman_str)      # add space after /
    penman_str = re.sub(r'\) \)', r'))', penman_str)            # remove space between )
    return penman_str

if __name__ == '__main__':
    # prediction = '( and, :op1, possible-01 ) ( possible-01, :polarity, - ) ( possible-01, :ARG1, save-01 ) ( save-01, :ARG1, nation ) ( save-01, :ARG1, person ) ( person, :name, name ) ( name, :op1, "Mao" ) ( name, :op2, "Zedong" )'
    # main(prediction)
    # amr = '( possible-01, :ARG1, leave-12 ) ( leave-12, :ARG0, this ) ( leave-12, :ARG1, ship ) ( ship, :part-of, class ) ( class, :name, name ) ( name, :op1, "LHD" ) ( name, :op2, 1 ) ( name, :op3, "Wasp" ) ( ship, :quant, only ) ( only, :op1, 8 ) ( leave-12, :time, until ) ( until, :op1, have-condition-91 ) ( have-condition-91, :ARG2, build-01 ) ( build-01, :ARG1, ship ) ( ship, :part-of, class ) ( class, :name, name ) ( name, :op1, "LHA(R)" ) ( possible-01, :mod, eventual )'
    # processed = replace_quotes(amr)
    # print(processed)

    # tri = "d instancedimension-01"
    # print(fix_error_triple(tri))
    from settings import PROJECT_DIR
    from trainer import TripleSeq2seqTrainer
    from transformers import MBartForConditionalGeneration, Seq2SeqTrainingArguments
    import argparse
    from pathlib import Path

    args = argparse.ArgumentParser()
    args.add_argument("--prediction_file", type=Path, required=True)
    args.add_argument("--without_variable", action="store_true")
    args = args.parse_args()

    prediction_file = PROJECT_DIR / args.prediction_file
    output_dir = prediction_file.parent.parent.parent.parent

    model = MBartForConditionalGeneration.from_pretrained("./mbart-large-50")
    training_args = Seq2SeqTrainingArguments(output_dir=output_dir)
    trainer_seq2seq = TripleSeq2seqTrainer.TripleSeq2seqTrainer(model=model, args=training_args)
    trainer_seq2seq.no_wikify = True
    with open(prediction_file, 'r') as f:
        data = f.read().split('\n')
        if args.without_variable:
            trainer_seq2seq.postprocess_predictions_vnd(metric_key_prefix="eval", predictions=data)
            print("++ Done ++ ")
        else:
            trainer_seq2seq.postprocess_prediction_penman(metric_key_prefix="eval", predictions=data)




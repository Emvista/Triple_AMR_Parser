from typing import List
import re
import argparse
from pathlib import Path
'''
This script post process the penman linearization format (reordered variables + recategorized named entities) 
to undo recategorized named entities using regex 
'''


class NamedEntPostprocessor:
    def __init__(self, penman_str:str, with_variables=True):
        self.penman_str = penman_str
        self.vars = self.get_variable_list(penman_str)
        self.with_variables=with_variables

    def postprocess_ne_match(self, match=True) -> str:
        matched_string = match.group(1)
        words_in_match = matched_string.split("_")

        recategorized_ne = ""
        for i, word in enumerate(words_in_match):
            recategorized_ne += ':op' + str(i+1) + f' "{word.strip()}" '

        ne_var_name = "n" + str(self.n_var_index())
        self.vars.append(ne_var_name)  # add the new variable to the list of variables

        if self.with_variables:
            return f':name ({ne_var_name} / name {recategorized_ne})'
        else:
            return f':name (name {recategorized_ne})'


    def undo_ne_recategorize(self) -> str:
        recat_named_entities = re.compile(r':name\s*\"(.+?)\"\s*')
        # named_entities_candidate = recat_named_entities.findall(penman_str)
        penman_str = recat_named_entities.sub(self.postprocess_ne_match, self.penman_str)
        return penman_str


    def n_var_index(self):
        # Extract indices of variables starting with 'n' and convert them to integers
        indices = [
            int(var[1:]) if var[1:].isdigit() else 1
            for var in self.vars if var.startswith('n')
        ]

        # Return the next index as a string or an empty string if no valid indices are found
        if indices:
            return str(max(indices) + 1)
        else:
            return ""

    @staticmethod
    def get_variable_list(penman_str) -> List[str]:
        variable_name_pattern = re.compile(r'\(\s*([a-z]\d*)\s*\/')
        variable_list = variable_name_pattern.findall(penman_str)
        return variable_list


if __name__ == '__main__':

    from AMR.wikify_file import wikify_file
    import subprocess
    from settings import PROJECT_DIR
    from utils import new_cd

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, help="input path to penman file")
    parser.add_argument("--without_variables", action="store_true", help="if the file contains variables or not, if not restore variables")
    parser.add_argument("--restore_wiki", action="store_true", help="restore wiki links")
    args = parser.parse_args()

    sent_file = args.input_path.parent / (args.input_path.name.split(".")[0] + ".en")
    assert sent_file.exists(), f"File {sent_file} does not exist"

    out_path = args.input_path.parent / "restored" / (args.input_path.name + ".wiki.restore.amr")
    restored = []

    with open(args.input_path, "r") as in_f, open(out_path, "w") as out_f:
         penman_str = in_f.read().strip()
         pms = penman_str.split("\n\n")

         for pm in pms:
             processor = NamedEntPostprocessor(pm, with_variables=not args.without_variables)
             restored.append(processor.undo_ne_recategorize())
         out_f.write("\n\n".join(restored))

    if args.without_variables:
        with new_cd(PROJECT_DIR / "AMR"):
            subprocess.run(["python", "postprocess_AMRs.py", "-f", out_path, "-s", sent_file, "--no_wiki"])
            out_path = out_path.parent / (out_path.name.split(".")[0] + ".pm.vnd.wiki.restore.amr.restore.final")

    if args.restore_wiki:
        wikify_file(out_path.as_posix(), sent_file.as_posix())
        out_path = out_path.parent / (out_path.name + ".wiki")

    subprocess.run(["python", "AMR/reformat_single_amrs.py", "-f", out_path, "-e", ".pm"])

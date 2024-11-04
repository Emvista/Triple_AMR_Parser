from AMR import reformat_single_amrs
import argparse
from pathlib import Path
import os
import re

from AMR_utils import get_default_amr
from postprocess.postprocess_penman import NamedEntPostprocessor
from postprocess.postprocess_utils import add_space
from postprocess.postprocess_tri import pass_sanity_check
from utils import new_cd
from AMR.restoreAMR import restore_amr as restore_call

def restore_amr(in_file, out_file, coref_type, force):
    '''Function that restores variables in output AMR
       Also restores coreference for index/absolute paths methods'''
    with new_cd(PROJECT_DIR / "AMR"):
        if not os.path.isfile(out_file) or force:
            restore_call = 'python3 restoreAMR/restore_amr.py -f {0} -o {1} -c {2}'.format(in_file, out_file, coref_type)
            os.system(restore_call)
    return out_file


def get_restored_and_formatted(input_file):

    with open(input_file, 'r') as f:
        single_line_amrs = f.readlines()

    formatted_amrs = reformat_single_amrs.reformat_amr(input_file)
    formatted_amrs = [amr for amr in formatted_amrs if amr.strip() != '']

    return single_line_amrs, formatted_amrs


def set_smatch_args(prediction_file_path:Path, reference_file_path:Path):
    parser = argparse.ArgumentParser(description="Smatch calculator")
    parser.add_argument(
        '-f',
        nargs=2,
        required=True,
        type=argparse.FileType('r'),
        help=('Two files containing AMR pairs. '
              'AMRs in each file are separated by a single blank line'))
    parser.add_argument(
        '-r',
        type=int,
        default=4,
        help='Restart number (Default:4)')
    parser.add_argument(
        '--significant',
        type=int,
        default=2,
        help='significant digits to output (default: 2)')
    parser.add_argument(
        '-v',
        action='store_true',
        help='Verbose output (Default:false)')
    parser.add_argument(
        '--vv',
        action='store_true',
        help='Very Verbose output (Default:false)')
    parser.add_argument(
        '--ms',
        action='store_true',
        default=False,
        help=('Output multiple scores (one AMR pair a score) '
              'instead of a single document-level smatch score '
              '(Default: false)'))
    parser.add_argument(
        '--pr',
        action='store_true',
        default=False,
        help=('Output precision and recall as well as the f-score. '
              'Default: false'))
    parser.add_argument(
        '--justinstance',
        action='store_true',
        default=False,
        help="just pay attention to matching instances")
    parser.add_argument(
        '--justattribute',
        action='store_true',
        default=False,
        help="just pay attention to matching attributes")
    parser.add_argument(
        '--justrelation',
        action='store_true',
        default=False,
        help="just pay attention to matching relations")

    # set the arguments with the given file paths
    args = parser.parse_args(['-f', prediction_file_path.as_posix(), reference_file_path.as_posix(), '-r', '5', '--significant', '3'])
    return args


def add_unmatched_parenthesis(line):
    # heavily borrowed from vannoord restore_amr.py
    # but skips replace var part

    # Make sure parentheses match
    open_count = 0
    close_count = 0
    for i, c in enumerate(line):
        if c == '(':
            open_count += 1
        elif c == ')':
            close_count += 1
        if open_count == close_count and open_count > 0:
            line = line[:i].strip()
            break

    old_line = line
    while True:
        open_count = len(re.findall(r'\(', line))
        close_count = len(re.findall(r'\)', line))
        if open_count > close_count:
            line += ')' * (open_count - close_count)
        elif close_count > open_count:
            for i in range(close_count - open_count):
                line = line.rstrip(')')
                line = line.rstrip(' ')

        if old_line == line:
            break
        old_line = line

    return line

def fix_easy_errors(line):
    # heavily borrowed from vannoord restore_amr.py
    # fix some easy errors such as unmatched parenthesis
    line = restore_call.unbracket.sub(r'\1', line, re.U)
    # line = restore_amr.dangling_edges.sub('', line, re.U) # skip this since this removes necessary variables in our case
    line = restore_call.missing_edges.sub(r'\1 :ARG2 (', line, re.U)
    line = restore_call.missing_variable.sub(r'vvvx / \1 ', line, re.U)
    line = restore_call.missing_quotes.sub(r'\1"', line, re.U)
    line = restore_call.misplaced_colon.sub(r'', line, re.U)
    line = restore_call.missing_concept_and_variable.sub(r'd / dummy ', line, re.U)
    line = restore_call.dangling_quotes.sub(r'\1', line, re.U)

    return line

def undo_ne_recat(line, with_variables=True) -> str:

    line = line.strip()
    # line = undo_var_restore(line)
    line = add_space(line)
    line = NamedEntPostprocessor(line, with_variables).undo_ne_recategorize()
    # sanity check for the restore AMR
    return line



def main(input_file):
    processed = []
    with open(input_file, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]

    for line in lines:
        line = add_unmatched_parenthesis(line)
        line = fix_easy_errors(line)
        line = undo_ne_recat(line)

        if not pass_sanity_check(line):
            breakpoint()
            line = get_default_amr("pm_1line")

        processed.append(line)

    output_file = input_file.parent / (input_file.name + '.postprocessed')
    with open(output_file, 'w') as f:
        print(f'Writing postprocessed AMRs to {output_file}')
        f.write('\n'.join(processed))


def temp_clean_wiki(input_file):
    with open(input_file, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]

    clean = []
    for line in lines:
        if not pass_sanity_check(line):
            print('Sanity check failed')
            print(line)
            print('Replacing with default AMR')
            line = get_default_amr("pm_1line")
        clean.append(line)

    with open(input_file, 'w') as f:
        f.write('\n'.join(clean))

if __name__ == '__main__':
    from settings import PROJECT_DIR
    import argparse

    parser = argparse.ArgumentParser(description="Postprocess AMR")
    parser.add_argument('--input_file', type=Path, help='Input file to postprocess')
    args = parser.parse_args()

    # main(args.input_file)
    temp_clean_wiki(args.input_file)

    # line = "(m / multi-sentence :snt1 (m2 / many :ARG0-of (s / sense-01 :ARG1 (u / urgency) :time (w / watch-01 :ARG0 m2 :ARG1 (t3 / thing :manner-of (d / develop-02 :ARG0 (t / thing))) :manner (q / quiet-04 :ARG1 m2)))) :snt2 (d2 / dragon :domain (y / you) :ARG0-of (c / coil-01)) :snt3 (t2 / tiger :domain (y2 / you) :ARG0-of (c2 / crouch-01)) :snt4 (a / admire-01 :ARG0 (i / i) :ARG1 (p / patriot :poss-of (m3 / mind :mod (n / noble)"
    # line = add_unmatched_parenthesis(line)
    # line = fix_easy_errors(line)
    # line = undo_ne_recat(line)
    # print(line)
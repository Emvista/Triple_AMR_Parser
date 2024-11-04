#!/bin/bash

# script to linearize all AMR data

# change this to the path of your anaconda3 installation
source $HOME/anaconda3/bin/activate
conda activate triple_amr

project_dir="${HOME}/projects/Triple_AMR_Parser"
data_dir="${project_dir}/data/amr/en/train"     # replace "train" with "eval" or "test" for other datasets
lang="en"                                       # change the langage to "fr" for French

# linearize amr in triples while retaining variables and inverse roles
python ${project_dir}/preprocess/linearize_amr_with_variables.py --input_path ${data_dir}/$lang-amr.pm
# linearize amr in triples while retaining variables
python ${project_dir}/preprocess/linearize_amr_with_variables.py --input_path ${data_dir}/$lang-amr.pm --no_inverse_role
# linearize amr in triples without variables and inverse roles
python ${project_dir}/preprocess/linearize_amr_without_variables.py  --input_path ${data_dir}/$lang-amr.pm
# linearize amr in triples without variables
python ${project_dir}/preprocess/linearize_amr_without_variables.py  --input_path ${data_dir}/$lang-amr.pm --no_inverse_role

# linearize amr in penman while retaining variables
python ${project_dir}/preprocess/linearize_penman_with_variables.py --input_path ${data_dir}/$lang-amr.pm

# linearize amr in penman without variables
python ${project_dir}/AMR/var_free_amrs.py -f ${data_dir}/$lang-amr.pm
# and then add empty lines to var_free_amrs (requires one more preprocess step)
python ${project_dir}/preprocess/preprocess_utils.py --input_file ${data_dir}/$lang-amr.pm.tf

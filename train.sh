#!/bin/bash

# script to linearize all AMR data
# check line 10 and 11 to specify data split and language

# change this to the path of your anaconda3 installation
source $HOME/anaconda3/bin/activate
conda activate triple_amr

# train the model while specifying task type and linearization format
# see train_amr_parser.py for more details

# example 1. Train the parser with AMR data in triples (--linearization_types amr) without variables and inverse roles
python train_amr_parser.py --task_name en-amr --linearization_types amr --generate_format amr --without_variables --without_invrole

## example 2. Train the parser with AMR data in triples (--linearization_types amr) without variables
#python train_amr_parser.py --task_name en-amr --linearization_types amr --generate_format amr --without_variables
## example 3. Train the parser with AMR data in triples (--linearization_types amr) with variables and inverse roles
#python train_amr_parser.py --task_name en-amr --linearization_types amr --generate_format amr
## example 4. Train the parser with AMR data in penman with variables
#python train_amr_parser.py --task_name en-amr --linearization_types penman --generate_format penman
## example 5. Train the parser with AMR data in penman without variables
#python train_amr_parser.py --task_name en-amr --linearization_types vnd --generate_format vnd
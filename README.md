This repository contains code implementation to train an AMR parser using triple linearization. 

## Pre-requisites
Clone the repository and install the required packages using the following command:
```
git clone https://github.com/RikVN/AMR.git
cd AMR
git clone https://github.com/snowblink14/smatch
cd ../
pip install -r requirements.txt
``` 
## Add followings packages to python path
```
export PYTHONPATH=$PYTHONPATH:/[project_dir]/AMR
export PYTHONPATH=$PYTHONPATH:/[project_dir]/AMR/smatch
```

## Download the data 
- AMR 3.0 
  - AMR3.0 is available on [LDC](https://catalog.ldc.upenn.edu/LDC2020T02). You need a proper license to download the data.

  - Structure of the data
    - Once the data is downloaded, they should be structured in the following way:
    ```Add tree
      - data
          - AMR
            - en
              - train
                - en-amr.pm      # Structured AMR graph, delimited with a blank line
                - en-amr.en   # Sentences corresponding to the AMR graphs
              - eval
                - en-amr.pm
                - en-amr.en
            ...
            - fr 
              - train
                - fr-amr.pm
                - fr-amr.fr
            ... 
    ```
## Preprocess + linearize data 
Preprocess the data using the following command:
```
cd preprocess
chmod +x preprocess.sh
./preprocess.sh
```
See the `preprocess.sh` file for more details to change the data path.


## Train the model 
Train the model using the following command:
```
./train.sh
```
See the `train.sh` file for more details to change training parameters. 



# SS-CMPE - High Tree-Width Markov Networks and Tractable Probabilistic Circuits

### Dataset Preparation

1. Begin by extracting the datasets from the datasets.zip file. This file contains the required datasets.
2. Convert the datasets to the required npz format by running the generate_npz.py script. This script will create a folder named datasets_npz with the converted dataset files. Use these files for running the following scripts.
3. For each method, please provide the location of the npz files using the --data-path argument. 

### Customization and Hyperparameters

Each Python script provides command-line arguments that allow customization of various hyperparameters. Please refer to the specific Python file for the list of available hyperparameters and their descriptions. You can modify these arguments to experiment with different settings and configurations according to your requirements.

## Run scripts to get results of ILP (1_ilp)

We can run the following script to extract the evaluation metrics for optimal solutions extracted from solvers. We use a single script to extract the optimal solutions and objective value for each dataset.

```python
python nn_ilp.py
```

## Run Scripts to get outputs of Supervised method (2_naive)

For this method we need to get outputs from the ILP method first which use SCIP to get the optimal solutions.

```python
current_string='dna-80-60'
# MSE Loss
python naive.py --loss=MSE --save-model --dataset=${current_string}

# MAE Loss
python naive.py --loss=MAE --save-model --dataset=${current_string}
```

## Run Scripts to get outputs of Supervised method with penalty (3_naive+penalty)

For this method we need to get outputs from the ILP method first which use SCIP to get the optimal solutions. This method adds penalty to the supervised method. 

```python
current_string='dna-80-60'
# MSE Loss
python naive_penalty.py --loss=MSE --save-model --dataset=${current_string}

# MAE Loss
python naive_penalty.py --loss=MAE --save-model --dataset=${current_string}
```

## Run Scripts to get outputs of Self-Supervised method with Penalty (4_ssl+penalty)

```python

current_string='dna-80-60'
python nn_ssl_penalty.py --save-model --dataset=${current_string}

```

## Run Scripts to get outputs of PDL (5_pdl)

```python
current_string='dna-80-60'
python nn_ssl_pdl.py --rho=10 --save-model --dataset=${current_string}

```

## Run Scripts to get outputs of SS-CMPE (ssl_cmpe)

This is the method which uses the loss given in equation 4 of the paper.


```python
current_string='dna-80-60'
python nn_ssl_ours.py --save-model --dataset=${current_string}

```

## Run Scripts to get outputs of SS-CMPE-Continous (ssl_cmpe_continous_loss)

This is the method which uses the loss given in equation 6 of the paper.


```python
current_string='dna-80-60'
python nn_ssl_ours_cont.py --save-model --dataset=${current_string}

```

Please ensure that you have navigated to the corresponding folder for each method before running the commands mentioned above.
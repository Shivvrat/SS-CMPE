# SS-CMPE

## Application - Adversarial Example Generation

### Train the classifier

Please use the train_nn_one_vs_all.py script to train Neural Network for each class.

```python
python train_nn_one_vs_all.py --class-index=$i --save-model --dataset=MNIST

```

After the classifier is trained we will use it in our experiments. 

Please go into the corresponding folder for each method and then run the commands given below

### Run scripts to get outputs of ILP (1_ilp)

We can run the following script to get true optimal solutions and objective values using SCIP solver

```python
# For class index = i and model that converts the class label from i to not i
python nn_ilp.py --class-index=$i --dataset=MNIST --model=classifier_location --use_nn
# For class index = i and model that converts the class label from not i to i
python nn_ilp.py --class-index=$i --not-class --dataset=MNIST --model=classifier_location --use_nn
```

### Run Scripts to get outputs of Supervised method (2_naive)

For this method we need to get outputs from the ILP method first which use SCIP to get the optimal solutions.

```python
# For class index = i and model that converts the class label from i to not i - with MSE Loss
python naive_sl.py --class-index=$i --loss=MSE --save-model --dataset=MNIST --model=trained_model_location --use_nn
# For class index = i and model that converts the class label from not i to i - with MSE Loss
python naive_sl.py --class-index=$i --loss=MSE --not-class --save-model --dataset=MNIST --model=trained_model_location --use_nn
# For class index = i and model that converts the class label from i to not i - with MAE Loss
python naive_sl.py --class-index=$i --loss=MAE --save-model --dataset=MNIST --model=trained_model_location --use_nn
# For class index = i and model that converts the class label from not i to i - with MAE Loss
python naive_sl.py --class-index=$i --loss=MAE --not-class --save-model --dataset=MNIST --model=trained_model_location --use_nn

```

### Run Scripts to get outputs of Supervised method with penalty (3_naive+penalty)

For this method we need to get outputs from the ILP method first which use SCIP to get the optimal solutions. This method adds penalty to the supervised method. 

```python
# For class index = i and model that converts the class label from i to not i - with MSE Loss
python naive_penalty_sl.py --class-index=$i --loss=MSE --save-model --dataset=MNIST --model=trained_model_location --use_nn
# For class index = i and model that converts the class label from not i to i - with MSE Loss
python naive_penalty_sl.py --class-index=$i --loss=MSE --not-class --save-model --dataset=MNIST --model=trained_model_location --use_nn
# For class index = i and model that converts the class label from i to not i - with MAE Loss
python naive_penalty_sl.py --class-index=$i --loss=MAE --save-model --dataset=MNIST --model=trained_model_location --use_nn
# For class index = i and model that converts the class label from not i to i - with MAE Loss
python naive_penalty_sl.py --class-index=$i --loss=MAE --not-class --save-model --dataset=MNIST --model=trained_model_location --use_nn

```

### Run Scripts to get outputs of Self-Supervised method with Penalty (4_ssl+penalty)

```python

# For class index = i and model that converts the class label from i to not i 
python nn_ssl_penalty.py --class-index=$i --save-model --dataset=MNIST --model=trained_model_location --use_nn
# For class index = i and model that converts the class label from not i to i 
python nn_ssl_penalty.py --class-index=$i --not-class --save-model --dataset=MNIST --model=trained_model_location --use_nn

```

### Run Scripts to get outputs of PDL (5_pdl)

```python

# For class index = i and model that converts the class label from i to not i 
python nn_ssl_pdl.py --class-index=$i --save-model --dataset=MNIST --model=trained_model_location --use_nn

# For class index = i and model that converts the class label from not i to i 
python nn_ssl_pdl.py --class-index=$i --not-class --save-model --dataset=MNIST --model=trained_model_location --use_nn

```

### Run Scripts to get outputs of SS-CMPE (ssl_cmpe)

This is the method which uses the loss given in equation 4 of the paper.

```python
# For class index = i and model that converts the class label from i to not i 
python nn_ssl_ours.py --class-index=$i --save-model --dataset=MNIST --model=trained_model_location
# For class index = i and model that converts the class label from i to not i 
python nn_ssl_ours.py --not-class --class-index=$i --save-model --dataset=MNIST --model=trained_model_location
```

### Run Scripts to get outputs of SS-CMPE-Continous (ssl_cmpe_continous_loss)

This is the method which uses the loss given in equation 6 of the paper.

```python
# For class index = i and model that converts the class label from i to not i 
python nn_ssl_ours_continous.py --class-index=$i --save-model --dataset=MNIST --model=trained_model_location
# For class index = i and model that converts the class label from i to not i 
python nn_ssl_ours_continous.py --not-class --class-index=$i --save-model --dataset=MNIST --model=trained_model_location

```
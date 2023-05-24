# This file is used to generate the npz files for the High Tree-Width Markov Network and Tractable Probabilistic Circuits
import glob
import os
import csv
import numpy as np  

def read_functions(file_name):
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        bivariate_functions = []
        univariate_functions = []   
        for row in reader:
            row = row[0].split(' ')
            row = [np.float64(x) for x in row]
            if len(row) == 3:
                univariate_functions.append(row)                
            elif len(row) == 6:
                bivariate_functions.append(row)
            else:
                raise ValueError(f"Row length is not 3 or 6: {row}")
    bivariate_functions = np.array(bivariate_functions, dtype=np.float64)
    univariate_functions = np.array(univariate_functions, dtype=np.float64)   
    return bivariate_functions, univariate_functions

def get_folder_paths(root_folder):
    return glob.glob(f'{root_folder}/*/')

# Example usage
root_folder = 'datasets' # Add your root folder for the data here - It should have folders for each dataset you get after unzipping the datasets.zip file
folder_paths = get_folder_paths(root_folder)
print(f"Folders in '{root_folder}':")
print(folder_paths)
output_folder = "datasets_npz"
os.makedirs(output_folder, exist_ok=True)
train_ratio = 0.9
# Calculate the number of examples for the train set

# Accessing files within each folder
datasets = []
for folder_path in folder_paths:
    dataset_name = folder_path.split('/')[-2]
    # if "grids" not in dataset_name.lower():
    if "80-60" not in dataset_name.lower():
        continue
    print(f"Dataset name: {dataset_name}")
    X = np.loadtxt(f'{folder_path}/samples_x.txt', dtype=np.float64)
    # Calculate the number of examples for the train set
    train_size = int(len(X) * train_ratio)
    # Generate random indices for the train set
    train_indices = np.random.choice(range(len(X)), size=train_size, replace=False)
    # Generate indices for the test set by excluding the train indices
    test_indices = np.setdiff1d(range(len(X)), train_indices)

    # You can perform operations on the files within each folder here
    file_names = glob.glob(f'{folder_path}/*')
    # print(f"Files in folder '{folder_path}':")
    for file_name in file_names:
        if "samples_x.txt" in file_name:
            X_train = X[train_indices]
            X_test = X[test_indices]
            print(X.shape)
        elif "which_is_x.txt" in file_name:
            X_boolean = np.loadtxt(file_name)
            print(X_boolean.shape)
        elif "samples_optimal_y.txt" in file_name:
            samples_optimal_y = np.loadtxt(file_name)
            samples_optimal_y_train = samples_optimal_y[train_indices]
            samples_optimal_y_test = samples_optimal_y[test_indices]
            print(samples_optimal_y.shape)
        elif "samples_feasible_y.txt" in file_name:
            samples_feasible_y = np.loadtxt(file_name)
            samples_feasible_y_train = samples_feasible_y[train_indices]
            samples_feasible_y_test = samples_feasible_y[test_indices]
            print(samples_feasible_y.shape)
        elif "new_bounds.txt" in file_name:
            bounds = np.loadtxt(file_name, dtype=np.float64)
            lb_denom = bounds[:, 0]
            initial_ub_num = bounds[:, 1]
            lb_denom_train = lb_denom[train_indices]
            lb_denom_test = lb_denom[test_indices]
            initial_ub_num_train = initial_ub_num[train_indices]
            initial_ub_num_test = initial_ub_num[test_indices]
            print(lb_denom.shape)
            print(initial_ub_num.shape)
        elif "f.txt" in file_name:
            f_bivariate_functions, f_univariate_functions = read_functions(file_name)
            print("Shapes of f functions")
            print(f_bivariate_functions.shape)
            print(f_univariate_functions.shape)
        elif "g.txt" in file_name:
            print("Shapes of g functions")
            g_bivariate_functions, g_univariate_functions = read_functions(file_name)
            print(g_bivariate_functions.shape)
            print(g_univariate_functions.shape)        
    output_location = f"{output_folder}/{dataset_name}.npz"
    if not os.path.isfile(output_location):
        print("File does not exist. Running the command...")
        np.savez(output_location, X_train=X_train, X_test=X_test, X_boolean=X_boolean, samples_optimal_y_train=samples_optimal_y_train, samples_optimal_y_test=samples_optimal_y_test, samples_feasible_y_train=samples_feasible_y_train, samples_feasible_y_test=samples_feasible_y_test, lb_denom_train=lb_denom_train, lb_denom_test=lb_denom_test, initial_ub_num_train=initial_ub_num_train, initial_ub_num_test=initial_ub_num_test, f_bivariate_functions=f_bivariate_functions, f_univariate_functions=f_univariate_functions, g_bivariate_functions=g_bivariate_functions, g_univariate_functions=g_univariate_functions)
        if "80-60" in dataset_name:
            datasets.append(dataset_name)
    else:
        print("File already exists.")
        print("Please remove these lines if you want to regenrate the data")
print(datasets)
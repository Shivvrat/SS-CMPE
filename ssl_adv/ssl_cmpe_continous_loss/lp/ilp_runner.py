import multiprocessing
from multiprocessing import Pool

import numpy as np
import torch
import wandb
from loguru import logger
from sklearn.metrics import accuracy_score
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import cProfile, pstats, io
from pstats import SortKey

from lp.ilp_solver_class_one_vs_all import create_solver
from utils import create_directory, get_date_as_string
from torchvision import datasets, transforms

def get_2d_np_array(data):
    values = []
    for idx, xi in enumerate(data):
        array = np.array(xi).reshape((-1))
        if array.shape[0] == 0:
            continue
        values.append(array)
    out_data = np.vstack(values)
    return out_data


def run_ilp(test_batch_size, nn_as_dict_of_np_array,  args, loader, add_f_plus_g=False, feasible_region=True, debug=False):
    """
    This function runs ILP to get outputs for a neural network and saves the results in a directory.
    
    :param debug: A boolean flag indicating whether to run the function in debug mode or not
    :param test_batch_size: The batch size used during testing
    :param nn_as_dict_of_np_array: A dictionary containing the weights and biases of a neural network as
    numpy arrays
    :param args: It is a dictionary containing various arguments for the ILP solver. The specific
    arguments and their values are not shown in this code snippet
    :param loader: The data loader object that loads the input data and target labels in batches for
    testing the neural network
    :param date: The date parameter is a string representing the current date, which is used to create a
    directory for storing debug outputs
    :param save_name: The name to use when saving the ILP outputs to a file
    :param dataset: The name of the dataset being used (default is "test"), defaults to test (optional)
    :return: three variables: updated, initial, and missed_examples.
    """
    updated = []
    initial = []
    missed_examples = []
    num_pools = multiprocessing.cpu_count()
    if debug:
        num_pools = test_batch_size
    logger.info(f"We will use {num_pools} Pools")
    logger.info("Getting outputs for the NN using ILP")

    # num_examples = len(cnn_predictions)
    idx_debug = 0
    for batch_idx, data in enumerate(tqdm(loader)):
        # data, target = data
        data, target, num_min, num_max, num_average, updated_numerator, denom, idx = data
        if debug:
            pr = cProfile.Profile()
            pr.enable()
            if idx_debug == 2:
                break
            if idx_debug == 1:
                data = data[:1]
                target = target[:1]
            idx_debug += 1
        pool = Pool(processes=num_pools)
        num_examples = len(data)
        inputs = [[] for _ in range(num_examples)]
        data = data.cpu().detach().numpy()
        for index, each_prediction in enumerate(data):
            inputs[index] = (
                f"ILP model_{index}", nn_as_dict_of_np_array, data[index], target[index], add_f_plus_g, feasible_region, debug, args,
                index)
        ilp_outputs = pool.map(create_solver, inputs)
        pool.close()
        pool.join()
        if debug:
            pr.disable()
            s = io.StringIO()
            # sortby = SortKey.TOTAL
            ps = pstats.Stats(pr, stream=s)  # .sort_stats(sortby)
            # ps.print_stats()
            # print(s.getvalue())
        updated_inputs = [[] for each_example in range(num_examples)]
        initial_inputs = [[] for each_example in range(num_examples)]
        for each_index, each_updated_input in ilp_outputs:
            if each_updated_input is not None:
                input_to_model = data[each_index]
                initial_inputs[each_index] = input_to_model.reshape(-1).tolist()
                updated_inputs[each_index] = each_updated_input.reshape(-1).tolist()
            else:
                missed_examples.append(each_index + batch_idx * test_batch_size)

        updated.extend(updated_inputs)
        initial.extend(initial_inputs)
    updated = get_2d_np_array(updated)
    initial = get_2d_np_array(initial)
    return updated, initial, missed_examples
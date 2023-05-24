import sys

import numpy as np
import pyscipopt
from loguru import logger
from numpy import argmax
from pyscipopt import Model, quicksum, quickprod


class CheckILPOutput():
    def __init__(self, name, args, nn, input_to_nn, debug=False):
        self.weights = None
        self.bias = None
        self.nn = nn
        self.variables = {}
        self.objective_values = []
        self.solver = Model(name)
        # Pre solve does not allow symmetry constraints - turn it off
        self.solver.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
        # Set the max time for solving
        if not debug:
            self.solver.setParam('limits/time', 120)
        else:
            self.solver.setParam('limits/time', 60)

        self.num_layers = len(nn) // 2
        # Layer number starts at zero - Values at zero is that of input
        self.input_as_vector = input_to_nn
        assert self.input_as_vector.shape[0] == args.input_size, "Input vector to ILP is not correct"
        for idx, val in enumerate(self.input_as_vector):
            self.variables[0, idx, "x_old"] = float(val)
            self.variables[0, idx, "x"] = self.solver.addVar(name=f"modified_input_idx_{idx}_x",
                                                             lb=0, ub=1, vtype="C")
            # Add the flip constraints for this index
            if args.func_f == "min_distance":
                self.add_constraints_for_f_min_distance(idx)
            
        for idx, val in enumerate(self.input_as_vector):
            if args.func_f == "min_distance_plus_grid":
                self.add_constraints_for_f_min_distance_plus_grid(idx)
                

    def add_constraints_for_f_min_distance(self, idx):
        """_summary_ This function is used to add constraints 

        Args:
            idx (_type_): _description_
        """
        negative_weight = 1e-2
        positive_weight = 2.0
        f_weights_X_Y = np.array([negative_weight, positive_weight , positive_weight, negative_weight])
        f_sum_list = []
        old_value = self.variables[0, idx, "x_old"]
        new_value = self.variables[0, idx, "x"]
        f_sum_list.append((1-old_value) * (1-new_value) * f_weights_X_Y[0] + (1-old_value) * new_value * f_weights_X_Y[1] + old_value * (1-new_value) * f_weights_X_Y[2] + old_value * new_value * f_weights_X_Y[3])
        self.objective_values.extend(f_sum_list)

    def add_constraints_for_f_min_distance_plus_grid(self, idx):
        # Add the constraints for min distance
        self.add_constraints_for_f_min_distance(idx)
        # Add the constraints for grid
        negative_weight = 1e-2
        positive_weight = 2.0
        f_weights_grid = np.array([negative_weight, positive_weight , positive_weight, negative_weight])

        row_index = idx // 28
        col_index = idx % 28
        f_grid_list = []
        # Calculate indices of neighboring cells
        top_index = (row_index - 1) * 28 + col_index
        if top_index < 0:
            top_index = None  # Top cell is outside the grid
        bottom_index = (row_index + 1) * 28 + col_index
        if bottom_index > 783:
            bottom_index = None  # Bottom cell is outside the grid
        left_index = row_index * 28 + (col_index - 1)
        if left_index < 0:
            left_index = None  # Left cell is outside the grid
        right_index = row_index * 28 + (col_index + 1)
        if right_index > 783:
            right_index = None  # Right cell is outside the grid
        this_value = self.variables[0, idx, "x"]
        if top_index is not None:
            top_value = self.variables[0, top_index, "x"]
            f_grid_list.append((1-this_value) * (1-top_value) * f_weights_grid[0] + (1-this_value) * top_value * f_weights_grid[1] + this_value * (1-top_value) * f_weights_grid[2] + this_value * top_value * f_weights_grid[3])
        if bottom_index is not None:
            bottom_value = self.variables[0, bottom_index, "x"]
            f_grid_list.append((1-this_value) * (1-bottom_value) * f_weights_grid[0] + (1-this_value) * bottom_value * f_weights_grid[1] + this_value * (1-bottom_value) * f_weights_grid[2] + this_value * bottom_value * f_weights_grid[3])
        if left_index is not None:
            left_value = self.variables[0, left_index, "x"]
            f_grid_list.append((1-this_value) * (1-left_value) * f_weights_grid[0] + (1-this_value) * left_value * f_weights_grid[1] + this_value * (1-left_value) * f_weights_grid[2] + this_value * left_value * f_weights_grid[3])
        if right_index is not None:
            right_value = self.variables[0, right_index, "x"]
            f_grid_list.append((1-this_value) * (1-right_value) * f_weights_grid[0] + (1-this_value) * right_value * f_weights_grid[1] + this_value * (1-right_value) * f_weights_grid[2] + this_value * right_value * f_weights_grid[3])
        self.objective_values.extend(f_grid_list)

    # @njit()
    def add_constraints_one_layer(self, layer_number, last_layer=False):
        # the size of weight vector is [output, input]
        for idx_o in range(self.weights.shape[0]):
            # The variables with key z are the values before activation function
            # Lower bound for z needs to be none since it can be negative as well!
            # The variables with key 0 are the values after activation function is applied
            if not last_layer:
                self.variables[layer_number, idx_o, "x"] = self.solver.addVar(name=f"op_{layer_number}_idx_{idx_o}_x",
                                                                              lb=0)
            else:
                # For last layer we don't need a lower bound.
                self.variables[layer_number, idx_o, "x"] = self.solver.addVar(name=f"op_{layer_number}_idx_{idx_o}_x",
                                                                              lb=None)
            # if last_layer:
            self.variables[layer_number, idx_o, "s"] = self.solver.addVar(name=f"op_{layer_number}_idx_{idx_o}_s",
                                                                          lb=0)
            # self.variables[layer_number, idx_o, "z"] = self.model.addVar(name=f"op_{layer_number}_idx_{idx_o}_s",
            #                                                              lb=0, ub=1, vtype="I",)
            # Add the values of all output from ReLu as object function
            if not last_layer:
                self.objective_values.append(self.variables[layer_number, idx_o, "x"])
                # No need for z since SCIP takes care of the indicator constraints
                # self.objective_values.append(self.variables[layer_number, idx_o, "z"])
            this_sum = []
            # Calculate output for layer "layer_number" and node index "idx_o"
            for idx_i in range(self.weights.shape[1]):
                # Values of w*x
                this_sum.append(self.variables[layer_number - 1, idx_i, "x"] * self.weights[idx_o, idx_i])
            # Values of b
            this_sum.append(self.bias[idx_o])
            if not last_layer:
                # Constraint -> x-s = sum_i(w_i*x_i)
                self.solver.addCons(quicksum(this_sum) == (
                        self.variables[layer_number, idx_o, "x"] - self.variables[layer_number, idx_o, "s"]), )
                # Only one of x or s can be > 0
                self.solver.addConsCardinality(
                        [self.variables[layer_number, idx_o, "x"], self.variables[layer_number, idx_o, "s"]], 1)
            else:
                # For final layer we just need the sum, since we apply softmax
                self.solver.addCons(quicksum(this_sum) == self.variables[layer_number, idx_o, "x"])

    def add_constraint_all_layers(self, ):
        # num_layers = self.num_layers
        # self.weights = self.nn[f"l1.weight"]
        # self.bias = self.nn[f"l1.bias"]
        # self.add_constraints_one_layer(0 + 1, True)
        
        for each_layer in range(0, self.num_layers):
            last_layer = False
            self.weights = self.nn[f"l{each_layer * 2}.weight"]
            self.bias = self.nn[f"l{each_layer * 2}.bias"]
            if each_layer == self.num_layers - 1:
                last_layer = True
            self.add_constraints_one_layer(each_layer + 1, last_layer)


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def create_solver(args):
    """
    Create a solver for the ILP task
    :param name:
    :param nn:
    :param input_to_nn:
    :return:
    """
    name, nn, input_to_nn, target, debug, main_args, index = args
    # Define a dictionary to hold the variables in the ILP problem
    input_to_nn = input_to_nn.reshape(-1)
    ilp_model = CheckILPOutput("ILP_for_nn", main_args, nn, input_to_nn, debug)
    if debug:
        ilp_model.solver.hideOutput(False)
    else:
        ilp_model.solver.hideOutput(True)

    ilp_model.add_constraint_all_layers()
    objective = quicksum(ilp_model.objective_values)
    if target == 1:
        ilp_model.solver.addCons(ilp_model.variables[ilp_model.num_layers, 0, "x"] <= - 1e-10)
    else:
        ilp_model.solver.addCons(- ilp_model.variables[ilp_model.num_layers, 0, "x"] <= 0)
    ilp_model.solver.setObjective(objective, "minimize")
    ilp_model.solver.optimize()
    if ilp_model.solver.getStatus() != "optimal":
        logger.info(f"Didn't find optimal solution for example index {index} due to time limit")
        return index, None
    updated_input = np.zeros_like(input_to_nn)
    for v in ilp_model.solver.getVars():
        name, value = v.name, ilp_model.solver.getVal(v)
        if name.startswith(f"modified_input_idx_") and name[-1] == "x":
            input_index = int(name.split("_")[-2])
            updated_input[input_index] = float(value)
        elif name.startswith(f"op_{ilp_model.num_layers}_") and name[-1] == "x":
            output_index = int(name.split("_")[-2])
            final_output = float(value)
    if final_output > 0:
        final_output = 1
    else:
        final_output = 0
    if debug:
        logger.info("We flipped the class!")
        logger.info(f"Output of nn {final_output}")
        logger.info(f"True Label {target}")
    if updated_input.shape[0] == 0:
        return index, None
    else:
        return index, updated_input

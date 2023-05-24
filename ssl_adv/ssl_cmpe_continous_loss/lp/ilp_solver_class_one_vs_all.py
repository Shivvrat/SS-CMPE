import sys

import numpy as np
import pyscipopt
from loguru import logger
from numpy import argmax
from pyscipopt import Model, quicksum, quickprod


class CheckILPOutput():
    def __init__(self, name, args, nn, input_to_nn, target, f_plus_g, debug=False):
        self.weights = None
        self.bias = None
        self.nn = nn
        self.variables = {}
        self.objective_values = []
        self.target = target
        self.solver = Model(name)
        args.func_f = "min_distance"
        self.f_plus_g = f_plus_g
        self.m_plus = 10000
        self.m_minus = -10000
        # Pre solve does not allow symmetry constraints - turn it off
        self.solver.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
        # Set the max time for solving
        if not debug:
            self.solver.setParam('limits/time', 180)
        else:
            self.solver.setParam('limits/time', 60)

        self.num_layers = len(nn) // 2
        # Layer number starts at zero - Values at zero is that of input
        self.input_as_vector = input_to_nn
        # assert self.input_as_vector.shape[0] == args.input_size, "Input vector to ILP is not correct"
        for idx, val in enumerate(self.input_as_vector):
            self.variables[0, idx, "x_old"] = float(val)
            self.variables[0, idx, "x"] = self.solver.addVar(name=f"modified_input_idx_{idx}_x",
                                                             lb=0, ub=1, vtype="C")
            # Add the flip constraints for this index
            if args.func_f == "min_distance":
                self.add_constraints_for_f_min_distance(idx)
            
        

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

    # @njit()
    def add_constraints_one_layer(self, layer_number, last_layer=False):
        # the size of weight vector is [output, input]
        for idx_o in range(self.weights.shape[0]):
            if not last_layer:
                self.variables[layer_number, idx_o, "x"] = self.solver.addVar(name=f"op_{layer_number}_idx_{idx_o}_x",
                                                                              lb=0)
            else:
                # For last layer we don't need a lower bound.
                self.variables[layer_number, idx_o, "x"] = self.solver.addVar(name=f"op_{layer_number}_idx_{idx_o}_x",
                                                                              lb=None)
            self.variables[layer_number, idx_o, "s"] = self.solver.addVar(name=f"op_{layer_number}_idx_{idx_o}_s",
                                                                          lb=0)
            # Add the values of all output from ReLu as object function
            if not last_layer:
                # Don't add the output for last layer since it can be zero
                self.objective_values.append(self.variables[layer_number, idx_o, "x"])
                # No need for z since SCIP takes care of the indicator constraints
            this_sum = []
            # Calculate output for layer "layer_number" and node index "idx_o"
            for idx_i in range(self.weights.shape[1]):
                # Values of w*x
                this_sum.append(self.variables[layer_number - 1, idx_i, "x"] * self.weights[idx_o, idx_i])
            # Values of b
            this_sum.append(self.bias[idx_o])
            if not last_layer:
                # We make the following variables continous so that we can have a LP relaxation
                self.variables[layer_number, idx_o, "b"] = self.solver.addVar(name=f"binary_{layer_number}_idx_{idx_o}_b",
                                                                          lb=0, ub=1, vtype="C")
                self.solver.addCons(self.variables[layer_number, idx_o, "x"] >= quicksum(this_sum))
                self.solver.addCons(self.variables[layer_number, idx_o, "x"] <= quicksum(this_sum) - self.m_minus * (1 - self.variables[layer_number, idx_o, "b"]))
                self.solver.addCons(self.variables[layer_number, idx_o, "x"] <= self.m_plus * self.variables[layer_number, idx_o, "b"])
                
            else:
                # If we need to optimize for f + g then add the output of the neural network to the objective function
                if self.f_plus_g:
                    if self.target == 0:
                        self.objective_values.append(- self.variables[layer_number, idx_o, "x"])
                    elif self.target == 1:
                        self.objective_values.append(self.variables[layer_number, idx_o, "x"])
                        
                # For final layer we just need the sum, since we apply softmax
                self.solver.addCons(quicksum(this_sum) == self.variables[layer_number, idx_o, "x"])

    def add_constraint_all_layers(self, ):
        num_layers = self.num_layers
        self.weights = self.nn[f"l1.weight"]
        self.bias = self.nn[f"l1.bias"]
        self.add_constraints_one_layer(0 + 1, True)


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
    name, nn, input_to_nn, target, f_plus_g, feasible_region, debug, main_args, index = args
    # feasible_region is a binary variable which deteremines if we want the values for feasible region or not
    # Define a dictionary to hold the variables in the ILP problem
    # f_plus_g is a binary variable which determines if we want to add g to the objective function or not
    input_to_nn = input_to_nn.reshape(-1)
    ilp_model = CheckILPOutput("ILP_for_nn", main_args, nn, input_to_nn, target, f_plus_g, debug)
    if debug:
        ilp_model.solver.hideOutput(False)
    else:
        ilp_model.solver.hideOutput(True)
    assert feasible_region != f_plus_g, "feasible_region and f_plus_g cannot be both true or both false"
    ilp_model.add_constraint_all_layers()
    objective = quicksum(ilp_model.objective_values)
    if feasible_region:
        # Feasible region (for adversarial case) is the region where our class flips
        if target == 1:
            ilp_model.solver.addCons(ilp_model.variables[ilp_model.num_layers, 0, "x"] <= - 1e-10)
        else:
            ilp_model.solver.addCons(- ilp_model.variables[ilp_model.num_layers, 0, "x"] <= 0)
    else:
        # Infeasible region (for adversarial case) is the region where our class does not flips
        if target == 0:
            ilp_model.solver.addCons(ilp_model.variables[ilp_model.num_layers, 0, "x"] <= - 1e-10)
        else:
            ilp_model.solver.addCons(- ilp_model.variables[ilp_model.num_layers, 0, "x"] <= 0)
    ilp_model.solver.setObjective(objective, "minimize")
    # Linear Relaxation of the ILP
    for v in ilp_model.solver.getVars():
        ilp_model.solver.chgVarType(v, 'C')
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
            # logger.info(name, value)
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

import torch
import torch.utils.data as data

from lp.ilp_runner import run_ilp

def print_range(tensor):
    # Find the minimum and maximum values in the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # Print the range of the tensor
    print(f"Range of tensor: [{min_val}, {max_val}]")

@torch.no_grad()
def get_initial_bounds_for_num(dataset, nn_as_dict_of_np_array, f_xy, args):
    debug = False
    test_batch_size = 1028
    # Create your dataloader object
    data_loader = data.DataLoader(dataset, shuffle=False, batch_size=test_batch_size)
    updated, initial, missed_examples = run_ilp(test_batch_size, nn_as_dict_of_np_array,  args, data_loader, feasible_region=True)
    updated = torch.from_numpy(updated)
    initial = torch.from_numpy(initial)
    bounds = f_xy(initial, updated)
    bounds = torch.mean(bounds, dim=1)
    new_row = bounds[0]
    for i, index in enumerate(missed_examples):
        bounds = torch.cat((bounds[:(index+i)], new_row.unsqueeze(0), bounds[(index+i):]), dim=0)
    return bounds
    
    
@torch.no_grad()
def get_denom_for_lambda(dataset, nn_as_dict_of_np_array, f_xy, g_y, args):
    debug = False
    test_batch_size = 1028
    # Create your dataloader object
    data_loader = data.DataLoader(dataset, shuffle=False, batch_size=test_batch_size)
    # We want to find the denominator for the lambda - need the bounds in infeasible region
    updated, initial, missed_examples = run_ilp(test_batch_size, nn_as_dict_of_np_array,  args, data_loader, add_f_plus_g=True, feasible_region=False)
    updated = torch.from_numpy(updated).float()
    initial = torch.from_numpy(initial).float()
    bounds = f_xy(initial, updated)
    bounds = torch.mean(bounds, dim=1)
    output_g_y = torch.sigmoid(g_y(updated.to(args.device)))
    if not args.not_class:
        # class 
        non_satisfied_loss = (output_g_y - 0.5)
    else:
        # not class
        non_satisfied_loss = -(output_g_y - 0.5)
    bounds += non_satisfied_loss.squeeze()
    new_row = bounds[0]
    for i, index in enumerate(missed_examples):
        bounds = torch.cat((bounds[:(index+i)], new_row.unsqueeze(0), bounds[(index+i):]), dim=0)
    return bounds
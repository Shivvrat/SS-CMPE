import numpy as np

def load_dataset(location):
    dataset = np.load(location)
    return dataset
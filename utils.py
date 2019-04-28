import torch
import dataset
import model

import numpy as np

def eval_accuracy(network, n_pts):
    d_set = dataset.sphere_dataset()

    correct = 0
    for i in range(n_pts):
        output = network(d_set[i][0])
        if output > 0.0 and d_set[i][1] == 1:
            correct += 1
        elif output <= 0.0 and d_set[i][1] == 0:
            correct += 1
    return correct/n_pts

def histogram_outputs(network, n_pts):
    outputs = np.zeros(n_pts)
    d_set = dataset.sphere_dataset()

    for i in range(n_pts):
        outputs[i] = network(d_set[i][0])

    freqs, bin_edges = np.histogram(outputs,bins=30)
    return freqs, (bin_edges[1:] + bin_edges[:-1])/2.

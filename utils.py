import torch
import dataset
import model

import numpy as np

def eval_accuracy(network, n_pts):
    d_set = dataset.sphere_dataset()
    eval_loader = torch.utils.data.dataloader.DataLoader(d_set, 
                                                         batch_size=n_pts)
    test_pts, labels = next(iter(eval_loader))
    with torch.no_grad():
        outputs = network(test_pts)

    # the prediction is correct if sign(output) == sign(labels-0.5)
    # the below code is just a convoluted way of calculating 
    # the number of correct predictions
    # TODO make this more readable
    
    correct = torch.sum(((outputs.sign().view(1,-1) * (labels.float() - 0.5).sign()) + 1.)/2.).item()
 
    return correct/n_pts

def histogram_outputs(network, n_pts):
    outputs = np.zeros(n_pts)
    d_set = dataset.sphere_dataset()

    for i in range(n_pts):
        outputs[i] = network(d_set[i][0])

    freqs, bin_edges = np.histogram(outputs,bins=30)
    return freqs, (bin_edges[1:] + bin_edges[:-1])/2.

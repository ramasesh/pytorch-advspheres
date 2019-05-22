import torch
import dataset
import model

import numpy as np

def eval_accuracy(network, n_pts):
    """ Evaluates the accuracy of a network with 
    points sampled uniformly from both spheres"""
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

def eval_accuracy_single(network, n_pts, radius, desired_label):
    """ Evaluates the accuracy of a network with 
    points sampled uniformly from a single sphere
    Arguments:
        network - the network to evaluate
        n_pts - the number of points to use for evaluation
        radius - radius of the sphere to sample points from
        desired_label - 0 for inner sphere (model output should be less than zero), 
                        1 for outer sphere (model output should be greater than zero)
    """
    d_set = dataset.single_sphere_dataset(radius=radius)
    eval_loader = torch.utils.data.dataloader.DataLoader(d_set, 
                                                         batch_size=n_pts)

    test_pts = next(iter(eval_loader))
    with torch.no_grad():
        outputs = network(test_pts)
        
        if desired_label == 0:
            correct = torch.sum(outputs < 0).item()
        elif desired_label == 1:
            correct = torch.sum(outputs > 0).item()

    return correct/n_pts 

def histogram_outputs(network, n_pts):
    outputs = np.zeros(n_pts)
    d_set = dataset.sphere_dataset()

    for i in range(n_pts):
        outputs[i] = network(d_set[i][0])

    freqs, bin_edges = np.histogram(outputs,bins=30)
    return freqs, (bin_edges[1:] + bin_edges[:-1])/2.

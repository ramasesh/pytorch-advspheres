import torch
import numpy as np
import dataset

def projected_GD(network, 
                 point, 
                 correct_classification,
                 max_iter=1000,
                 step_size=0.001,
                 verbose=False):
    """ performs projected gradient descent to find an adversarial perturbation
    to the initial point 'point' that causes a misclassification by the network """
    
    radius = torch.norm(point, p=2)

    loss_fun = torch.nn.BCEWithLogitsLoss()
    correct_classification_tensor = torch.tensor([correct_classification]).float()

    point.requires_grad_(True)
    point.grad = None
    
    found = False
    for iter_num in range(max_iter):
        output = network(point)
        if (output < 0 and correct_classification == 1) or (output > 0 and correct_classification == 0):
            found = True
            break

        loss = loss_fun(output, correct_classification_tensor)
             
        loss.backward()
   
        delta = step_size*point.grad
        
        with torch.no_grad():
            point = point + delta
            point = point/torch.norm(point, p=2)
        point.grad = None
        point.requires_grad_(True)

        if verbose and iter_num % 100 == 0:
            print('Iter {}, loss {}'.format(iter_num, loss.item()))
 
    if found:
        return point.data
    else:
        return None 

def distance_to_errorset(network,
                         point,
                         correct_classification,
                         max_iter=1000,
                         step_size=0.001,
                         verbose=True):

    adversarial_point = projected_GD(network, 
                                      point,
                                      correct_classification,
                                      max_iter,
                                      step_size,
                                      verbose)

    if adversarial_point is None:
        return None
    else:
        return torch.norm(adversarial_point - point, p=2)

def avg_distance_to_errorset(network,
                             radius,
                             correct_classification,
                             n_pts = 100,    
                             max_iter=1000,
                             step_size=0.001,
                             verbose=True):
    """ For a given network, calculates the average distance (2-norm) from 
    points uniformly sampled on the sphere of a give radius to the error set"""

    d_set = dataset.single_sphere_dataset(radius=radius)
    
    distances = []
    
    for test_idx in range(n_pts):
        dist = distance_to_errorset(network,
                                    d_set[test_idx],
                                    correct_classification=correct_classification,
                                    max_iter=max_iter,
                                    step_size=step_size)
        if dist is not None:
            distances.append(dist)
    
    return torch.mean(torch.tensor(distances)).item()


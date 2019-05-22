import numpy as np
import torch

import model
import utils

import glob
import yaml

log_file = 'log_file_{}.yaml'

network_types = ['large', 'small']
network_params = {'large': {'sphere_dim': 500, 'n_hidden': 2000},
                  'small': {'sphere_dim': 500, 'n_hidden': 1000}}

evaluation_params = {'n_pts': int(1e6), 'radius': 1, 'desired_label': 0}

accuracies = {}

folder_names = ['batch_1']

for folder in folder_names:
    for network_type in network_types:
        files_to_run = glob.glob('trained_models/{}/*{}*.pth'.format(folder,network_type))
        
        print(files_to_run)

        if network_type == 'large':
            test_net = model.LargeReLU(**network_params['large'])
        elif network_type == 'small':
            test_net = model.SmallReLU(**network_params['small'])
       
        for filename in files_to_run:

            loaded_params = torch.load(filename)
            test_net.load_state_dict(loaded_params['model_state_dict'])
            
            accuracy = utils.eval_accuracy_single(test_net, **evaluation_params)
        
            accuracies[filename] = accuracy

    with open(log_file.format(folder), 'w') as f:
        yaml.dump(accuracies, f)

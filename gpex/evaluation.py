

import numpy as np
import torch
import torch.utils.data
import gpex




def GPdiffANN_ontorchds(gpmodel, ds_input, func_similaritymeasure, input_device):
    '''
    Comptues the difference between a GP and ANN on the dataset.
    '''
    gpmodel.eval()
    with torch.no_grad():
        list_toret = []
        for n in range(len(ds_input)):
            print(" instance {} from {}".format(n, len(ds_input)), end='\r')
            x, y, _ = ds_input[n]
            #feed to gp ====
            gp_output, _, _ = gpmodel.testingtime_forward(
                                x.unsqueeze(0).to(input_device), y, n
                            )
            gp_output = gp_output.detach().cpu().numpy().flatten()
            #feed to ann ===
            ann_output = gpmodel.module_tobecomeGP(x.unsqueeze(0).to(input_device))
            ann_output = ann_output.detach().cpu().numpy().flatten()
            list_toret.append(func_similaritymeasure(gp_output , ann_output))
            
            
        print("\n")
    gpmodel.train()
    return list_toret



def GPdiffANN_ontorchdl(gpmodel, dl_input, input_device):
    '''
    Comptues the difference between a GP and ANN on a torch dataloader.
    '''
    gpmodel.eval()
    set_seen = set([])
    dict_n_to_diff = {
        n:None for n in range(len(dl_input))
    }
    dict_n_to_gt = {
        n:None for n in range(len(dl_input))
    }
    for idx, data in enumerate(dl_input):
        print("Visited {} instances out of {}.".format(
                len(set_seen), len(dl_input.dataset)
                ),
              end='\r'
        )
        #compute GPdiffANN on the minibatch
        x, y, n = data
        with torch.no_grad():
            #feed to gp ====
            gp_output, _, _ = gpmodel.testingtime_forward(
                            x.to(input_device), y, n
            )
            gp_output = gp_output[0].detach().cpu().numpy()[:,:,0,0]
            #feed to ann ===
            ann_output = gpmodel.module_tobecomeGP(x.to(input_device)) #[N x Dv]
            ann_output = ann_output.detach().cpu().numpy() #[N x Dv]
        
        #grab results ====
        gp_cat_ann = np.concatenate([gp_output, ann_output], 1)
        n = n.tolist()
        y = y.tolist()
        set_seen = set_seen.union(set(n))
        for local_j, global_j in enumerate(n):
            dict_n_to_diff[global_j] = gp_cat_ann[local_j, :]
            dict_n_to_gt[global_j] = y[local_j]
    gpmodel.train()
    return dict_n_to_diff, dict_n_to_gt

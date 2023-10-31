



#general TODOs: now the code asumes [NxC x *] where * is any number of 1s rather than any number of hxwx... 's. Include the more general assumption.




import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import sys
import os
import time
import statistics
#import psutil
import copy
import pickle
import re
from abc import ABC, abstractmethod
import math
import copy
import xml.etree.ElementTree as ET
import gc
from copy import deepcopy
from pathlib import Path
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import skimage
import PIL
from skimage.filters import threshold_otsu
import torchvision.models as torchmodels


import torch.utils.data
import torchvision
import torch.nn as nn
from torch.autograd import Function

clamp_min, clamp_max = 0.00001, 5.0 #TODO:check #---override


def set_moduleweights_to_zero(m, variance_bais=0.1):
    '''
    Given a module, sets the weights of the module to zero, 
        and sets biases to random values generated around zero.
    '''
    setmodule_havingweight = {
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.Linear,
        nn.Bilinear,
    }
    if type(m) in setmodule_havingweight:
        torch.nn.init.zeros_(m.weight)
        if(m.bias is not None):
            m.bias.data.fill_(np.random.randn()*variance_bais)




class ControlVariate:
    def __init__(self, int_mode):
        self.int_mode = int_mode
        assert(self.int_mode in [1,2])
        print("Controlvariate mode is set to {}".format(self.int_mode))
        
    def __call__(self, x, xhat):
        '''
        Input.
            x: a tensor of shape [N x D].
            xhat: a tensor of shape [N x D].
        '''
        with torch.no_grad():
            N, D = list(x.size())[0], list(x.size())[1]
            #copmute cov_xxhat =====
            mu_1 = torch.sum(x, 0).unsqueeze(0) #[1 x D]
            mu_2 = torch.sum(xhat, 0).unsqueeze(0) #[1 x D]
            normalized_x    = x - mu_1 #[N x D]
            normalized_xhat = xhat - mu_2 #[N x D]
            cov_xxhat = torch.mean(normalized_x * normalized_xhat, 0) #[D]
            #compute cov_xhatxhat ===
            cov_xhatxhat = torch.mean(normalized_xhat * normalized_xhat, 0) #[D]
            astar = (cov_xxhat / cov_xhatxhat).unsqueeze(0) #[1xD]
        
        #compute the modified input ===
        if(self.int_mode == 1):
            toret = x - (astar*normalized_xhat).detach()
            return toret #TODO:check the effect of E[h(z)]
        elif(self.int_mode == 2):
            mu_2_gradpass = torch.sum(xhat, 0).unsqueeze(0) #[1 x D]
            normalized_xhat_gradpass = xhat - mu_2_gradpass #[N x D]
            toret = x - (astar.detach()*normalized_xhat_gradpass)
            return toret #TODO:check the effect of E[h(z)]
        else:
            assert False
        


def func_printcudameminfo():
    '''Prints total memory used by cuda.'''
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(" >>>> total={}, reserved={}, allocated={}, free={}\n\n\n".format(t, r, a, f))



class MatmulInvXXTplusSigma2:
    '''
    This class implements multiplication of 
    (XX^T + sigma2I)^(-1) by another vector. 
    X is [NxD], where N>>D. 
    '''
    def __init__(self, sigma2, device):
        #grab args
        self.sigma2 = sigma2
        self.device = device
    
   
    def unscalable_forward(self, X, y):
        XT = torch.transpose(X, 0, 1) #[DxN]
        XXT = torch.matmul(X, XT) #[NxN]
        N, D = X.size()[0], X.size()[1]
        
        temp = torch.inverse(XXT + self.sigma2*torch.eye(N)) 
        toret = torch.matmul(temp, y)
        return toret
        
    def forward(self, X, y, flag_getlogdetmat1=False, dict_args_efficiency=None):
        '''
        Inputs.
            - X: tensor of shape [NxD].
            - y: tensor of shape [N]. 
        Output.
            - (XX^T + sigma2I)^(-1) * y, tensor of shape [Nx1].
        '''
        
        #compute the eigen vectors/values of XTX ====
        if(dict_args_efficiency is None):
            XT = torch.transpose(X, 0, 1) #[DxN]
            XTX = torch.matmul(XT, X) #[DxD]
        else:
            assert("mode" in list(dict_args_efficiency.keys()))
            if(dict_args_efficiency["mode"] == "allowgrad"):
                #get args ==
                precomputed_XTX = dict_args_efficiency["precomputed_XTX"]
                idx_in_inputarg = dict_args_efficiency["idx_in_inputarg"]
                idx_in_globalGPX = dict_args_efficiency["idx_in_globalGPX"]
                global_GPX = dict_args_efficiency["global_GPX"]
                idx_slice_gpx = dict_args_efficiency["slice_GPX"]#note that this function is called for only one slice of Du.
                dimv = dict_args_efficiency["dimv"]
                
                #compute XTX ===
                X_oldvals = global_GPX.detach()[idx_in_globalGPX, :]
                X_oldvals = X_oldvals[:, idx_slice_gpx[0]:idx_slice_gpx[1]]
                X_gradpass = X[idx_in_inputarg[0]:idx_in_inputarg[1], :]
                XTX = precomputed_XTX[dimv].detach() - torch.matmul(torch.transpose(X_oldvals,0,1), X_oldvals).detach() + torch.matmul(torch.transpose(X_gradpass,0,1), X_gradpass)
                
                # ~ XT = torch.transpose(X, 0, 1) #[DxN]
                # ~ XTX_totest = torch.matmul(XT, X) #[DxD]
                # ~ print(">>>>>>>>>>>>>>> max diff between the two XTXs = {}".format(torch.max(XTX-XTX_totest)))
            elif(dict_args_efficiency["mode"] == "detachGP"):
                print("the mode is detachGP")
                dimv = dict_args_efficiency["dimv"]
                XTX = dict_args_efficiency["precomputed_XTX"][dimv]
            else:
                print("unknown mode {}".format(dict_args_efficiency["mode"]))
                assert False
            
        D = list(XTX.size())[0]
        N, D = X.size()[0], X.size()[1]
        if(True):
            landa_1toD, Qd= torch.linalg.eigh(XTX) #TODO:majorchange torch.symeig(XTX, eigenvectors=True) #u, s, v = torch.svd(L)
            landa_1toD = torch.clamp(landa_1toD, min=0.0001, max=np.inf) #TODO:check
            #find pairwisedifferences between the eigen-values =====
            with torch.no_grad():
                np_landa_1xD = np.expand_dims(landa_1toD.detach().cpu().numpy().flatten(), 0) #[1xD]
                np_landa_Dx1 = np.expand_dims(landa_1toD.detach().cpu().numpy().flatten(), 1) #[Dx1]
                np_pairwisediff = (np_landa_Dx1 - np_landa_1xD) + np.eye(D) #[DxD]
                flag_svdsafe = True
                # ~ if(np.min(np.abs(np_pairwisediff)) < 0.000000001): #TODO:check
                    # ~ flag_svdsafe = False
                    
            if(flag_svdsafe == False):
                print("<<<<<<<<<<<<<<< detected unsafe svd >>>>>>>>>>>>>>>>")
                return -1
        
        
        
        #compute the eigne vectors/values of XXT
        Qn = torch.matmul(X, Qd) #[NxD]
        landa_n = torch.cat(
            [landa_1toD.flatten(),\
             torch.Tensor(np.array((N-D)*[0.0])).float().to(self.device)]
        ) #[N]
        
        
        #normalize Qn ===
        norm_Qn = torch.sqrt(torch.sum(Qn*Qn, 0)).unsqueeze(0) #[1xD]
        Qn = Qn/norm_Qn
        
        
        #compute the eigen vectors/values of XXT + sigma2I ===
        Qn_plussigma2I = Qn
        landa_plussigma2I = landa_n + self.sigma2
        
        
        #compute the eigen vectors/values of inv(XXT + sigma2I) ===
        Qn_plussigma2I_inv = Qn_plussigma2I #[NxD]
        landa_plussigma2I_inv = 1.0/landa_plussigma2I #[N]
        
        
        #compute the part in the D-largest space ===
        mat1 = Qn_plussigma2I_inv #[NxD]
        mat2 = torch.diag(landa_plussigma2I_inv[0:D].flatten()) #[DxD]
        mat3 = torch.transpose(Qn_plussigma2I_inv, 0, 1) #[DxN]
        mat4 = y #[Nx1]
        matmul_234 = torch.matmul(
                torch.matmul(
                    mat2,
                    mat3
                ),
                mat4
            )#[Dx1]
        output_Dspace = torch.matmul(mat1, matmul_234) #[Nx1]
        
        
        #compute the part in the (N-D)-space ====
        y_on_Dspace = torch.matmul(
            Qn_plussigma2I_inv,
            torch.matmul(
                torch.transpose(Qn_plussigma2I_inv, 0, 1),
                y
            )
        ) #[Nx1]
        y_on_NminDspace = y - y_on_Dspace #[Nx1]
        output_NminDspace = landa_plussigma2I_inv[-1] * y_on_NminDspace #[Nx1]
        
        #test if y_on_NminDspace is orthogonal to the D-space ===
        #temp = torch.matmul(torch.transpose(Qn_plussigma2I_inv, 0, 1), y_on_NminDspace)
        #print(landa_1toD)
        
        #return the determinant of the covariance matrix 
        if(flag_getlogdetmat1 == True):
            return output_Dspace + output_NminDspace,\
                   torch.sum(torch.log(landa_plussigma2I)),\
                   torch.sum(landa_plussigma2I_inv)
        else:
            return output_Dspace + output_NminDspace
        






class forward_replaced:
    """
    termporarily replaces the forward of the module under consideration.
    """
    def __init__(self, module_tobecomeGP, func_newforward):
        #grab args ===
        self.module_tobecomeGP = module_tobecomeGP
        self.func_newforward = func_newforward
        #save the original forward fucntion ====
        self.original_forward = module_tobecomeGP.forward
        
    def __enter__(self):
        self.module_tobecomeGP.forward = self.func_newforward

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.module_tobecomeGP.forward = self.original_forward






class GPEXModule(nn.Module):
    '''
    The main module to be created in order to use GPEX.
    '''
    def __init__(self, module_rawmodule, size_recurringdataset, device, func_mainmodule_to_moduletobecomeGP,
                 func_feed_noise_minibatch,
                 func_feed_inducing_minibatch, func_feed_nonrecurring_minibatch, func_feed_test_minibatch,
                 func_get_indices_lastrecurringinstances,
                 func_get_modulef1,
                 flag_efficient = True,
                 flag_detachcovpvn = True,
                 flag_setcovtoOne =  False,
                 flag_controlvariate = True,
                 int_mode_controlvariate = 2,
                 flag_train_memefficient = False,
                 memefficeint_heads_in_compgraph = None
         ):
        '''
        Inputs.
            - module_rawmodule: the raw module in which a module it to be replaced by GP.
            - size_recurringdataset: the size of the inducing dataset (i.e. the variable M in paper).
            - device: torch device to be used.
            - func_mainmodule_to_moduletobecomeGP: a function that takes in your pytorch module, and returns the ANN submodule to be replaced by GP.
            - func_feed_inducing_minibatch: in this function you should implement how a mini-batch from the inducing dataset is fed to your pytorch module.
              This function has to have 0 input arguments.
            - func_feed_noise_minibatch: in this function you should implement how a mini-batch of instances over which the GP is matched to ANN, is fed to your pytorch module.
              As explained in the paper and implemented in the sample notebook, a proper way is to feed a minibatch of samples like `lambda*x + (1.0-lambda)*(1-x)`.   
              This function has to have 0 input arguments.
            - func_feed_nonrecurring_minibatch: in this function you should implement how a mini-batch from the training dataset if fed to your pytorch module.
              This function has to have 0 input arguments.
            - func_feed_test_minibatch: in this function you should implement how a mini-batch from the testing dataset is fed to your pytorch module.
              This function has to have 0 input arguments.
            - func_get_indices_lastrecurringinstances: A function that returns the indices of the inducing instances which are fed to the module.
              In other words, when implementing `func_feed_xxx_minibatch` you should put the indices of the inducing instances which are last fed, in a list or os,
              so you can return it later on in this function. 
              Importantly, you have to update the list "before" calling any forward functions (as done in the sample notebook).
              Otherwised, it may lead to unwanted behaviour.
            - func_get_modulef1: This function has to have 0 input arguments, and returns the kenel module. 
              In the notation of the paper, let's say ANN has L output heads so there will be L kernel functions.
              If each kernel-space is considered D-dimensional, the output of the kernel module has to be D*L dimensional,
              where each group of D dimensions should be L2-normalized. 
        '''
        super(GPEXModule, self).__init__()
        #grab arguments ===
        self.module_rawmodule = module_rawmodule#TODO:check why deepcopy doesn't work. copy.deepcopy(module_rawmodule)
        self.size_recurringdataset = size_recurringdataset
        self.device = device
        self.func_mainmodule_to_moduletobecomeGP = func_mainmodule_to_moduletobecomeGP
        self.func_feed_noise_minibatch = func_feed_noise_minibatch
        self.func_feed_recurring_minibatch = func_feed_inducing_minibatch
        self.func_feed_nonrecurring_minibatch = func_feed_nonrecurring_minibatch
        self.func_feed_test_minibatch = func_feed_test_minibatch
        self.func_get_indices_lastrecurringinstances = func_get_indices_lastrecurringinstances
        self.func_get_modulef1 = func_get_modulef1
        self.flag_efficient = flag_efficient
        self.flag_detachcovpvn = flag_detachcovpvn
        self.flag_setcovtoOne = flag_setcovtoOne
        self.flag_controlvariate = flag_controlvariate
        self.flag_train_memefficient = flag_train_memefficient
        self.memefficeint_heads_in_compgraph = memefficeint_heads_in_compgraph
        
             
        
        #make internals ===
        self.module_tobecomeGP = func_mainmodule_to_moduletobecomeGP(self.module_rawmodule)
        self.func_rawforward_tobecomeGP = self.module_tobecomeGP.forward
        self.module_f1 = self.func_get_modulef1()
        if(self.flag_controlvariate == True):
            self.controlvariate = ControlVariate(int_mode = int_mode_controlvariate)
        
        
        #synch self._rng_heads_in_compgragh with that of module_f1
        if(self.flag_train_memefficient == True):
            assert(self.memefficeint_heads_in_compgraph is not None)
            self._set_rng_headsincompgraph([0, self.memefficeint_heads_in_compgraph+0])
        
        
        #infer Du and Dv ===
        #if(self.flag_train_memefficient == True):
        #    self.module_f1.set_rng_outputheads(rng_outputhead = self._rng_heads_in_compgragh)
        self._inferDvDv()
        self._infer_outputsize_mdouletobecomeGP()
        #if(self.flag_train_memefficient == True):
        #    self.module_f1.set_rng_outputheads(rng_outputhead = None)
        
        
        
        #to enable multi-kernel, devide Dv over Du dimensions ====
        list_kerneldims = [0 for i in range(self.Dv)]
        idx_dims = 0
        for i in range(self.Du):
            list_kerneldims[idx_dims] += 1
            idx_dims = int((idx_dims+1)%len(list_kerneldims))
        list_rngkernels = []
        count_filleddims = 0
        for u in list_kerneldims:
            list_rngkernels.append([count_filleddims, count_filleddims+u])
            count_filleddims += u
        self._list_rngkernels = list_rngkernels
        
        #make GP internal modules (e.g. GPX and GPY) ====
        self.flag_svdfailed = False
        self.sigma2_GP = 1.0 #TODO:check
        self.module_matmulxxtplussigma2 = MatmulInvXXTplusSigma2(sigma2 = self.sigma2_GP, device = self.device)
        self.GP_X = torch.nn.Parameter( 
                 torch.randn(self.size_recurringdataset, self.Du, requires_grad = False, device=self.device)
                ) #[N x Du]
        self.GP_Y = torch.nn.Parameter(
                    torch.randn(self.size_recurringdataset, self.Dv, requires_grad = True, device=self.device)
                 )#[N x Dv]
        self.cov_qvn = torch.Tensor([[0.0]]).float().to(self.device)         
        if(self.flag_efficient == True):
            self.precomputed_XTX = [0 for i in range(self.Dv)]
            self.renew_precomputed_XTX()
            # ~ self.precomputed_XTX = []
            # ~ for dimv, rngdim in enumerate(self._list_rngkernels):
                # ~ sliced_GPX = self.GP_X[:, int(rngdim[0]):int(rngdim[1])] #[size_recurringdataset x urng]
                # ~ self.precomputed_XTX.append(torch.matmul(torch.transpose(sliced_GPX,0,1), sliced_GPX).detach())
        
    
    
    def init_UV(self):
        '''
        Initializes U (the kernel-space representaitons of the inducing points) and V (the GP posterior values at the inducing points).
        This function must be called before calling the function `getcost_explainANN`.
        '''
        self.initV_from_theannitself()
        self.initU_from_kernelmappings()
        
    
    def getparams_explainANN(self):
        '''
        Returns the parameters to be optimized for explain ANN.
        When optimizing the cost returned by `getcost_explainANN`, the optimizer has to operate on the parameters
        returned by this function.
        '''
        return self.module_f1.parameters()
        
    
    
    def _inc_rng_headsincompgraph(self):
        if(self._rng_heads_in_compgragh[1] == self.Dv):
            #the head is in its last position
            self._rng_heads_in_compgragh = [0 , self.memefficeint_heads_in_compgraph] #reset the head back to the beginning.
            return
        
        self._rng_heads_in_compgragh = [
                self._rng_heads_in_compgragh[0] + self.memefficeint_heads_in_compgraph,
                self._rng_heads_in_compgragh[1] + self.memefficeint_heads_in_compgraph
        ]
        if(self._rng_heads_in_compgragh[1] > self.Dv):
            #the head has gone of the end limit
            len_head = self._rng_heads_in_compgragh[1]-self._rng_heads_in_compgragh[0]
            self._rng_heads_in_compgragh = [self.Dv-len_head , self.Dv] #move the head to its end position.
        self.module_f1.set_rng_outputheads(rng_outputhead = self._rng_heads_in_compgragh)
        
    def _set_rng_headsincompgraph(self, input_rng):
        self._rng_heads_in_compgragh = input_rng
        self.module_f1.set_rng_outputheads(rng_outputhead = self._rng_heads_in_compgragh)
        
    def _synch_rng_heads_in_compgragh(self):
        self.module_f1.set_rng_outputheads(rng_outputhead = self._rng_heads_in_compgragh)
        
    def split_vectors_inkernelspace(self, input_x):
        '''
        Splits the Du dimensions in Dv groups, the same way they are splitted to build different kernels. 
        '''
        list_toret = []
        for dimv, rngdim in enumerate(self._list_rngkernels):
            sliced_x = input_x[:, int(rngdim[0]):int(rngdim[1])]
            list_toret.append(sliced_x)
        if(isinstance(input_x, np.ndarray)):
            return np.stack(list_toret)
        elif(isinstance(input_x, torch.Tensor)):
            return torch.stack(list_toret)
        else:
            print("Undefined input type {}.".format(type(input_x)))
            assert False
    
    def get_W_of_kernelspace(self):
        '''
        Returns the weights of the linear transform on the kernel space.
        '''
        local_GPX, local_GPY = self.GP_X.detach(), self.GP_Y.detach()
        list_w_dimv = []
        for dimv, rngdim in enumerate(self._list_rngkernels):
            sliced_GPX = local_GPX[:, int(rngdim[0]):int(rngdim[1])] #[size_recurringdataset x urng]
            sliced_GPY = local_GPY[:, dimv].unsqueeze(-1)  #[size_recurringdataset x 1]
            xxtplussigma2inverse = self.module_matmulxxtplussigma2.forward(
                                            sliced_GPX, sliced_GPY, dict_args_efficiency=None
                                    ) #[size_recurringdataset x 1]
            w_dimv = torch.matmul(
                        torch.transpose(sliced_GPX, 0, 1),
                        xxtplussigma2inverse
                     ) #[urng x 1]
            list_w_dimv.append(w_dimv)
        return torch.stack(list_w_dimv)
            
    
    def _forwardGP(self, x, GPX, GPY, flag_clampcov=True, flag_returnsimilarites=False, dict_args_efficiency=None):
        '''
        Forwards to the GP itself.
        Returns mux and covx.
        '''
        N, M, h, w = list(x.size())
        local_GPX, local_GPY = GPX, GPY #self.GP_X.detach(), self.GP_Y.detach()
        descriptors_vectorview = x.permute(0,2,3,1).reshape((-1, M)) #[Nhw x Du]
        if(self.flag_train_memefficient == True):
            du_per_gp = int(self.Du/self.Dv)
        
        
        list_mux, list_covx, list_similarities = [], [], []
        for dimv, rngdim in enumerate(self._list_rngkernels):
            if((self.flag_train_memefficient == True) and ((dimv >= self._rng_heads_in_compgragh[1]) or (dimv < self._rng_heads_in_compgragh[0]))):
                continue;
            
            
            if(self.flag_train_memefficient == False):
                sliced_GPX = local_GPX[:, int(rngdim[0]):int(rngdim[1])] #[size_recurringdataset x urng]
            else:
                sliced_GPX = local_GPX[:, du_per_gp*(dimv-self._rng_heads_in_compgragh[0]):du_per_gp*(1+dimv-self._rng_heads_in_compgragh[0])]
                
            
            if(self.flag_train_memefficient == False):
                sliced_GPY = local_GPY[:, dimv].unsqueeze(-1)  #[size_recurringdataset x 1]
            else:
                sliced_GPY = local_GPY[:, dimv-self._rng_heads_in_compgragh[0]].unsqueeze(-1) #in this case, local_GPY is the GPYs for a subset of GPs.
            
            if(self.flag_train_memefficient == False):
                sliced_descriptors_vectorview = descriptors_vectorview[:, int(rngdim[0]):int(rngdim[1])] #[Nhw x urng]
            else:
                sliced_descriptors_vectorview = descriptors_vectorview[:, du_per_gp*(dimv-self._rng_heads_in_compgragh[0]):du_per_gp*(1+dimv-self._rng_heads_in_compgragh[0])] #[Nhw x urng]
            
            if(dict_args_efficiency is not None):
                if(self.flag_train_memefficient == False):
                    dict_args_efficiency["slice_GPX"] = [int(rngdim[0]) , int(rngdim[1])]
                else:
                    dict_args_efficiency["slice_GPX"] = [int(rngdim[0]) , int(rngdim[1])]
                dict_args_efficiency["dimv"] = dimv
            xxtplussigma2inverse = self.module_matmulxxtplussigma2.forward(sliced_GPX, sliced_GPY, dict_args_efficiency=dict_args_efficiency) #[size_recurringdataset x 1]
            if(isinstance(xxtplussigma2inverse, int)):
                if(xxtplussigma2inverse == -1):
                    #in this case, the svd has failed =====
                    self.flag_svdfailed = True
                    print(">>>>>>>>>>>>>>>>>>> SVD Failed <<<<<<<<<<<<<<<<<<<<<")
                    return torch.ones(N*h*w, self.Dv).float().to(self.device), torch.ones(N*h*w, self.Dv).float().to(self.device) 
            mux = torch.matmul(
                        sliced_descriptors_vectorview,
                        torch.transpose(sliced_GPX, 0, 1) 
                     ) #[* x size_recurringdataset]
            mux = torch.matmul(mux , xxtplussigma2inverse) #[* x num_outputheads]
            list_mux.append(mux)
            #compute covx ====
            first_term = torch.matmul(sliced_descriptors_vectorview, sliced_descriptors_vectorview.permute(1,0)) #[Nhw x Nhw]
            second_term = self.module_matmulxxtplussigma2.forward(
                                    sliced_GPX,
                                    torch.matmul(sliced_GPX, sliced_descriptors_vectorview.permute(1,0)),
                                    dict_args_efficiency=dict_args_efficiency
                                   )#[size_recurringdataset x Nhw]
            second_term = torch.matmul(
                              torch.matmul(sliced_descriptors_vectorview, sliced_GPX.permute(1, 0)),
                              second_term
                           ) #[Nhw x Nhw]
            covx = torch.diagonal(first_term) - torch.diagonal(second_term) #[Nhw]
            #print("     >>>>>>>>>>> min of covx before clamp = {}".format(np.min(covx.detach().cpu().numpy())))
            if(flag_clampcov == True):
                covx = torch.clamp(covx, min=clamp_min, max=clamp_max)#TODO:check
            covx = covx.unsqueeze(-1) #[Nhw x 1]
            list_covx.append(covx)
            
            if(flag_returnsimilarites == True):
                toappend = torch.matmul(
                    sliced_descriptors_vectorview,
                    torch.transpose(
                        sliced_GPX, 0, 1
                      )
                   )
                list_similarities.append(
                    toappend
                 )
        mux = torch.cat(list_mux, 1)
        covx = torch.cat(list_covx, 1)
        if(flag_returnsimilarites == True):
            similarities = torch.stack(list_similarities, 0) #[Dv x nhw x size_recurringdataset]
        
        
        
        #return mux and covx
        if(flag_returnsimilarites == False):
            return mux, covx
        else:
            return mux, covx, similarities
    
    def _reshapeUorV(self, x):
        '''
        Reshapes a tensor of shape [N x D x *] to a tensor of shape [* x D].
        '''
        NDstar = list(x.size())
        N, D, star = NDstar[0], NDstar[1], NDstar[2:]
        x = x.permute(*([0]+[i+2 for i in range(len(star))]+[1])) #[N x * x D]
        x = x.view(-1, D) #[N* x D]
        return x, N, D
    
        
    def testingtime_forward(self, *args, **kwargs):
        with torch.no_grad():
            with forward_replaced(self.module_tobecomeGP, self._forward_testingtime_withoutuncertainty):
                output = self.module_rawmodule.forward(*args, **kwargs)
        if(self.flag_train_memefficient == True):
            self._set_rng_headsincompgraph(input_rng = [0 , self.memefficeint_heads_in_compgraph])
        return output, self._testingtime_uncertainties, self._testingtime_similarities
        
                
        
    
    def forward_GPwithoutuncertainty(self):
        '''
        Forwards a non-recurring mini-batch as if the GP is trained without uncertainty.
        '''
        with forward_replaced(self.module_tobecomeGP, self._forward_GPwithoutuncertainty):
            output = self.func_feed_nonrecurring_minibatch()
        return output
    
    
    
    def update_U(self):
        '''
        Updates some elements of GPX based on the current value of the function (i.e. the module) f1(.).
        '''
        self.module_rawmodule.eval()
        with torch.no_grad():
            with forward_replaced(self.module_tobecomeGP, self._forward_updateGPX):
                output = self.func_feed_recurring_minibatch()
        #self.renew_precomputed_XTX()
        #print("WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! remove the above line.")
        self.module_rawmodule.train()
        if(self.flag_train_memefficient == True):
            self._inc_rng_headsincompgraph()
    
    
    def getcost_GPmatchNN(self):
        '''
        Computes cost w.r.t. model params.
        '''
        self.flag_svdfailed = False
        self.module_rawmodule.eval()
        with forward_replaced(self.module_tobecomeGP, self._forward_makecostGPmatchNN):
            output = self.func_feed_noise_minibatch()
            #now the following tensors are set self._costModelParams_term1
        self.module_rawmodule.train()
        
        return self._cost_GPmatchNN_term1
        
    
    
    '''
    def getcost_explainANN(self):
        #Computes and returns the cost that encourages the Gaussian processes to behave similar to ANNs.
        self.flag_svdfailed = False
        self.module_rawmodule.eval()
        if(self.flag_train_memefficient == True):
            self._synch_rng_heads_in_compgragh()
        func_forward = self._forward_makecostModelParams if(self.flag_train_memefficient == False) else self._forward_makecostModelParams_trainmemefficient
        with forward_replaced(self.module_tobecomeGP, func_forward):
            tempoutput = self.func_feed_noise_minibatch()
            #tempoutput = tempoutput.detach()
            #now the following tensors are set self._costModelParams_term1
        cost_toret = self._cost_GPmatchNN_term1 + 0.0
        
        
        #now compute the dummy output (only not to break the forward) ====
        with torch.no_grad():
            with forward_replaced(self.module_tobecomeGP, self._forward_dummy):
                output = self.func_feed_nonrecurring_minibatch()
        self.module_rawmodule.train()
        
        if(self.flag_train_memefficient == True):
            self._inc_rng_headsincompgraph()
        
        return self._cost_GPmatchNN_term1, output
    '''
    
    def get_costQvn(self):
        '''
        Computes the KL-divergence part of elbo w.r.t. vn.
        '''
        #set eval-train settings of modules ===
        self.module_rawmodule.eval() #TODO:check
        self.module_f1.eval() #TODO:check
        self.module_tobecomeGP.train() #TODO:check
        
        
        with forward_replaced(self.module_tobecomeGP, self._forward_makecostQvn):
            output = self.func_feed_nonrecurring_minibatch()
            #now the following tensors are computed: self._cost_Qvn_klpart.
        
        toret_KL = self._cost_Qvn_klpart
        
        #revert train-eval settings ====
        self.module_rawmodule.train() #TODO:check
        self.module_f1.train() #TODO:check
        self.module_tobecomeGP.train() #TODO:check
        return toret_KL, output
    
    
    
    def get_costQvhatm(self):
        '''
        Computes the cost for Qvhatm parameters.
        '''
        self.module_rawmodule.eval() #TODO:check
        self.module_f1.eval() #TODO:check
        
        with forward_replaced(self.module_tobecomeGP, self._forward_makcostQvhatm):
            output = self.func_feed_nonrecurring_minibatch()
            #now the following tensors are computed: self._costQvhatm_term1, self._costQvhatm_klterm
        
        self.module_rawmodule.train() #TODO:check
        self.module_f1.train() #TODO:check    
        return self._costQvhatm_term1, self._costQvhatm_klterm, output
        
    
    def _forward_checkequalf1pathgpath(self, x):
        '''
        The forward when checking whether f1(.) path is equal to g(.) path.
        '''
        if(self.flag_train_memefficient == False):
            #pass x to g(.)
            output_g = self.func_rawforward_tobecomeGP(x) #[N x Dv] #REQUIREMENT: output of the ANN module has to be [N x Dv].
            assert(len(list(output_g.size())) == 2)
            toret = output_g + 0.0
            #output_g = output_g.squeeze() #[N x Dv]
        
            #pass x to GP(.) path
            output_gp, _ = self._getMuCovforQvhatm(self._reshapeUorV(self.module_f1(x))[0]) #[N x Dv]
        
            #add to list of outputs
            self._list_outputgp_versus_outputg.append([output_gp.detach().cpu().numpy(), output_g.detach().cpu().numpy()])
        
            return toret
        else:
            #pass x to g(.)
            output_g = self.func_rawforward_tobecomeGP(x) #[N x Dv x *]
            toret = output_g + 0.0
            output_g = output_g.squeeze() #[N x Dv]
            
            #pass x to GP(.) path
            tensor_u = []
            for idx_gp in range(self.Dv):
                self._set_rng_headsincompgraph(input_rng = [idx_gp , idx_gp+1])
                tensor_u.append(self.module_f1(x))
            tensor_u = torch.cat(tensor_u, 1) #[N x Dv]
            output_gp, _ = self._getMuCovforQvhatm(self._reshapeUorV(tensor_u)[0]) #[N x Dv]
            
            self._list_outputgp_versus_outputg.append([output_gp.detach().cpu().numpy(), output_g.detach().cpu().numpy()])
            return toret
    
    
    def _forward_checkequalf1pathgpath_ontorchdl(self, x):
        '''
        The forward when checking whether f1(.) path is equal to g(.) path on a Pytorch Dataloader.
        After this forward, the following fields will be set:
        - 
        - 
        '''
        if(self.flag_train_memefficient == False):
            #pass x to g(.)
            output_g = self.func_rawforward_tobecomeGP(x) #[N x Dv x *]
            toret = output_g + 0.0
            output_g = output_g.squeeze() #[N x Dv]
        
            #pass x to GP(.) path
            output_gp, _ = self._getMuCovforQvhatm(self._reshapeUorV(self.module_f1(x))[0]) #[N x Dv]
        
            #add to list of outputs
            self._f1pathversusgpath_ontorchdl_gpout = output_gp.detach().cpu().numpy()
            self._f1pathversusgpath_ontorchdl_annout = output_g.detach().cpu().numpy()
            #self._list_outputgp_versus_outputg.append([output_gp.detach().cpu().numpy(), output_g.detach().cpu().numpy()])
        
            return toret
        else:
            #pass x to g(.)
            output_g = self.func_rawforward_tobecomeGP(x) #[N x Dv x *]
            toret = output_g + 0.0
            output_g = output_g.squeeze() #[N x Dv]
            
            #pass x to GP(.) path
            tensor_u = []
            for idx_gp in range(self.Dv):
                self._set_rng_headsincompgraph(input_rng = [idx_gp , idx_gp+1])
                tensor_u.append(self.module_f1(x))
            tensor_u = torch.cat(tensor_u, 1) #[N x Dv]
            output_gp, _ = self._getMuCovforQvhatm(self._reshapeUorV(tensor_u)[0]) #[N x Dv]
            
            #self._list_outputgp_versus_outputg.append([output_gp.detach().cpu().numpy(), output_g.detach().cpu().numpy()])
            self._f1pathversusgpath_ontorchdl_gpout = output_gp.detach().cpu().numpy()
            self._f1pathversusgpath_ontorchdl_annout = output_g.detach().cpu().numpy()
            return toret
    
    
    def check_GPmatchANN_on_aDataloader(self, func_feed_dlinstances, func_get_lastidx_fedinstances, list_allidx):
        '''
        Given a daloader, checks whether the GP and the ANN match on the dataloader.
        Inputs. 
            - func_feed_dlinstances: a function. This function feeds some instances in the dataloader to the raw module.
            - func_get_lastidx_fedinstances: a function. This fucntion returns the indices of the last fed instances.
            - list_allidx: a list. list of all indices of the dataloader's instances that has to be fed to the raw module.
        '''
        self.eval()
        set_allidx = set(list_allidx)
        set_seensofar = set([])
        dict_idx_to_gpout = {}
        dict_idx_to_annout = {}
        #self._list_outputgp_versus_outputg = []
        with torch.no_grad():
            while(True):
                print("Visited {} instances out of {}".format(len(list(set_seensofar)), len(list_allidx)), end='\r')
                #check whether all instances are seen
                flag_done = (set_seensofar == set_allidx)
                if(flag_done == True):
                    break
                #feed new instances ====
                with forward_replaced(self.module_tobecomeGP, self._forward_checkequalf1pathgpath_ontorchdl):
                    output = func_feed_dlinstances()
                    idx_in_global = func_get_lastidx_fedinstances()
                    for idx_in_local in range(len(idx_in_global)):
                        dict_idx_to_gpout[idx_in_global[idx_in_local]] = self._f1pathversusgpath_ontorchdl_gpout[idx_in_local, :].flatten().tolist()
                        dict_idx_to_annout[idx_in_global[idx_in_local]] = self._f1pathversusgpath_ontorchdl_annout[idx_in_local, :].flatten().tolist()
                    set_seensofar = set(
                        list(set_seensofar) + idx_in_global
                    )
                
        self.train()
        #self._set_rng_headsincompgraph(input_rng = [0 , self.memefficeint_heads_in_compgraph])
        return dict_idx_to_gpout, dict_idx_to_annout
    
    
    def checkequal_f1path_gpath(self, num_iters):
        self.eval()
        self._list_outputgp_versus_outputg = []
        with torch.no_grad():
            for n in range(num_iters):
                with forward_replaced(self.module_tobecomeGP, self._forward_checkequalf1pathgpath):
                    output = self.func_feed_nonrecurring_minibatch()
        self.train()
        self._set_rng_headsincompgraph(input_rng = [0 , self.memefficeint_heads_in_compgraph])
        return self._list_outputgp_versus_outputg
        
    
    def checkequal_GPout_ANNout_ontest(self, num_iters):
        self.eval()
        self._list_outputgp_versus_outputg = []
        
        
        with torch.no_grad():
            for n in range(num_iters):
                with forward_replaced(self.module_tobecomeGP, self._forward_checkequalf1pathgpath):
                    output = self.func_feed_test_minibatch()
        self.train()
        if(self.flag_train_memefficient == True):
            self._set_rng_headsincompgraph(input_rng = [0 , self.memefficeint_heads_in_compgraph])
        return self._list_outputgp_versus_outputg
        
        
        
    
    
    def renew_precomputed_XTX(self):
        with torch.no_grad():
            for dimv, rngdim in enumerate(self._list_rngkernels):
                sliced_GPX = self.GP_X[:, int(rngdim[0]):int(rngdim[1])] #[size_recurringdataset x urng]
                self.precomputed_XTX[dimv] = torch.matmul(torch.transpose(sliced_GPX,0,1), sliced_GPX).detach() 
    
    
    def _forward_updateGPX(self, x):
        '''
        The forward function during updating some elements of GPX (the elemenets that correspond to the last mini-batch from recurring dataset).
        '''
        #get indices of last recurring instances ====
        list_idxrecurring = self.func_get_indices_lastrecurringinstances()
        
        #get outputs from the f1(.) module ====
        tensor_u = self.module_f1(x) #[N x Du x *]
        tensor_u, N, Du = self._reshapeUorV(tensor_u) #[N*  x Du]
        newvals_GPX = torch.nan_to_num(tensor_u.detach()) + 0.0 #It can be either [N x Du] or [N x Dusubset] based on self.flag_train_memefficient
        if(self.flag_train_memefficient == True):
            du_per_gp = int(self.Du/self.Dv)
        
        for dimv, rngdim in enumerate(self._list_rngkernels):
            #handle the case where self.flag_train_memefficient is set to True.
            if((self.flag_train_memefficient == True) and ((dimv >= self._rng_heads_in_compgragh[1]) or (dimv < self._rng_heads_in_compgragh[0]))):
                continue;
            
            sliced_GPX = self.GP_X[:, int(rngdim[0]):int(rngdim[1])] #[size_recurringdataset x urng]
            if(self.flag_train_memefficient == False):
                sliced_newvalsGPX = newvals_GPX[:, int(rngdim[0]):int(rngdim[1])]
            else:
                idx_in_slice = dimv - self._rng_heads_in_compgragh[0]
                sliced_newvalsGPX = newvals_GPX[:, idx_in_slice*du_per_gp:(idx_in_slice+1)*du_per_gp]
            
            newval_precomputed_XTX = self.precomputed_XTX[dimv].detach() -\
                                         torch.matmul(torch.transpose(sliced_GPX[list_idxrecurring, :],0,1), sliced_GPX[list_idxrecurring, :]).detach() +\
                                         torch.matmul(torch.transpose(sliced_newvalsGPX,0,1), sliced_newvalsGPX).detach()
            self.precomputed_XTX[dimv] = newval_precomputed_XTX
            
            #actualval = torch.matmul(torch.transpose(sliced_GPX,0,1), sliced_GPX).detach()
            #print("maxabs for dimv={} is {}".format(dimv, torch.max(torch.abs(newval_precomputed_XTX - actualval))))
        
        if(self.flag_train_memefficient == False):
            self.GP_X[list_idxrecurring, :] =  newvals_GPX + 0.0
        else:
            du_per_gp = int(self.Du/self.Dv)
            self.GP_X[list_idxrecurring, du_per_gp*self._rng_heads_in_compgragh[0]:du_per_gp*self._rng_heads_in_compgragh[1]] =  newvals_GPX + 0.0
        
        # ~ for dimv, rngdim in enumerate(self._list_rngkernels):
            # ~ sliced_GPX = self.GP_X[:, int(rngdim[0]):int(rngdim[1])] #[size_recurringdataset x urng]
            # ~ actualval = torch.matmul(torch.transpose(sliced_GPX,0,1), sliced_GPX).detach()
            # ~ print("maxabs for dimv={} is {}".format(dimv, torch.max(torch.abs(self.precomputed_XTX[dimv] - actualval))))
        
        
        return self._forward_dummy(x)
    
    
    def _forward_inferOutputsizeModuletobecomeGP(self, x):
        '''
        This forward infers the size of the output from `self.module_tobecomeGP`'s original forward function.
        '''
        toret = self.func_rawforward_tobecomeGP(x)
        self._size_output_moduletobecomeGP = list(toret.size())[1::]
        return toret
    
    def _forward_initGPYfromg(self, x):
        #get indices of last recurring instances ====
        list_idxrecurring = self.func_get_indices_lastrecurringinstances()
        
        #pass x to g(.)
        output_g = self.func_rawforward_tobecomeGP(x) #[N x Dv x *]
        toret = output_g + 0.0
        if(len(output_g.size()) > 2):
            N, Dv, star = list(output_g.size())[0], list(output_g.size())[1], list(output_g.size())[2:]
            assert(set(star) == set([1])) #TODO:handle * dimensions bigger than 1.
        else:
            N, Dv = list(output_g.size())[0], list(output_g.size())[1]
        output_g = output_g.view(N, Dv) #[N x Dv]
        if(len(list(output_g.size())) == 1):
            output_g = output_g.unsqueeze(0) #this happens when N=1 ==> output_g becomes [Dv] after squeezing
        for n in range((output_g.size())[0]):
            try:
                self._dict_initvalsGPYfromg[list_idxrecurring[n]] = output_g[n,:].cpu().numpy()
                self._set_visitedinstance_resetPGY.add(list_idxrecurring[n])
            except:
                #TODO:check whether the calls actually add some new lements to list_idxrecurring, otherwise raise a warning.
                print("len(list_idxrecurring) = {}".format(len(list_idxrecurring)))
                print("x.size() = {}".format(x.size()))
        return toret
        
    def _forward_initGPXfromf(self, x):
        '''
        When `self.flag_train_memefficient` is set to True, `module_f1` is required to have a function called
        `set_rng_outputheads(.)`.
        '''
        #get indices of last recurring instances ====
        list_idxrecurring = self.func_get_indices_lastrecurringinstances()
        
        #pass x to f1(.)
        if(self.flag_train_memefficient == False):
            output_f = self.module_f1(x) #[N x Du x *]
        else:
            self.module_f1.set_rng_outputheads(self._rng_heads_in_compgragh)
            output_f = self.module_f1(x) #[N x someDu x *]
        if(len(output_f.size()) == 2):
            pass #[NxDu]
        else:
            N = list(x.size())[0]
            num_outputheads = list(output_f.size())[1] #self.Du if(self.flag_train_memefficient == False) else (self._rng_heads_in_compgragh[1] - self._rng_heads_in_compgragh[0])
            output_f = torch.mean(output_f.view(N, num_outputheads, -1), 2) #[NxDu]
        
        
        #set GPX from output_f
        if(self.flag_train_memefficient == False):
            for n in range((output_f.size())[0]):
                self._dict_initvalsGPXfromf[list_idxrecurring[n]] = output_f[n,:].cpu().numpy()
                self._set_visitedinstance_resetGPX.add(list_idxrecurring[n])
        else:
            for n in range((output_f.size())[0]):
                self._dict_initvalsGPXfromf[
                    list_idxrecurring[n] ,
                    self._rng_heads_in_compgragh[0],
                    self._rng_heads_in_compgragh[1]
                ] = output_f[n,:].cpu().numpy()
                self._set_visitedinstance_resetGPX[list_idxrecurring[n]] =\
                    self._set_visitedinstance_resetGPX[list_idxrecurring[n]] +\
                    [j+self._rng_heads_in_compgragh[0] for j in range(self._rng_heads_in_compgragh[1] - self._rng_heads_in_compgragh[0])]
                self._set_visitedinstance_resetGPX[list_idxrecurring[n]] = list(set(self._set_visitedinstance_resetGPX[list_idxrecurring[n]]))
                self._set_visitedinstance_resetGPX[list_idxrecurring[n]].sort()
                self._initGPX_count_visitedelements +=  self._rng_heads_in_compgragh[1] - self._rng_heads_in_compgragh[0]
            
            
        #set toret so that the forward continues ===
        # ~ if(self.flag_train_memefficient == False):
            # ~ toret = (output_f + 0.0).unsqueeze(-1).unsqueeze(-1) #[N x Du x 1 x 1]
        # ~ else:
            # ~ num_excluded_dims = self.Dv - (self._rng_heads_in_compgragh[1] - self._rng_heads_in_compgragh[0])
            # ~ toret = torch.cat([output_f, torch.zeros((output_f.size()[0] , num_excluded_dims), device = output_f.device)], 1) #[NxDu]
            # ~ toret = toret.unsqueeze(-1).unsqueeze(-1) #[N x Du x 1 x 1].
        toret = torch.randn([x.size()[0]]+self._size_output_moduletobecomeGP).to(output_f.device)
        return toret
        
    def initV_from_theannitself(self):
        '''
        This function initializes the GP_Y based on g(.) values at recurring points.
        '''
        self.module_rawmodule.eval()
        with torch.no_grad():
            self._set_visitedinstance_resetPGY = set()
            self._dict_initvalsGPYfromg = {}
            while(self._set_visitedinstance_resetPGY != set(range(self.size_recurringdataset))):
                print("visited {} instances.".format(len(list(self._set_visitedinstance_resetPGY))), end="\r")
                with forward_replaced(self.module_tobecomeGP, self._forward_initGPYfromg):
                    output = self.func_feed_recurring_minibatch()
        
            #renew GPY ====
            newval_GPY = np.array([self._dict_initvalsGPYfromg[n] for n in range(self.size_recurringdataset)]) #[N x Dv]
            self.GP_Y = torch.nn.Parameter(
                    torch.tensor(newval_GPY, requires_grad = True, device=self.device)
                 )#[N x Dv]
        self.module_rawmodule.train()
            
    
    def initU_from_kernelmappings(self):
        '''
        Initializes GPX from module f1.
        '''
        self.module_rawmodule.eval()
        with torch.no_grad():
            self._set_visitedinstance_resetGPX = set() if(self.flag_train_memefficient == False) else {i:[] for i in range(self.size_recurringdataset)}
            if(self.flag_train_memefficient == False):
                _set_toreach_visistedinstance = set(range(self.size_recurringdataset))
            else:
                _set_toreach_visistedinstance = {
                    i:[j for j in range(self.Dv)]
                    for i in range(self.size_recurringdataset)
                }
                self._initGPX_count_visitedelements = 0
                
            
            self._dict_initvalsGPXfromf = {}
            flag_init_finished = False
            count_while = 0
            time_startinit = time.time()
            while(flag_init_finished == False):
                #print the status ====
                count_while += 1
                if(self.flag_train_memefficient == False):
                    print("For GPX, visited {} instances.".format(len(list(self._set_visitedinstance_resetGPX))), end="\r")
                else:
                    if((count_while%50) == 0):
                        #print("reacehd here ....")
                        #num_observedelems = np.sum([len(self._set_visitedinstance_resetGPX[k]) for k in self._set_visitedinstance_resetGPX.keys()])
                        time_epalsedtime = time.time() - time_startinit
                        print(
                            "For GPX, visited {} elements out of {} in {} seconds.".format(
                                self._initGPX_count_visitedelements,
                                self.size_recurringdataset*self.Du,
                                time_epalsedtime
                             ),
                             end='\r'
                        )
                with forward_replaced(self.module_tobecomeGP, self._forward_initGPXfromf):
                    output = self.func_feed_recurring_minibatch()
                if(self.flag_train_memefficient == True):
                    self._inc_rng_headsincompgraph()
                #check if initialization has finished ===
                if(self.flag_train_memefficient == False):
                    flag_init_finished = set(self._set_visitedinstance_resetGPX) == _set_toreach_visistedinstance
                else:
                    #t_beforeflagcheck = time.time()
                    flag_init_finished = (self._set_visitedinstance_resetGPX == _set_toreach_visistedinstance)
                    #t_afterflagcheck = time.time()
                    #print("\n checking if finished took {} seconds.".format(t_afterflagcheck - t_beforeflagcheck))
            #update GPX ====
            if(self.flag_train_memefficient == False):
                newval_GPX = np.array([self._dict_initvalsGPXfromf[n] for n in range(self.size_recurringdataset)]) #[N x Du]
            else:
                newval_GPX = np.zeros((self.size_recurringdataset , self.Du))
                du_per_dim = int(self.Du/self.Dv)
                for k in self._dict_initvalsGPXfromf.keys():
                    i, rng0, rng1 = k
                    newval_GPX[i, du_per_dim*rng0:du_per_dim*rng1] = self._dict_initvalsGPXfromf[k]
                    
            self.GP_X = torch.nn.Parameter( 
                 torch.tensor(newval_GPX, requires_grad = False, device=self.device)
                ) #[N x Du]
        if(self.flag_train_memefficient == True):
            self.module_f1.set_rng_outputheads(rng_outputhead = None) #to avoid manipulating module_f1 for later calls by user.
        self.module_rawmodule.train()
    
    def _getMuCovforQvn(self, x):
        '''
        Passes a tensor of shape [* x Du] to the GP, by detaching model params. 
        '''
        # ~ N, M, h, w = list(output_tail.size())
        local_GPX, local_GPY = self.GP_X.detach(), self.GP_Y.detach()
        xxtplussigma2inverse = self.module_matmulxxtplussigma2.forward(local_GPX, local_GPY) #[size_recurringdataset x 1] self.module_attention_whenbnlikedisabled_head(output_tail) #[Nx1xhxw].
        descriptors_vectorview = x #output_tail.permute(0,2,3,1).reshape((-1, M)) #[Nhw x M]
        mux = torch.matmul(
                    descriptors_vectorview,
                    torch.transpose(local_GPX, 0, 1) 
                 ) #[* x size_recurringdataset]
        mux = torch.matmul(mux , xxtplussigma2inverse) #[* x num_outputheads]
        
        #compute covx ====
        first_term = torch.matmul(descriptors_vectorview, descriptors_vectorview.permute(1,0)) #[Nhw x Nhw]
        second_term = self.module_matmulxxtplussigma2.forward(
                                local_GPX,
                                torch.matmul(local_GPX, descriptors_vectorview.permute(1,0))
                               )#[size_recurringdataset x Nhw]
        second_term = torch.matmul(
                          torch.matmul(descriptors_vectorview, local_GPX.permute(1, 0)),
                          second_term
                       ) #[Nhw x Nhw]
        covx = torch.diagonal(first_term) - torch.diagonal(second_term) #[Nhw]
        #print("     >>>>>>>>>>> min of covx before clamp = {}".format(np.min(covx.detach().cpu().numpy())))
        covx = torch.clamp(covx, min=clamp_min, max=clamp_max)#TODO:check
        covx = covx.unsqueeze(-1) #[Nhw x 1]
        
        #print("mux.shape = {}, cov.shape = {}".format(mux.shape, covx.shape))
        
        #return mux and covx
        return mux.detach(), covx.detach()
    
    
    def _getMuCovforQvhatm(self, x):
        '''
        Passes a tensor of shape [* x Du] to the GP, without detaching GP_Y. 
        '''
        
        # ~ N, M, h, w = list(output_tail.size())
        local_GPX, local_GPY = self.GP_X.detach(), self.GP_Y
        xxtplussigma2inverse = self.module_matmulxxtplussigma2.forward(local_GPX, local_GPY) #[size_recurringdataset x 1] self.module_attention_whenbnlikedisabled_head(output_tail) #[Nx1xhxw].
        descriptors_vectorview = x #output_tail.permute(0,2,3,1).reshape((-1, M)) #[Nhw x M]
        mux = torch.matmul(
                    descriptors_vectorview,
                    torch.transpose(local_GPX, 0, 1) 
                 ) #[* x size_recurringdataset]
        mux = torch.matmul(mux , xxtplussigma2inverse) #[* x num_outputheads]
        
        #compute covx ====
        first_term = torch.matmul(descriptors_vectorview, descriptors_vectorview.permute(1,0)) #[Nhw x Nhw]
        second_term = self.module_matmulxxtplussigma2.forward(
                                local_GPX,
                                torch.matmul(local_GPX, descriptors_vectorview.permute(1,0))
                               )#[size_recurringdataset x Nhw]
        second_term = torch.matmul(
                          torch.matmul(descriptors_vectorview, local_GPX.permute(1, 0)),
                          second_term
                       ) #[Nhw x Nhw]
        covx = torch.diagonal(first_term) - torch.diagonal(second_term) #[Nhw]
        #print("     >>>>>>>>>>> min of covx before clamp = {}".format(np.min(covx.detach().cpu().numpy())))
        covx = torch.clamp(covx, min=clamp_min, max=clamp_max)#TODO:check
        covx = covx.unsqueeze(-1) #[Nhw x 1]
        #return mux and covx
        return mux, covx
        
    def _getMuCovforModelParams(self, x):
        '''
        Passes a tensor of shape [* x Du] to the GP, with detaching GP_Y. 
        '''
        
        #get the part of GPX that must pass the gradient ===
        with forward_replaced(self.module_tobecomeGP, self._forward_setfield_lastGPX_recurring):
            self.func_feed_recurring_minibatch()
        
        #feed x to module_f1 to get the representation in U-space.
        tensor_u = self.module_f1(x) #[N x Du x *]
        tensor_u, N, Du = self._reshapeUorV(tensor_u) #[N*  x Du]
        
        #get indices of last recurring instances ====
        list_idxrecurring = self.func_get_indices_lastrecurringinstances()
        
        #separate the gradpass and detach parts for GPX and GPY
        with torch.no_grad():
            idx_fixed = list(set(range(self.size_recurringdataset)) - set(list_idxrecurring))
            GPX_fixed = self.GP_X[idx_fixed, :].detach() #[n' x M], [n' x num_outputheads]
            GPY_fixed = self.GP_Y[idx_fixed, :].detach() #TODO:check should be detached???
        GPX_gradpass = self.lastGPX_recurring #[minibatch x Du]
        GPY_gradpass = self.GP_Y[list_idxrecurring, :].detach()
        local_GPX = torch.cat([GPX_fixed, GPX_gradpass], 0) #[size_recurringdataset x Du]
        local_GPY = torch.cat([GPY_fixed, GPY_gradpass], 0) #[size_recurringdataset x Dv]
        
        #compute mu ========
        xxtplussigma2inverse = self.module_matmulxxtplussigma2.forward(local_GPX, local_GPY) #[size_recurringdataset x 1] self.module_attention_whenbnlikedisabled_head(output_tail) #[Nx1xhxw].
        toret = torch.matmul(
                    tensor_u,
                    torch.transpose(local_GPX, 0, 1) 
                ) #[Nhw x size_recurringdataset]
        toret = torch.matmul(toret , xxtplussigma2inverse) #[Nstar x Dv]
        toret = toret.view(*([N] +  self.stardimDv +[-1])) #[N x h x w x Dv]
        mux = toret.permute(*([0] + [1+len(self.stardimDv)] + [i+1 for i in range(len(self.stardimDv))])) #[N x num_outputheads x h x w]
        
        
        #compute covx TODO:HERE ====
        covx = torch.Tensor([[1.0]]).float().to(self.device) #---override
        # ~ first_term = torch.matmul(descriptors_vectorview, descriptors_vectorview.permute(1,0)) #[Nhw x Nhw]
        # ~ second_term = self.module_matmulxxtplussigma2.forward(
                                # ~ local_GPX,
                                # ~ torch.matmul(local_GPX, descriptors_vectorview.permute(1,0))
                               # ~ )#[size_recurringdataset x Nhw]
        # ~ second_term = torch.matmul(
                          # ~ torch.matmul(descriptors_vectorview, local_GPX.permute(1, 0)),
                          # ~ second_term
                       # ~ ) #[Nhw x Nhw]
        # ~ covx = torch.diagonal(first_term) - torch.diagonal(second_term) #[Nhw]
        # ~ #print("     >>>>>>>>>>> min of covx before clamp = {}".format(np.min(covx.detach().cpu().numpy())))
        # ~ covx = torch.clamp(covx, min=1.0, max=1.0)#TODO:check
        # ~ covx = covx.unsqueeze(-1) #[Nhw x 1]
        return mux, covx
    
    
    def _forward_makecostModelParams(self, x):
        
        '''
        The forwrad when making cost for model parameters.
        This forward must be called when feeding non-recurring instnaces.
        '''
        #generate samples from Qvn ====
        with torch.no_grad():
            muqvn = self.func_rawforward_tobecomeGP(x) #[N x C x 1 x 1]
            originalsize_vn = list(muqvn.size())
            muqvn = muqvn.squeeze() #[N x C]
            
            #print("muqvn.shape = {}".format(muqvn.shape))
            Z = torch.normal(mean=0.0, std=1.0, size=list(muqvn.size())).to(self.device) #[Nhw x num_outputheads]
            generated_vn = muqvn + torch.sqrt(self.cov_qvn)*Z #[* x Dv]
        #generated_vn = generated_vn.detach()
        
        
        #feed a recurring mini-batch
        with forward_replaced(self.module_tobecomeGP, self._forward_setfield_lastGPX_recurring):
            self.func_feed_recurring_minibatch()
        
        
        #separate the gradpass and detach parts for GPX and GPY
        list_idxrecurring = self.func_get_indices_lastrecurringinstances()
        with torch.no_grad():
            idx_fixed = list(set(range(self.size_recurringdataset)) - set(list_idxrecurring))
            GPX_fixed = self.GP_X[idx_fixed, :].detach() #[n' x M], [n' x num_outputheads]
            GPY_fixed = self.GP_Y[idx_fixed, :].detach() #TODO:check should be detached???
        GPX_gradpass = self.lastGPX_recurring #[minibatch x Du]
        GPY_gradpass = self.GP_Y[list_idxrecurring, :].detach()
        local_GPX = torch.cat([GPX_fixed, GPX_gradpass], 0) #[size_recurringdataset x Du]
        local_GPY = torch.cat([GPY_fixed, GPY_gradpass], 0) #[size_recurringdataset x Dv]
        
        #compute mu() and cov() according to GP and by detaching GP_Y====
        if(self.flag_efficient == True):
            dict_args_efficiency = {
                 "mode":"allowgrad",
                 "precomputed_XTX":self.precomputed_XTX,
                 "idx_in_inputarg":[list(GPX_fixed.size())[0] , list(GPX_fixed.size())[0]+list(GPX_gradpass.size())[0]],
                 "idx_in_globalGPX":list_idxrecurring,
                 "global_GPX":self.GP_X
            }
        else:
            dict_args_efficiency = None
        mu_pvn, cov_pvn = self._forwardGP(self.module_f1(x), local_GPX, local_GPY, dict_args_efficiency=dict_args_efficiency) #self._getMuCovforModelParams(x)
        if(self.flag_detachcovpvn == True):
            cov_pvn = cov_pvn.detach()
            #print("cov pvn was detached")
        else:
            pass
            #print("cov pvn was NOT detached.")
        if(self.flag_setcovtoOne == True):
            cov_pvn = torch.Tensor([1.0]).to(self.device)
        else:
            pass
        
        
        
        
        #compute term1
        term1 = NormalUtils.loglikelihood_1D(muqvn.detach(), mu_pvn, cov_pvn)
        
        if(self.flag_controlvariate == True):
            term1 = self.controlvariate(term1, (mu_pvn-muqvn)*(mu_pvn-muqvn))
        
        self._cost_GPmatchNN_term1 =  -torch.mean(torch.sum(term1, 1))
        
        return torch.reshape(generated_vn, originalsize_vn)
    
    
    
    def _forward_makecostModelParams_trainmemefficient(self, x):
        
        '''
        The forwrad when making cost for model parameters.
        This forward must be called when feeding non-recurring instnaces.
        Also this version of this function must be called only if `self.flag_train_memefficient` is set to True.
        '''
        assert(self.flag_train_memefficient == True)
        #generate samples from Qvn ====
        if(True):#with torch.no_grad():
            muqvn = self.func_rawforward_tobecomeGP(x) #[N x Dv x 1 x 1]
            originalsize_vn = list(muqvn.size())
            muqvn = muqvn.squeeze() #[N x Dv]
            muqvn = muqvn[: , self._rng_heads_in_compgragh[0]:self._rng_heads_in_compgragh[1]] #[N x Dvsubset]
            
            #print("muqvn.shape = {}".format(muqvn.shape))
            Z = torch.normal(mean=0.0, std=1.0, size=list(muqvn.size())).to(self.device) #[Nhw x Dvsubset]
            generated_vn = muqvn + torch.sqrt(self.cov_qvn)*Z #[* x Dvsubset]
        #generated_vn = generated_vn.detach()
        
        
        #feed a recurring mini-batch
        with forward_replaced(self.module_tobecomeGP, self._forward_setfield_lastGPX_recurring):
            self.func_feed_recurring_minibatch()
        #now self.lastGPX_recurring is [N x Dusubset]
        
        
        #separate the gradpass and detach parts for GPX and GPY
        list_idxrecurring = self.func_get_indices_lastrecurringinstances()
        du_per_gp = int(self.Du/self.Dv)
        with torch.no_grad():
            idx_fixed = list(set(range(self.size_recurringdataset)) - set(list_idxrecurring))
            GPX_fixed = self.GP_X[idx_fixed, du_per_gp*self._rng_heads_in_compgragh[0]:du_per_gp*self._rng_heads_in_compgragh[1]].detach() #[n' x M], [n' x num_outputheads]
            GPY_fixed = self.GP_Y[idx_fixed, self._rng_heads_in_compgragh[0]:self._rng_heads_in_compgragh[1]].detach() #TODO:check should be detached???
        GPX_gradpass = self.lastGPX_recurring #[minibatch x Dusubset]
        GPY_gradpass = self.GP_Y[list_idxrecurring, self._rng_heads_in_compgragh[0]:self._rng_heads_in_compgragh[1]].detach()
        local_GPX = torch.cat([GPX_fixed, GPX_gradpass], 0) #[size_recurringdataset x Dusubset]
        local_GPY = torch.cat([GPY_fixed, GPY_gradpass], 0) #[size_recurringdataset x Dvsubset]
        
        #compute mu() and cov() according to GP and by detaching GP_Y====
        if(self.flag_efficient == True):
            dict_args_efficiency = {
                 "mode":"allowgrad",
                 "precomputed_XTX":self.precomputed_XTX,
                 "idx_in_inputarg":[list(GPX_fixed.size())[0] , list(GPX_fixed.size())[0]+list(GPX_gradpass.size())[0]],
                 "idx_in_globalGPX":list_idxrecurring,
                 "global_GPX":self.GP_X
            }
        else:
            dict_args_efficiency = None
        mu_pvn, cov_pvn = self._forwardGP(self.module_f1(x), local_GPX, local_GPY, dict_args_efficiency=dict_args_efficiency) #self._getMuCovforModelParams(x)
        if(self.flag_detachcovpvn == True):
            cov_pvn = cov_pvn.detach()
            #print("cov pvn was detached")
        else:
            pass
            #print("cov pvn was NOT detached.")
        if(self.flag_setcovtoOne == True):
            cov_pvn = torch.Tensor([1.0]).to(self.device)
        else:
            pass
        
        
        
        
        #compute term1
        term1 = NormalUtils.loglikelihood_1D(muqvn.detach(), mu_pvn, cov_pvn)
        
        if(self.flag_controlvariate == True):
            term1 = self.controlvariate(term1, (mu_pvn-muqvn)*(mu_pvn-muqvn))
        
        self._cost_GPmatchNN_term1 =  -torch.mean(torch.sum(term1, 1))
        
        return torch.reshape(generated_vn, originalsize_vn)
    
    
    
    
    def _forward_makecostGPmatchNN(self, x):
        
        '''
        The forwrad when making cost for GP-match-ANN.
        This forward must be called when feeding non-recurring instnaces.
        The input `x` has to be a tensor of shape [N x *] where * is any additional dimensions.
        '''
        #generate samples from Qvn ====
        with torch.no_grad():
            muqvn = self.func_rawforward_tobecomeGP(x) #[N x *]
            dim_2 = np.prod(list(muqvn.size())[1::])
            muqvn = muqvn.view(-1, dim_2)
            #muqvn = muqvn.squeeze() #[N x C]
            
            #print("muqvn.shape = {}".format(muqvn.shape))
            Z = torch.normal(mean=0.0, std=1.0, size=list(muqvn.size())).to(self.device) #[Nhw x num_outputheads]
            generated_vn = muqvn + torch.sqrt(self.cov_qvn)*Z #[* x Dv]
        generated_vn = generated_vn.detach()
        
        
        #feed a recurring mini-batch
        with forward_replaced(self.module_tobecomeGP, self._forward_setfield_lastGPX_recurring):
            self.func_feed_recurring_minibatch()
        
        
        #separate the gradpass and detach parts for GPX and GPY
        list_idxrecurring = self.func_get_indices_lastrecurringinstances()
        with torch.no_grad():
            idx_fixed = list(set(range(self.size_recurringdataset)) - set(list_idxrecurring))
            GPX_fixed = self.GP_X[idx_fixed, :].detach() #[n' x M], [n' x num_outputheads]
            GPY_fixed = self.GP_Y[idx_fixed, :].detach() #TODO:check should be detached???
        GPX_gradpass = self.lastGPX_recurring #[minibatch x Du]
        GPY_gradpass = self.GP_Y[list_idxrecurring, :].detach()
        local_GPX = torch.cat([GPX_fixed, GPX_gradpass], 0) #[size_recurringdataset x Du]
        local_GPY = torch.cat([GPY_fixed, GPY_gradpass], 0) #[size_recurringdataset x Dv]
        
        #compute mu() and cov() according to GP and by detaching GP_Y====
        if(self.flag_efficient == True):
            dict_args_efficiency = {
                 "mode":"allowgrad",
                 "precomputed_XTX":self.precomputed_XTX,
                 "idx_in_inputarg":[list(GPX_fixed.size())[0] , list(GPX_fixed.size())[0]+list(GPX_gradpass.size())[0]],
                 "idx_in_globalGPX":list_idxrecurring,
                 "global_GPX":self.GP_X
            }
        else:
            dict_args_efficiency = None
        
        mu_pvn, cov_pvn = self._forwardGP(
                            self.module_f1(x),
                            local_GPX,
                            local_GPY,
                            dict_args_efficiency=dict_args_efficiency
                        ) #self._getMuCovforModelParams(x)
        
        cov_pvn = cov_pvn.detach() #TODO:check cov_pvn = torch.ones_like(cov_pvn).to(self.device)#TODO:check cov_pvn.detach() #TODO:check
        
        
        #compute term1
        term1 = NormalUtils.loglikelihood_1D(muqvn.detach(), mu_pvn, cov_pvn)
        self._cost_GPmatchNN_term1 =  -torch.mean(torch.sum(term1, 1))
        
        return generated_vn
        
        
    def _forward_makcostQvhatm(self, x):
        '''
        The forward when making cost for Qvhatm.
        This forward must be called when feeding non-recurring instances.
        '''
        #generate samples from Qvn ====
        with torch.no_grad():
            muqvn = self.func_rawforward_tobecomeGP(x)
            Z = torch.normal(mean=0.0, std=1.0, size=list(muqvn.size())).to(self.device) #[Nhw x num_outputheads]
            generated_vn = muqvn + torch.sqrt(self.cov_qvn)*Z #[* x Dv]
        generated_vn = generated_vn.detach()
        
        #compute mu() and cov(.) according to GP ====
        mu_pvn, cov_pvn = self._getMuCovforQvhatm(self._reshapeUorV(self.module_f1(x.detach()))[0])
        
        term_1 = NormalUtils.loglikelihood_1D(generated_vn, mu_pvn, cov_pvn)#[Nhwxnum_outputheads]
        term_1 = torch.sum(term_1, 1) #[Nhx]
        self._costQvhatm_term1 = -torch.mean(term_1)
        
        #make the KL-term (i.e. term2) =======
        klterm = torch.sum(self.GP_Y*self.GP_Y, 1)*0.0/(1000.0)#TODO:check NormalUtils.KLdiv_lowranksigma2(
                        # ~ mu1 = self.GP_Y.permute(1,0)+0.0,
                        # ~ diagsigma1 = self.cov_vhat,
                        # ~ x_covar2 = self.GP_X,
                        # ~ module_matmulcovar2 = self.module_matmulcovaronGPX
                      # ~ )
        self._costQvhatm_klterm = torch.mean(klterm)
        
        return generated_vn
        
    
    def _forward_feedonnoise_makecostQvn(self, x):
        '''
        This forward is called within `_forward_makecostQvn` and with input noise.
        This forward build `self._cost_Qvn_klpart`.
        '''
        #pass the nosie x to g(.) ===
        mu_qvn = self.func_rawforward_tobecomeGP(x) #[N x C x 1 x 1]
        if(len(list(mu_qvn.size())) == 2):
            mu_qvn = mu_qvn.unsqueeze(-1).unsqueeze(-1) #to assert [N x C x 1 x 1]
        #muqvn = muqvn.squeeze() #[N x C]
        
        #pass the noise x to f(.) and GP ====
        local_GPX = self.GP_X.detach() #[size_recurringdataset x Du]
        local_GPY = self.GP_Y.detach() #[size_recurringdataset x Dv]
        if(self.flag_efficient == True):
            dict_args_efficiency = {
                 "mode":"detachGP",
                 "precomputed_XTX":self.precomputed_XTX
            }
        else:
            dict_args_efficiency = None
        mu_pvn, cov_pvn = self._forwardGP(self.module_f1(x), local_GPX, local_GPY, dict_args_efficiency=dict_args_efficiency) #self._getMuCovforModelParams(x)
        mu_pvn = mu_pvn.detach()
        cov_pvn = cov_pvn.detach()
        
        
        
        #compute the self._cost_Qvn_klpart ====
        N, _, h, w = list(mu_qvn.size())
        mu_qvn = mu_qvn.permute(0,2,3,1) #[N x h x w x num_outputheads]
        mu_qvn = mu_qvn.reshape((-1, self.Dv)) #[Nhw x num_outputheads]
        cov_qvn = self.cov_qvn
        
        #compute the kl-divergence ===
        toret_KL = NormalUtils.KLdiv_twonormals_withdiagonalcovar(
                        mu1 = mu_qvn,
                        diagsigma1 = cov_qvn,
                        mu2 = mu_pvn,
                        diagsigma2 = cov_pvn
                    ) #[Nhw]
        self._cost_Qvn_klpart = toret_KL
        return mu_qvn.detach()
        
        
    
    def _forward_makecostQvn(self, x):
        '''
        The forward make the cost for Qvn.
        In this forward, for a single non-recurring batch the vn representations must be computed in two ways:
            1. via g(.)
            2. via f1(.) and GP.
        '''
        #generate samples from Qvn (to be used by, e.g., the cross-entropy loss) ====
        muqvn = self.func_rawforward_tobecomeGP(x) #[N x C x 1 x 1]
        muqvn = muqvn.squeeze() #[N x C]
        Z = torch.normal(mean=0.0, std=1.0, size=list(muqvn.size())).to(self.device) #[Nhw x num_outputheads]
        generated_vn = muqvn + torch.sqrt(self.cov_qvn)*Z #[* x Dv]
        
        
        #feed a noise mini-batch
        with forward_replaced(self.module_tobecomeGP, self._forward_feedonnoise_makecostQvn):
            _ = self.func_feed_noise_minibatch()
            #now self._cost_Qvn_klpart is ready.
        
        return generated_vn
        
        # ~ #compute via g(.) ====
        # ~ #print("x.shape = {}".format(x.shape))
        # ~ self._muqvn_forQvncost = self.func_rawforward_tobecomeGP(x)
        # ~ #print("    self._muqvn_forQvncost.shape = {}".format(self._muqvn_forQvncost.shape))
        
        # ~ #compute via f1(.) and GP
        # ~ with torch.no_grad():
            # ~ #feed x to module_f1 to get the representation in U-space.
            # ~ tensor_u = self.module_f1(x) #[N x Du x *]
            # ~ tensor_u, N, Du = self._reshapeUorV(tensor_u) #[N*  x Du]
            # ~ self._mupvn_forQvncost, self._covpvn_forQvncost = self._getMuCovforQvn(tensor_u)
            
        
        # ~ #generate a sample from q(vn) and reuturn it as foward's output ====
        # ~ if(len(list(self._muqvn_forQvncost.size())) == 2):
            # ~ self._muqvn_forQvncost = self._muqvn_forQvncost.unsqueeze(-1).unsqueeze(-1)
        # ~ Z = torch.normal(mean=0.0, std=1.0, size=list(self._muqvn_forQvncost.size())).to(self.device) #[Nhw x num_outputheads]
        # ~ generated_vn = self._muqvn_forQvncost + torch.sqrt(self.cov_qvn)*Z #[Nhw x num_outputheads]
        # ~ return generated_vn
        
    
    def _forward_dummy(self, x):
        '''
        Forwards x as if GPX and GPY are random number.
        '''
        #feed x to module_f1 to get the representation in U-space.
        tensor_u = self.module_f1(x) #[N x Du x *]
        tensor_u, N, Du = self._reshapeUorV(tensor_u) #[N*  x Du]
        
        with torch.no_grad():
            N = list(x.size())[0]
            #print(" >>> reached here 3")
            #print("N = {}".format(N))
            toret = torch.zeros((N, self.Dv, 1, 1)).float().to(self.device)
            
        
        return toret
    
    
    def _forward_setfield_lastGPX_recurring(self, x):
        '''
        This forward is called when some elements of GPX related to f1(.) (which has to backprop the gradient) are needed.
        '''
        #print("     >>> reached here 1.")
        #feed x to module_f1 to get the representation in U-space.
        tensor_u = self.module_f1(x) #[N x Du x *]
        tensor_u, N, Du = self._reshapeUorV(tensor_u) #[N*  x Du]
        #print("     >>> reached here 2.")
        #print("      tensor_u.shape = {}".format(tensor_u.shape))
        #print("       self.func_get_indices_lastrecurringinstances() = {}".format(self.func_get_indices_lastrecurringinstances()))
        self.lastGPX_recurring = tensor_u#[self.func_get_indices_lastrecurringinstances(), :]
        
        
        
        with torch.no_grad():
            N = list(x.size())[0]
            #print(" >>> reached here 3")
            #print("N = {}".format(N))
            toret = torch.zeros((N, self.Dv, 1, 1)).float().to(self.device)
            #print(" >>> returned 4")
            return toret
            
         
    
    def _forward_testingtime_withoutuncertainty(self, x):
        '''
        TODO:adddoc.
        '''
        #get the part of GPX that must pass the gradient ===
        #with forward_replaced(self.module_tobecomeGP, self._forward_setfield_lastGPX_recurring):
        #    self.func_feed_recurring_minibatch()
        
        if(self.flag_train_memefficient == False):
            #feed x to module_f1 to get the representation in U-space.
            tensor_u = self.module_f1(x) #[N x Du x *]
            tensor_u, N, Du = self._reshapeUorV(tensor_u) #[N*  x Du]
            
            #get indices of last recurring instances ====
            local_GPX = self.GP_X #torch.cat([GPX_fixed, GPX_gradpass], 0) #[size_recurringdataset x Du]
            local_GPY = self.GP_Y #torch.cat([GPY_fixed, GPY_gradpass], 0) #[size_recurringdataset x Dv]
            
            
            mu_pvn, cov_pvn, similarities = self._forwardGP(self.module_f1(x), local_GPX, local_GPY, flag_clampcov=False, flag_returnsimilarites=True, dict_args_efficiency=None) #self._getMuCovforModelParams(x)
            mu_pvn = mu_pvn.unsqueeze(-1).unsqueeze(-1) #[N x num_outputheads x 1 x 1]
            #print(">>>>>> mu_pvn.shape = {}".format(mu_pvn.shape))
            
            #make similarities and uncertainties =====
            self._testingtime_uncertainties = cov_pvn.detach().cpu().numpy()
            self._testingtime_similarities = similarities #toret_similarities.detach().cpu().numpy()
            return mu_pvn
        else:
            mu_pvn, testingtime_uncertainties, testingtime_similarities = [], [], []
            du_per_gp = int(self.Du/self.Dv)
            for idx_gp in range(self.Dv):
                #find tensor_u of this GP ===
                self._set_rng_headsincompgraph(input_rng = [idx_gp, idx_gp+1])
                tensor_u_ofgp_origshape = self.module_f1(x) #[N x Dusubset x *]
                tensor_u_ofgp, N, _ = self._reshapeUorV(tensor_u_ofgp_origshape) #[N*  x Dusubset]
                
                #get indices of last recurring instances ====
                local_GPX = self.GP_X #torch.cat([GPX_fixed, GPX_gradpass], 0) #[size_recurringdataset x Du]
                local_GPY = self.GP_Y #torch.cat([GPY_fixed, GPY_gradpass], 0) #[size_recurringdataset x Dv]
                
                #find mu, of this GP ==
                mu_pvn_ofgp, cov_pvn_ofgp, similarities_ofgp = self._forwardGP(tensor_u_ofgp_origshape, local_GPX, local_GPY, flag_clampcov=False, flag_returnsimilarites=True, dict_args_efficiency=None) #self._getMuCovforModelParams(x)
                mu_pvn_ofgp = mu_pvn_ofgp.unsqueeze(-1).unsqueeze(-1) #[N x num_outputheads x 1 x 1]
                
                #grab the results of gp into the global lists ===
                mu_pvn.append(mu_pvn_ofgp)
                testingtime_uncertainties.append(cov_pvn_ofgp.detach().cpu().numpy())
                testingtime_similarities.append(similarities_ofgp)
            
            self._testingtime_uncertainties = np.array([u.flatten().tolist()[0] for u in testingtime_uncertainties])
            self._testingtime_similarities = np.array([u.flatten().tolist()[0] for u in testingtime_similarities])
            return mu_pvn
            
            
            
            
        
        
    
        
        
    def _forward_GPwithoutuncertainty(self, x):
        '''
        TODO:adddoc.
        '''
        #get the part of GPX that must pass the gradient ===
        with forward_replaced(self.module_tobecomeGP, self._forward_setfield_lastGPX_recurring):
            self.func_feed_recurring_minibatch()
        
        #feed x to module_f1 to get the representation in U-space.
        tensor_u = self.module_f1(x) #[N x Du x *]
        tensor_u, N, Du = self._reshapeUorV(tensor_u) #[N*  x Du]
        
        #get indices of last recurring instances ====
        list_idxrecurring = self.func_get_indices_lastrecurringinstances()
        
        #separate the gradpass and detach parts for GPX and GPY
        # ~ with torch.no_grad():
            # ~ idx_fixed = list(set(range(self.size_recurringdataset)) - set(list_idxrecurring))
            # ~ GPX_fixed = self.GP_X[idx_fixed, :].detach() #[n' x M], [n' x num_outputheads]
        # ~ GPY_fixed = self.GP_Y[idx_fixed, :]#.detach() #TODO:check should be detached???
        # ~ GPX_gradpass = self.lastGPX_recurring #[minibatch x Du]
        # ~ GPY_gradpass = self.GP_Y[list_idxrecurring, :]
        local_GPX = self.GP_X.detach() #torch.cat([GPX_fixed, GPX_gradpass], 0) #[size_recurringdataset x Du]
        local_GPY = self.GP_Y.detach() #torch.cat([GPY_fixed, GPY_gradpass], 0) #[size_recurringdataset x Dv]
        
        mu_pvn, _ = self._forwardGP(self.module_f1(x), local_GPX, local_GPY, dict_args_efficiency=None)
        
        # ~ #compute the attention_mask ========
        # ~ xxtplussigma2inverse = self.module_matmulxxtplussigma2.forward(local_GPX, local_GPY) #[size_recurringdataset x 1] self.module_attention_whenbnlikedisabled_head(output_tail) #[Nx1xhxw].
        # ~ toret = torch.matmul(
                    # ~ tensor_u,
                    # ~ torch.transpose(local_GPX, 0, 1) 
                # ~ ) #[Nhw x size_recurringdataset]
        # ~ toret = torch.matmul(toret , xxtplussigma2inverse) #[Nstar x Dv]
        # ~ toret = toret.view(*([N] +  self.stardimDv +[-1])) #[N x h x w x Dv]
        # ~ toret = toret.permute(*([0] + [1+len(self.stardimDv)] + [i+1 for i in range(len(self.stardimDv))])) #[N x num_outputheads x h x w]
        return mu_pvn
        
    def _forward_inferDu(self, x):
        '''
        The forwardf of the module will temporarily replaced by this fucntion.
        '''
        #pass x to both module_g and module_f1 ====
        output_f1 = self.module_f1(x)
        output_g = self.func_rawforward_tobecomeGP(x)
        
        #print(" output_f1.shape = {}".format(output_f1.shape))
        NCstar_of_u = list(output_f1.size())
        self.Du = NCstar_of_u[1]
        #print("       Du was set to {}".format(self.Du))
        return output_g
    
    def _forward_inferDv(self, x):
        '''
        The forward of the module will temporarily be replaced by this forward fucntion.
        '''
        #print("x.shape = {}".format(x.shape))
        print("x.shape = {}".format(x.shape))
        output = self.func_rawforward_tobecomeGP(x)
        #print("output.shape = {}".format(output.shape))
        
        #infer and grab Dv here ====
        NCstar_of_v = list(output.size())
        self.Dv = NCstar_of_v[1]
        self.stardimDv = NCstar_of_v[2:]
        print("    Dv was set to {}".format(self.Dv))
        return output
    
    def _infer_outputsize_mdouletobecomeGP(self):
        '''
        After calling this function, the field self._size_output_moduletobecomeGP will be set to, e.g., [C x h*w x 1].
        '''
        self.eval()
        with torch.no_grad():
            #output size of module_tobecomeGP
            with forward_replaced(self.module_tobecomeGP, self._forward_inferOutputsizeModuletobecomeGP):
                output = self.func_feed_recurring_minibatch()
        self.train()
    
    
    def _inferDvDv(self):
        '''
        Infers Dv (with notation of Fig.1 of paper).
        '''
        self.eval()
        with torch.no_grad():
            #infer Dv
            with forward_replaced(self.module_tobecomeGP, self._forward_inferDv):
                output = self.func_feed_recurring_minibatch()
            
            #infer Du
            with forward_replaced(self.module_tobecomeGP, self._forward_inferDu):
                output = self.func_feed_recurring_minibatch()
            if(self.flag_train_memefficient == True):
                num_outputheads = self._rng_heads_in_compgragh[1] - self._rng_heads_in_compgragh[0]
                #TODO:raise an excpetion if the current Du is not divisible by num_outputheads.
                Du_per_f1module = self.Du/num_outputheads
                print("self.Du = {}".format(self.Du))
                print("num_outputheads = {}".format(num_outputheads))
                print("Du_per_f1module = {}".format(Du_per_f1module))
                self.Du = int(self.Dv * Du_per_f1module)
        self.train()
        
        
        
        
        
class NormalUtils:
    '''
    Utitlity functions like log-likelihood, etc. for normal distribution.
    '''
    def __init__(self):
        pass
    
    
    @staticmethod
    def loglikelihood_1D(x, mu, cov):
        '''
        Returns the log-likelihood of 1-D normal distrbiution for a batch of inputs.
        Inputs must have broadcastable shapes.
        Inputs.
            - x: the values.
            - mu: the means (s).
            - cov: the covatiance (s).
        Output.
            - a tensor of the same shape as `x`.
        '''
        toret = -0.5*((x-mu)*(x-mu))/cov - 0.5*torch.log(cov)
        with torch.no_grad():
            np_xminmu = ((x-mu)).detach().cpu().numpy()
            #print("||x-mu||_2 in range [{},{}]".format(np.min(np_xminmu), np.max(np_xminmu)))
            #print(">>>> term 1 = {}".format(torch.mean((((x-mu)*(x-mu))/cov)).detach().cpu().numpy()))
            #print(">>>>>> term 2 = {}".format(torch.mean(torch.log(cov))))
        return toret
        
    @staticmethod
    def KLdiv_twonormals_withdiagonalcovar(mu1, diagsigma1, mu2, diagsigma2):
        '''
        Computes the KL-divergence between two batches of normal distributions.
        All normal distributions must have diagonal covariance matrices.
        Inputs.
            - mu1, mu2: means of the first and the second batch, two tensor of shape [NxD].
            - diagsigma1, diagsigma2: the diagonal of the covariance matrices of the first and the second batch, two tensors of shape [NxD].
        '''
        #compute term 1 ========
        logdet1 = torch.sum(torch.log(diagsigma1), 1) #[N]
        logdet2 = torch.sum(torch.log(diagsigma2), 1) #[N]
        term_1 = 0.0 #TODO:checklogdet2 - logdet1 #[N]
        
        #compute term2 =====
        term_2 = 0.0 #TODO:check not dependant on varpars -(list(mu1.size())[1])
        
        #compute term3 ====
        term_3 =  0.0#TODO:check torch.sum(diagsigma1/diagsigma2, 1) #[N]
        
        #compute term4 ====
        term_4 = ((mu2 - mu1)*(mu2 - mu1))/(diagsigma2) #[NxD] #TODO:check
        term_4 = torch.sum(term_4, 1) #[N]
        
        #return the result
        toret =  term_1 + term_2 + term_3 + term_4  #[N]
        with torch.no_grad():
            np_term4 = term_4.detach().cpu().numpy()
            np_mu2minmu1 = (mu2-mu1).detach().cpu().numpy()
            np_diagsigma2 = diagsigma2.detach().cpu().numpy()
            #print(" >>>>>>>>>>>> term 4 in range [{},{}]".format(np.min(np_term4), np.max(np_term4)))
            #print(" >>>>>>>>>>>>>>>>>>> mu2-mu1 in range [{},{}]".format(np.min(np_mu2minmu1), np.max(np_mu2minmu1)))
            #print(" >>>>>>>>>>>>>>>>>>> disgsigma2 in range [{},{}]".format(np.min(np_diagsigma2), np.max(np_diagsigma2)))
            
        # ~ print("    terms 1-4 = [{} , {} , {} , {}]".format(
                    # ~ term_1.detach().cpu().numpy(),
                    # ~ term_2,
                    # ~ term_3.detach().cpu().numpy(),
                    # ~ term_4.detach().cpu().numpy() 
                # ~ )
            # ~ )
            
        # ~ print(" ------ range of diagsigma2 = [{} , {}]".format(
                        # ~ np.min(diagsigma2.detach().cpu().numpy()),
                        # ~ np.max(diagsigma2.detach().cpu().numpy())
                    # ~ )
            # ~ )
        return toret
        
    @staticmethod
    def KLdiv_lowranksigma2(mu1, diagsigma1, x_covar2, module_matmulcovar2):
        '''
        Computes the KL-divergence between two batches of normal distributions.
        The first normal distribution has a diagonal covariance matrix.
        The second normal distrbution has a [MxM] but low-rank covariance matrix. 
        Inputs.
            - mu1: mean of the first normal distribution, tensor of shape [DxM].
            - diagsigma1: the diagonal of the covariance matrices of the first batch, a scalar.
            - x_covar2: the matrices that form the second covariance matrix, tensor of shape [M x n'].
        '''
        sigma2invmu1, logdet_sigma2, trace_sigma2inv = \
                module_matmulcovar2.forward(x_covar2, mu1.permute(1,0), flag_getlogdetmat1=True) #[MxD], scalar, and scalar
        
        #compute term_1 ====
        D = list(mu1.size())[1]
        term_1 = 0.0#TODO:checklogdet_sigma2 - np.log(diagsigma1) #scalar
        
        #copmute term_2 ===
        term_2 = 0.0 
        
        #compute term_3 ====
        term_3 =  0.0#TODO:checkdiagsigma1 * trace_sigma2inv #scalar
        
        #compute term_4 ====
        term_4 = torch.matmul(mu1, sigma2invmu1) #[DxD]
        term_4 = 1.0*torch.diagonal(term_4) #---override  completely delete the regularization on Qvhat.mu #[D]
        
        #return the result ===
        toret = term_1 + term_2 + term_3 + term_4
        
        
        # ~ print("    terms 1-4 = [{} , {} , {} , {}]".format(
                    # ~ term_1.detach().cpu().numpy(),
                    # ~ term_2,
                    # ~ term_3.detach().cpu().numpy(),
                    # ~ term_4.detach().cpu().numpy() 
                # ~ )
            # ~ )
        
        return toret
        


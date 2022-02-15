
import pickle
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def unpickle(file):
    with open(file, 'rb') as fo:
        dict_toret = pickle.load(fo, encoding='bytes')
    return dict_toret

class ImgnetDenormalize(object):
    def __init__(self):
        self.tfm1 = torchvision.transforms.Normalize(mean=[0.0, 0.0, 0.0],\
                                                     std=[1.0/0.229, 1.0/0.224, 1.0/0.225])
        self.tfm2 = torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406],\
                                                     std=[1.0, 1.0, 1.0])
    def __call__(self, x):
        if(len(list(x.size()))==3):
            #[CxHxW] case
            toret = self.tfm2(self.tfm1(x))
        else:
            #[N x C x H x W] case
            N = list(x.size())[0]
            toret = [self.tfm2(self.tfm1(x[n,:,:,:])) for n in range(N)]
            return torch.stack(toret)
        return toret

class Cifar10Dataset(Dataset):
    '''
    This class implements the Dataset for loading MYCIHC WSI images.
    '''
    def __init__(self, rootdir, fname_batchfile, str_trainoreval, flag_enabledataaugmentation = False, flag_loadalltraining = False, list_subsamplelabels = None,\
                 fname_flippedlabels = None, list_correctedlabels_byuser = None):
        '''
        dsdfsdf
        '''
        #grab args====
        self.rootdir = rootdir
        self.fname_batchfile = fname_batchfile
        self.str_trainoreval = str_trainoreval
        self.flag_enabledataaugmentation = flag_enabledataaugmentation
        if(self.flag_enabledataaugmentation == True):
            assert(self.str_trainoreval == "train")
        assert(self.str_trainoreval in ["train", "eval"])
        self.list_subsamplelabels = list_subsamplelabels
        if(self.list_subsamplelabels is not None):
            assert(len(self.list_subsamplelabels) == 2)
            assert(self.list_subsamplelabels[1] > self.list_subsamplelabels[0])
        self.list_correctedlabels_byuser = list_correctedlabels_byuser
        #read the batchfile ====
        if(flag_loadalltraining == False):
            self.dict_batch = unpickle(os.path.join(self.rootdir, self.fname_batchfile))
            self.dict_labelnames = unpickle(os.path.join(self.rootdir, "batches.meta"))
            #read X and Y ====
            self.Xraw = self.dict_batch[b'data']
            self.Y = self.dict_batch[b'labels']
            self.label_names = self.dict_labelnames[b'label_names']
        else:
            self.dict_labelnames = unpickle(os.path.join(self.rootdir, "batches.meta"))
            self.Xraw = []
            self.Y = []
            for idx_split in range(5):
                dict_batch = unpickle(os.path.join(self.rootdir, "data_batch_{}".format(idx_split+1)))
                self.Xraw.append(
                    dict_batch[b'data']
                )
                self.Y = self.Y + dict_batch[b'labels']
            self.Xraw = np.concatenate(self.Xraw, 0) #[50000 x D]
            
        #select specific classes if needed ===
        if(self.list_subsamplelabels is not None):
            idx_in_class_0 = np.where(np.array(self.Y) == self.list_subsamplelabels[0])[0].tolist()
            idx_in_class_1 = np.where(np.array(self.Y) == self.list_subsamplelabels[1])[0].tolist()
            idx_in_class_01 = idx_in_class_0 + idx_in_class_1
            idx_in_class_01.sort()
            print(
                "Only instances in the following classes were selected: {} and {}".format(
                    self.dict_labelnames[b'label_names'][self.list_subsamplelabels[0]],
                    self.dict_labelnames[b'label_names'][self.list_subsamplelabels[1]]
                )
            )
            original_N = self.Xraw.shape[0]
            self.Xraw = self.Xraw[idx_in_class_01, :]
            self.Y = np.array(self.Y)[idx_in_class_01].tolist()
            print("Total number of selected instances = {} out of {}.".format(self.Xraw.shape[0], original_N))
            self.idx_in_class_01 = idx_in_class_01
            assert(set(self.Y) == set(self.list_subsamplelabels))
            
        #flip labels if needed ====
        if(fname_flippedlabels is not None):
            #load newY
            assert(self.list_subsamplelabels is not None)
            file_flippedlabels = open(fname_flippedlabels,'rb')
            newY = pickle.load(file_flippedlabels)
            assert(len(newY) == len(self.Y))
            
            
            #print status of the flipped labels ====
            num_flips_0to1 = np.sum(  (np.array(self.Y) == self.list_subsamplelabels[0]) * (np.array(newY) == self.list_subsamplelabels[1]) )
            num_flips_1to0 = np.sum(  (np.array(self.Y) == self.list_subsamplelabels[1]) * (np.array(newY) == self.list_subsamplelabels[0]) )
            num_00 = np.sum(  (np.array(self.Y) == self.list_subsamplelabels[0]) * (np.array(newY) == self.list_subsamplelabels[0]) )
            num_11 = np.sum(  (np.array(self.Y) == self.list_subsamplelabels[1]) * (np.array(newY) == self.list_subsamplelabels[1]) )
            print("percentage of flips from 0 to 1 = {}".format(num_flips_0to1 / np.sum(np.array(self.Y) == self.list_subsamplelabels[0])))
            print("percentage of flips from 1 to 0 = {}".format(num_flips_1to0 / np.sum(np.array(self.Y) == self.list_subsamplelabels[1])))
            self.original_Y = self.Y + []
            self.Y = newY
            disaggreement_Y_and_origY = np.sum( np.array(self.Y) != np.array(self.original_Y) )
            print("Number of disagreements between current and original Y = {}".format(disaggreement_Y_and_origY))
            if(self.list_correctedlabels_byuser is not None):
                for idx_corrected in self.list_correctedlabels_byuser:
                    idx_in_01 = self.list_subsamplelabels.index(self.Y[idx_corrected])
                    idx_in_01 = 1 - idx_in_01
                    self.Y[idx_corrected] = self.list_subsamplelabels[idx_in_01]
                afterdebug_disaggreement_Y_and_origY = np.sum( np.array(self.Y) != np.array(self.original_Y) )
                print(
                    " >>>> BUT user has corrected {} labels, and new number of disagreement is {}. ".format(
                            len(self.list_correctedlabels_byuser), afterdebug_disaggreement_Y_and_origY
                    )
                )
            
            
        #make internal transformation  ====
        if(self.str_trainoreval == "train"):
            if(flag_enabledataaugmentation == False):
                self.tfms = torchvision.transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),\
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
            elif(flag_enabledataaugmentation == True):
                self.tfms = torchvision.transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        elif(self.str_trainoreval == "eval"):
            self.tfms = torchvision.transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
    
    
    def __len__(self):
        return self.Xraw.shape[0]
    
    def __getitem__(self, n):
        img_toret = np.reshape(self.Xraw[n,:], (3,32,32))
        img_toret = np.transpose(img_toret, [1,2,0])
        img_toret = self.tfms(img_toret)
        label_toret = self.Y[n]
        if(self.list_subsamplelabels is not None):
            orig_label_toret = label_toret + 0.0
            label_toret = self.list_subsamplelabels.index(label_toret)
            #print("     >>>>>>> Label {} was chaanged to {}.".format(orig_label_toret, label_toret))
        return img_toret, label_toret, n
    






class Cifar10DatasetForParamAnal2(Dataset):
    '''
    This class implements Cifar10 dataset to be used in gpex paramanalysis2.
    '''
    def __init__(self, rootdir, fname_batchfile, str_trainoreval, flag_enabledataaugmentation = False, flag_loadalltraining = False,\
                 num_selectedinstances = "all"):
        '''
        dsdfsdf
        '''
        #grab args====
        self.rootdir = rootdir
        self.fname_batchfile = fname_batchfile
        self.str_trainoreval = str_trainoreval
        self.flag_enabledataaugmentation = flag_enabledataaugmentation
        if(self.flag_enabledataaugmentation == True):
            assert(self.str_trainoreval == "train")
        assert(self.str_trainoreval in ["train", "eval"])
        
        #read the batchfile ====
        if(flag_loadalltraining == False):
            self.dict_batch = unpickle(os.path.join(self.rootdir, self.fname_batchfile))
            self.dict_labelnames = unpickle(os.path.join(self.rootdir, "batches.meta"))
            #read X and Y ====
            self.Xraw = self.dict_batch[b'data']
            self.Y = self.dict_batch[b'labels']
            self.label_names = self.dict_labelnames[b'label_names']
        else:
            self.dict_labelnames = unpickle(os.path.join(self.rootdir, "batches.meta"))
            self.Xraw = []
            self.Y = []
            for idx_split in range(5):
                dict_batch = unpickle(os.path.join(self.rootdir, "data_batch_{}".format(idx_split+1)))
                self.Xraw.append(
                    dict_batch[b'data']
                )
                self.Y = self.Y + dict_batch[b'labels']
            self.Xraw = np.concatenate(self.Xraw, 0) #[50000 x D]
            self.label_names = self.dict_labelnames[b'label_names']
            
        #select some samples while preserving class frequencies ====
        self.flag_selectall = isinstance(num_selectedinstances, str) and (num_selectedinstances=="all")
        if(self.flag_selectall == True):
            #do not subsample
            pass
        else:
            #make dict label list_instances ====
            dict_label_to_listinstances = {l:[] for l in range(len(self.label_names))}
            for n in range(self.Xraw.shape[0]):
                dict_label_to_listinstances[self.Y[n]].append(n)
            #make list_idx_selected ====
            self.list_idx_selected = []
            while(len(self.list_idx_selected) != num_selectedinstances):
                for l in range(len(self.label_names)):
                    idx_to_add = dict_label_to_listinstances[l][0]
                    self.list_idx_selected.append(idx_to_add)
                    dict_label_to_listinstances[l] = dict_label_to_listinstances[l][1::]
                    if(len(self.list_idx_selected) == num_selectedinstances):
                        break
            #subsample the dataset ====
            self.Xraw = self.Xraw[self.list_idx_selected, :]
            self.Y = np.array(self.Y)[self.list_idx_selected].flatten().tolist()
            assert(self.Xraw.shape[0] == len(self.Y))
        #print dataset specs======
        print("length of the dataset = {}".format(self.Xraw.shape[0]))
        print("  label frequencies: ")
        for l in range(len(self.label_names)):
            num_in_l = np.sum(np.array(self.Y) == l)
            print("     class {}: {}".format(l, num_in_l))
                
            
        #make internal transformation  ====
        if(self.str_trainoreval == "train"):
            if(flag_enabledataaugmentation == False):
                self.tfms = torchvision.transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),\
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
            elif(flag_enabledataaugmentation == True):
                self.tfms = torchvision.transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        elif(self.str_trainoreval == "eval"):
            self.tfms = torchvision.transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
    
    
    def __len__(self):
        return self.Xraw.shape[0]
    
    def __getitem__(self, n):
        img_toret = np.reshape(self.Xraw[n,:], (3,32,32))
        img_toret = np.transpose(img_toret, [1,2,0])
        img_toret = self.tfms(img_toret)
        label_toret = self.Y[n]
        return img_toret, label_toret, n







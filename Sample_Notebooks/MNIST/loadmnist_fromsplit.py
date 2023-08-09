
import pickle
import random
import os
from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import gzip

class MNISTSplit:
    def __init__(self, rootdir, perctrain):
        '''
        Inputs.
            - perctrain: a number between 0 and 100.
        '''
        #grab args ===
        self.rootdir = rootdir
        self.perctrain = perctrain
        
        #read the whole dataset ===
        self.images, self.labels = MNISTSplit.read_dataset(self.rootdir)
        
        #make internals ===
        self._make_dict_label_to_listindex()
        self.list_idx_train, self.list_idx_test = self._get_listtrain_listtest()
        
        #doublechekcs/prints ==
        self._make_doublechecks()
        self._printstat()
    
    def _make_dict_label_to_listindex(self):
        set_labels = list(set(self.labels))
        self.dict_label_to_listindex = {c:[] for c in set_labels}
        for n in range(self.images.shape[0]):
            self.dict_label_to_listindex[self.labels[n]].append(n)
        
                
    def _get_listtrain_listtest(self):
        list_idx_train = []
        for c in self.dict_label_to_listindex.keys():
            list_idx_train = list_idx_train + random.sample(
                self.dict_label_to_listindex[c],
                k = int((self.perctrain/100.0)*len(self.dict_label_to_listindex[c]))
            )
        list_idx_test = list(
            set(range(self.images.shape[0])) - set(list_idx_train)
        )
        return list_idx_train, list_idx_test
    
    def _make_doublechecks(self):
        assert(
            set(self.list_idx_train).intersection(set(self.list_idx_test)) == set([])
        )
        assert(
            set(self.list_idx_train).union(set(self.list_idx_test)) == set(range(self.images.shape[0]))
        )
        list_temp = []
        for c in self.dict_label_to_listindex.keys():
            list_temp = list_temp + self.dict_label_to_listindex[c]
        assert(
            set(list_temp) == set(range(self.images.shape[0]))
        )
    
    def _printstat(self):
        for c in self.dict_label_to_listindex.keys():
            print("{}:{}".format(c, len(self.dict_label_to_listindex[c])))
    
    @abstractmethod
    def read_dataset(rootdir):
        #read the labels ===
        labels_train = MNISTSplit._read_labels('train-labels-idx1-ubyte.gz', rootdir)
        labels_test  = MNISTSplit._read_labels('t10k-labels-idx1-ubyte.gz', rootdir)
        all_labels = labels_train + labels_test
        
        #read the images ===
        images_train = MNISTSplit._read_images('train-images-idx3-ubyte.gz', rootdir)
        images_test = MNISTSplit._read_images('t10k-images-idx3-ubyte.gz', rootdir)
        images = np.concatenate([images_train, images_test], 0)
        return images, all_labels
    
    @staticmethod
    def _read_labels(fname_gz, rootdir):
        assert(
            fname_gz in ['train-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
        )
        f = gzip.open(os.path.join(rootdir, fname_gz),'r')
        f.read(8)
        #num_images = #60000 if(self.str_trainortest=="train") else 10000
        if(fname_gz == 'train-labels-idx1-ubyte.gz'):
            num_images = 60000
        elif(fname_gz == 't10k-labels-idx1-ubyte.gz'):
            num_images = 10000
        else:
            assert(False)
        labels_toret = []
        for i in range(0, num_images):   
            buf = f.read(1)
            label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            labels_toret.append(int(label[0]))
        return labels_toret
        
    @abstractmethod
    def _read_images(fname_gz, rootdir):
        assert(
            fname_gz in ['train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz']
        )
        f = gzip.open(os.path.join(rootdir, fname_gz),'r')
        
        
        image_size = 28
        if(fname_gz == 'train-images-idx3-ubyte.gz'):
            num_images = 60000
        elif(fname_gz == 't10k-images-idx3-ubyte.gz'):
            num_images = 10000
        else:
            assert(False)
        f.read(16)
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, image_size, image_size, 1)[:,:,:,0]
        images = data.astype(np.uint8)
        return images



class MNISTDatasetFromSplit(Dataset):
    '''
    This class implements the Dataset for loading MNIST dataset.
    '''
    def __init__(self, rootdir, fname_split, str_trainortest, flag_enabledataaugmentation = True, rootpath_splits = "Splits/"):
        '''
        Inputs.
            - fname_split: a string like "split_0.pkl"
        '''
        SIZE_INPUT = 80
        #grab args====
        self.rootdir = rootdir
        self.fname_split = fname_split
        self.str_trainortest = str_trainortest
        self.rootpath_splits = rootpath_splits
        assert(self.str_trainortest in {"train", "test"})
        self.flag_enabledataaugmentation = flag_enabledataaugmentation
        
        #make internal transformation  ====
        if(self.str_trainortest == "train"):
            if(flag_enabledataaugmentation == False):
                self.tfms = torchvision.transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(size = SIZE_INPUT),
                            transforms.ToTensor(),\
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
            elif(flag_enabledataaugmentation == True):
                self.tfms = torchvision.transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(size = SIZE_INPUT),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        elif(self.str_trainortest == "test"):
            self.tfms = torchvision.transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(size = SIZE_INPUT),
                            transforms.ToTensor(),\
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
        else:
            assert False
                    
        #read image names according to partition ====
        self.load_imagesandlabels()
        self.printstats()
        #self.read_images()
        #self.read_labels()
        
    
    def printstats(self):
        set_labels = list(set(self.labels))
        set_labels.sort()
        for c in set_labels:
            print("{}: {}".format(c, np.sum(np.array(self.labels)==c)))
    
    def show_random_instances(self, num_toshow):
        list_idx_toshow = random.sample(range(self.__len__()), num_toshow)
        for idx_toshow in list_idx_toshow:
            plt.figure()
            plt.imshow(self.images[idx_toshow, :, :], cmap="gray")
            plt.title("label = {}".format(self.labels[idx_toshow]))
            plt.show()
        
    def load_imagesandlabels(self):
        with open(os.path.join(self.rootpath_splits, self.fname_split), 'rb') as f:
            content_split = pickle.load(f)
        list_idx_touse = content_split['list_idx_train'] if(self.str_trainortest == "train") else content_split['list_idx_test']
        images_all, all_labels_all = MNISTSplit.read_dataset(self.rootdir)
        self.images = images_all[list_idx_touse, :, :]
        self.labels = np.array(all_labels_all)[list_idx_touse].tolist()
        
    
    def __len__(self):
        toret = self.images.shape[0] #60000 if(self.str_trainortest == "train") else 10000
        return toret
    
        
    def __getitem__(self, n):
        pass
        img_n = self.images[n,:,:] #[28x28]
        img_n = np.stack([img_n, img_n, img_n], axis=-1)
        img_n = img_n.astype(np.uint8)
        x_n = self.tfms(img_n)
        y_n = self.labels[n]
        return x_n, y_n, n
        
        

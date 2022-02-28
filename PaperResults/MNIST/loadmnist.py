
import pickle
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import gzip


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
        


class MNISTDataset(Dataset):
    '''
    This class implements the Dataset for loading Kather dataset.
    '''
    def __init__(self, rootdir, str_trainortest, flag_enabledataaugmentation = True):
        '''
        dsdfsdf
        '''
        SIZE_INPUT = 80
        #grab args====
        self.rootdir = rootdir
        self.str_trainortest = str_trainortest
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
        self.read_images()
        self.read_labels()
        
    
    def show_random_instances(self, num_toshow):
        list_idx_toshow = random.sample(range(self.__len__()), num_toshow)
        for idx_toshow in list_idx_toshow:
            plt.figure()
            plt.imshow(self.images[idx_toshow, :, :], cmap="gray")
            plt.title("label = {}".format(self.labels[idx_toshow]))
            plt.show()
        
    
    def read_labels(self):
        #decide on fname.gz
        if(self.str_trainortest == "train"):
            f = gzip.open(os.path.join(self.rootdir, 'train-labels-idx1-ubyte.gz'),'r')
        elif(self.str_trainortest == "test"):
            f = gzip.open(os.path.join(self.rootdir, 't10k-labels-idx1-ubyte.gz'),'r')
        else:
            assert False
        f.read(8)
        num_images = 60000 if(self.str_trainortest=="train") else 10000
        self.labels = []
        for i in range(0, num_images):   
            buf = f.read(1)
            label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            self.labels.append(int(label[0]))
        
    
    def read_images(self):
        #decide on fname.gz
        if(self.str_trainortest == "train"):
            f = gzip.open(os.path.join(self.rootdir, 'train-images-idx3-ubyte.gz'),'r')
        elif(self.str_trainortest == "test"):
            f = gzip.open(os.path.join(self.rootdir, 't10k-images-idx3-ubyte.gz'),'r')
        else:
            assert False
        
        image_size = 28
        num_images = 60000 if(self.str_trainortest=="train") else 10000
        f.read(16)
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, image_size, image_size, 1)[:,:,:,0]
        self.images = data.astype(np.uint8)
        
             
    def __len__(self):
        toret = 60000 if(self.str_trainortest == "train") else 10000
        return toret
    
        
    def __getitem__(self, n):
        pass
        img_n = self.images[n,:,:] #[28x28]
        img_n = np.stack([img_n, img_n, img_n], axis=-1)
        img_n = img_n.astype(np.uint8)
        x_n = self.tfms(img_n)
        y_n = self.labels[n]
        return x_n, y_n, n
        
        

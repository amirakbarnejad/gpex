
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnetforcifar
import gpex
from gpex.kernelmappings.image import Resnet50BackboneKernelDivideAvgPool


class ClampAndTanh(torch.nn.Module):
    def __init_(self, minval=-1.0, maxval=1.0):
        self.minval, self.maxval = minval, maxval
        super(ClampAndTanh, self).__init__()
        
        
    def forward(self, x):
        output = torch.nn.functional.tanh(
                        torch.clamp(
                            x, -1.0, 1.0
                        )
                    )
        return output
    
def initweights_to_zero(m):
    if type(m) in {nn.Linear, nn.Conv2d, nn.Linear}:
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(np.random.randn()*0.1) #TODO:check

class ModuleF1(torch.nn.Module):
    def __init__(self, module_caller):
        super(ModuleF1, self).__init__()
        #make internals ===
        self.module = Resnet50BackboneKernelDivideAvgPool(
            num_classes = 10,
            du_per_class = 20
        ) 
        print("<><><><><><><><><> finisehd creating module_tail <><><><><><><><>.")
    
    def set_rng_outputheads(self, rng_outputhead):
        self.module.set_rng_outputheads(rng_outputhead)
    
    
    def forward(self, x):
        toret = self.module(x)
        return toret
            
class MainModule(nn.Module):
    def __init__(self, num_classes, device, ds_inducing, dl_recurring, dl_nonrecurring, dl_test, batchsize, dim_wideoutput):
        '''
        Inputs:
            - size_input: size of the input, e.g., [32 x 2000 x 7 x 7].
            - device: the device on which the GP fields are going to be created.
            - num_outputheads: an integer, number of output heads.
        '''
        super(MainModule, self).__init__()
        #grab args ===
        self.num_classes = num_classes
        self.device = device
        self.ds_recurring = ds_inducing
        self.dl_recurring = dl_recurring
        self.dl_nonrecurring = dl_nonrecurring
        self.dl_test = dl_test
        self.batchsize = batchsize
        #self.iter_dl_recurring = iter(self.dl_recurring)
        #make internal module_tobecomeGP ===
        self.module_tobecomeGP = resnetforcifar.ResNet18(
            num_classes=num_classes,
            dim_wideoutput=dim_wideoutput
        )
        #make internals ===
        self.dic_dlname_to_iter = {
            "dl_recurring":iter(self.dl_recurring),
            "dl_nonrecurring":iter(self.dl_nonrecurring),
            "dl_test":iter(self.dl_test)
        }
        #make module f1 ===
        self.module_f1 = ModuleF1(self)
        self._lastidx_recurring = []
        #internal field to subsample when feeding minbatch ====
        self.n_subsampleminibatch = None
            
    def forward(self, x, y, n):
        return self.module_tobecomeGP(x), y, n
        
    
    def func_get_modulef1(self):
        return self.module_f1
    
    def func_mainmodule_to_moduletobecomeGP(self, module_input):
        return module_input.module_tobecomeGP
    
    
    def _func_feed_minibatch(self, dl_input, str_dlname, flag_addnoisetoX = False):
        #print("reached here 1")
        if(False):#iter_dl is None):
            pass #x, y, n = next(iter(dl_input))
        else:
            try:
                x, y, n = next(self.dic_dlname_to_iter[str_dlname])
            except (StopIteration):
                self.dic_dlname_to_iter[str_dlname] = iter(dl_input)
                x, y, n = next(self.dic_dlname_to_iter[str_dlname])
                
        
        if(flag_addnoisetoX == True):
            idx_permutex = np.random.permutation([u for u in range(list(x.size())[0])]).tolist()
            idx_permutex = torch.LongTensor(idx_permutex)
            x_perumted = x[idx_permutex, :, :, :]
            
            rand_w = torch.rand((list(x.size())[0])).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) #[Nx1x1x1]
            rand_w = rand_w*(rng_randw[1]-rng_randw[0]) + rng_randw[0] #in [rng[0] , rng[1]]
            rand_w = rand_w.detach()
            #print("rand_w.shape = {}".format(rand_w.shape))
            x = rand_w*x + (1.0-rand_w)*x_perumted  #[N x 3 x 224 x 224]
            x = x + 0.1*torch.randn_like(x).float()
        
        #print("reached here 2")
#         if(dl_input.dataset == ds_recurring):
#             self._lastidx_recurring = n #TODO:move this operation inside the dl.
        #print("reached here 3")
        #print("x.shape = {}".format(x.shape))
        #print("y.shape = {}".format(y.shape))
        
        if(self.n_subsampleminibatch is None):
            pass
        else:
            x = x[0:self.n_subsampleminibatch, :,:,:]
            y = y[0:self.n_subsampleminibatch]
            n = n[0:self.n_subsampleminibatch]
        
        if(dl_input.dataset == self.ds_recurring):
            self._lastidx_recurring = n
            if(len(n) != list(x.size())[0]):
                assert False
        
        output, _, _ = self.forward(x.to(self.device), y, n)
        
        #print("reached here 4")
        return output, y, n
    
    def func_feed_inducing_minibatch(self):
        output, y, n = self._func_feed_minibatch(
                            self.dl_recurring,
                            str_dlname = "dl_recurring"
                         )
        return output, y, n
    
    def func_feed_noise_minibatch(self):
        output, y, n = self._func_feed_minibatch(
                        self.dl_nonrecurring,
                        str_dlname = "dl_nonrecurring",
                        flag_addnoisetoX=True
                      )
        return output, y, n
    
    def func_feed_nonrecurring_minibatch(self):
        output, y, n = self._func_feed_minibatch(
                                self.dl_nonrecurring,
                                str_dlname = "dl_nonrecurring",
                                flag_addnoisetoX=False
                              )
        return output, y, n
    
    def func_feed_test_minibatch(self):
        output, y, n = self._func_feed_minibatch(
                                    self.dl_test,
                                    str_dlname = "dl_test"
                                )
        return output, y, n
    
    def func_get_indices_lastrecurringinstances(self):
        return self._lastidx_recurring.cpu().numpy().tolist()


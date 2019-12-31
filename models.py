
import torch
import torch.nn as N
import numpy as np

class VAE(torch.nn.Module):
    '''
    Variational Auto-Encoder: https://arxiv.org/pdf/1312.6114.pdf
    
    Architecture is hard-coded for now, only number of hidden units can be controlled through parameters.
    #TODO: Accept layers, activations functions as list.
    
    Note that this is the architecture proposed in RaTC paper.: https://arxiv.org/pdf/1906.04281.pdf
    
    '''
    
    def __init__(self,M,eh1=600,eh2=200,dh2=600):
        super(VAE,self).__init__()
        
        # Encoder
        self.ehl1 = torch.nn.Linear(M,eh1)
        self.afe1  = torch.nn.Tanh()
        self.ehl2mean = torch.nn.Linear(eh1,eh2)
        self.ehl2var  = torch.nn.Linear(eh1,eh2)
        
        # Decoder
        self.dhl2     = torch.nn.Linear(eh2,dh2)
        self.afd1  = torch.nn.Tanh()
        self.dhl1 = torch.nn.Linear(dh2,M)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1) ## This is more efficient&stable then Log(Softmax)
        
    def encode(self,X):
        '''
        Encoder part of VAE. Outputs mean and variance of a Multivariate Gaussian. Note that var(z) is diagonal.
        '''
        e1 = self.ehl1(X)
        e2 = self.afe1(e1)
        mean = self.ehl2mean(e2)
        varlog  = self.ehl2var(e2) # log(variance)
        std = torch.exp(0.5*varlog) # std = sqrt(variance)

        KL = 0.5*(torch.sum(mean*mean+std*std-varlog,dim=1)-mean.shape[1]) # KL divergence for VAE. 
        '''
        See https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes/370048#370048
        for full derivation.
        '''
        return mean,std,KL
    
    def reparameterize(self,mean,std):
        '''
        Reparametrization trick. Sample z using an auxiliary random variable, epsilon. 
        '''
        epsilon = torch.randn_like(mean)
        z = mean + std*epsilon
        return z
    
    def decode(self,z): # Decoder part of VAE.
        d2 = self.dhl2(z)
        d1 = self.afd1(d2)
        d0 = self.dhl1(d1)
        xlog = self.logsoftmax(d0)
        return xlog
    
    def forward(self,X): # Full forward propagation with composition of three functions above.
        mean,std,KL = self.encode(X)
        z = self.reparameterize(mean,std)
        xlog = self.decode(z)
        return xlog,KL
            
class Critic(torch.nn.Module):
    '''
    TODO: Accept the architecture as parameters.
    Note that this is the architecture proposed in RaTC paper.: https://arxiv.org/pdf/1906.04281.pdf
    '''
    def __init__(self,ci):
        super(Critic,self).__init__()
        self.bn  = torch.nn.BatchNorm1d(ci)
        self.hl1 = torch.nn.Linear(ci,100)
        self.ha1 = torch.nn.ReLU()
        self.hl2 = torch.nn.Linear(100,100)
        self.ha2 = torch.nn.ReLU()
        self.hl3 = torch.nn.Linear(100,10)
        self.ha3 = torch.nn.ReLU()
        self.ol = torch.nn.Linear(10,1)
        self.oa = torch.nn.Sigmoid()
    
    def feature_vector(self,x,LE,mask):
        '''
        A heuristic feature vector proposed in RaTC paper. Note that this function can be modified to 
        include different features. 
        LE : Negative Log Likelihood
        H0 : Number of unobserved items that a user will interact.
        H1 : Number of observed   items that a user has interacted.
        
        mask : A binary matrix of shape like x. 1 if an interaction observed, 0 otherwise.
        '''
        H0 = torch.sum(x*(1-mask),dim=1)
        H1 = torch.sum(x*mask,dim=1)
        return torch.stack([LE,H0,H1],dim=1)

        
    def forward(self,x, LE, mask):
        h = self.feature_vector(x,LE,mask)
        xn  = self.bn(h)
        z1_ = self.hl1(xn)
        z1  = self.ha1(z1_)  
        z2_ = self.hl2(z1)
        z2  = self.ha2(z2_)
        z3_ = self.hl3(z2)
        z3  = self.ha3(z3_)
        y_   = self.ol(z3)
        y  = self.oa(y_)
        return y
    

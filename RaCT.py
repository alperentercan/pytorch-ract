
import torch
import torch.nn as N
import numpy as np
from models import VAE, Critic
from evaluation_metrics import NDCG_binary_at_k_batch

class RaCT():
    def __init__(self,M,eh1,eh2,dh2,ci,lr_ac=0.001,lr_cr = 0.001):
        ## Network initializations
        # Actor
        self.actor = VAE(M,eh1,eh2,dh2) # Number of inputs, units in encoder_hidden_layer1, encoder_hidden_layer2,
                                                #decoder_hidden_layer1
        # Critic
        self.critic = Critic(ci) # Length of feature vector
        # Optimizers
        self.optim_actor = torch.optim.Adam(self.actor.parameters(),lr=lr_ac)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(),lr=lr_cr)
        self.mse = torch.nn.MSELoss()

 
    def pretrain_actor(self,X,batch_size,beta_max,epochs,epochs_annealing,val_set,masked=True):
        '''
        Pretraining of actor using MLE cost = NLL + Beta*KL
        
        Minimize NLL: Maximize the probability of interactions in the reconstruction which are 1 in the input.
        KL: Regulatory, makes sure the distribution of z is not very different than prior.
        
        X: Interaction Matrix, training dataset
        beta_max : Max Beta that will be reached after annealing
        epochs : Total number of epochs
        epochs_annealing: Number of epochs for annealing. Since beta_max is set, controls how quick beta grows.
        val_set : Validation set for validation
        
        masked: Controls the training task. If True, only a partial history is given to actor and only unobserved 
        interactions are considered in NLL. Proposed method is not clear in the paper.
        
        '''
        beta = 0
        beta_increase = beta_max/epochs_annealing # Linear Growth of Beta
        for epoch in range(epochs):
            self.optim_actor.zero_grad()
            ## Sample a batch
            batch_ind = np.random.choice(X.shape[0],batch_size)
            xbatch = X[batch_ind,:]
            xbatch = torch.tensor(xbatch.toarray(),dtype=torch.float32) # Scipy Sparse to Tensor Dense


            ## UNMASKED
            if not masked:
                xlog,KL = self.actor.forward(xbatch)
                nll = -torch.mean(xlog*xbatch,dim=1)
                elbo_beta = torch.mean(nll + beta*KL)
            
            ## MASKED        
            else:
                # Sample masks
                mask,xbatch_masked = self.mask(xbatch)
                xbatch_reverse_masked = xbatch*(1-mask)
                xlog,KL = self.actor.forward(xbatch_masked)
                nll = -torch.mean(xlog*xbatch_reverse_masked,dim=1)
                elbo_beta = torch.mean(nll + beta*KL)            
            
            print('NLLL : ',torch.mean(nll.detach()))
            print('Elbo : ',elbo_beta.detach())
            elbo_beta.backward()
            self.optim_actor.step()  # Update the actor
            if epoch < epochs_annealing:
                beta = beta + beta_increase
            if epoch%20 == 0:
                self.evaluate(val_set)
                
    def pretrain_critic(self,X,batch_size,epochs):
        '''
        Pretraining of critic using MSE between score predictions of Critic network and NDCG@100.
        Critic tries to learn giving similar results with NDCG.
        No unmasked option here, since NDCG only accounts for unobserver interactions.
        '''
        for epoch in range(epochs):
            self.optim_actor.zero_grad()
            self.optim_critic.zero_grad()
            # Sample a batch
            batch_ind = np.random.choice(X.shape[0],batch_size)
            xbatch_spr = X[batch_ind,:]
            xbatch = torch.tensor(xbatch_spr.toarray(),dtype=torch.float32)
            # Prepare masks
            mask, xbatch_masked = self.mask(xbatch)
            xbatch_reverse_masked = xbatch*(1-mask)
            # Find score prediction of critic given masked input
            xlog,KL = self.actor.forward(xbatch_masked)
            nll = -torch.mean(xlog*xbatch_reverse_masked,dim=1)
            score_pred = self.critic.forward(xbatch, nll, mask)
            
            ## I will try the one from implementation. 1st-arg=prediction, 2nd-arg = reverse-masked-input
            # 4th-arg = masked_input
            ndcg = NDCG_binary_at_k_batch(xlog.detach().numpy(),xbatch_reverse_masked,100,xbatch_masked)
            ndcg = torch.tensor(ndcg.reshape(-1,1),dtype=torch.float32)
            print('NDCG mean :',torch.mean(ndcg))
            mse_loss = self.mse(score_pred,ndcg) ## Minimize the difference between Critic and NDCG
            print('MSE : ',mse_loss.detach())
            mse_loss.backward()
            self.optim_critic.step()
          

            
    def alternative_training(self,X,batch_size,beta,epochs,recalculate_actor=False):
        '''
        Train both of them together. Do the following epochs times.
        
        1. Train Actor to maximize the score of predictions.Use Critic as a both
        differentiable and accurate metric.(At least this is what we hope to get.)
        2. Train Critic using MSE cost with NDCG. We need this to make sure that we can predict the score of 
        distributions produced by the new Actor.
        
        Note that in the tests, this stage is observed to be too unstable. Unlucky seeds can cause collapse of 
        the whole training.
        
        TODO: Work on the unstability.
        
        recalculate_actor : Experimental parameter for Critic Phase. If True, reconstruct the graph of actor network for 
        the training of Critic. If false, use the results from Actor phase as constants.
        
        '''
        for epoch in range(epochs):
            # Sample a batch. Will use the same batch for both phases.
            batch_ind = np.random.choice(X.shape[0],batch_size)
            xbatch_spr = X[batch_ind,:]
            xbatch = torch.tensor(xbatch_spr.toarray(),dtype=torch.float32)
            # Mask it
            mask, xbatch_masked = self.mask(xbatch)
            xbatch_reverse_masked = xbatch*(1-mask)
            ### Actor Phase            
            self.optim_actor.zero_grad()
            self.optim_critic.zero_grad()
            xlog,KL = self.actor.forward(xbatch_masked)
            nll = -torch.mean(xlog*xbatch_reverse_masked,dim=1)
            actor_loss = -self.critic.forward(xbatch, nll, mask).mean()# Use -critic_score as the loss. 
                                                                            #So maximize the critic score
            actor_loss.backward()
            self.optim_actor.step()
            print('Critic ',epoch,' , ',actor_loss.detach())
            print('NLLL : ',torch.mean(nll.detach()))

            
            ### Critic Phase
            self.optim_actor.zero_grad();self.optim_critic.zero_grad();
 
            if recalculate_actor:
                xlog,KL = self.actor.forward(xbatch)
                nll = -torch.mean(xlog*xbatch,dim=1)
            else:
                nll.detach_()
            score_pred = self.critic.forward(xbatch, nll, mask)
            ndcg = NDCG_binary_at_k_batch(xlog.detach().numpy(),xbatch_reverse_masked,100,xbatch_masked)
            ndcg = torch.tensor(ndcg.reshape(-1,1),dtype=torch.float32)
            print('NDCG mean :',torch.mean(ndcg))
            mse_loss = self.mse(score_pred,ndcg)
            mse_loss.backward()
#             print('MSE Loss : ',mse_loss.detach())
            self.optim_critic.step()            
        
    def mask(self,X,p=0.5):
        '''
        Generates a random(Bernoulli) matrix(mask) of same shape with X. p is the probability of each element being 1. 
        Note that elements in the matrix sampled from independent distributions.  
        '''
        mask = torch.distributions.bernoulli.Bernoulli(p).sample(sample_shape=X.shape)
        X_masked = X*mask
        return mask,X_masked
    
    def evaluate(self,val_set):
        with torch.no_grad():
            ## Convert from Scipy sparse to Torch Tensor.
            xbatch = torch.tensor(val_set.toarray(),dtype=torch.float32)
            mask, xbatch_masked = self.mask(xbatch)
            xbatch_reverse_masked = xbatch*(1-mask) # Reverse Mask, 1 if not observed           
            xlog,KL = self.actor.forward(xbatch_masked) # Accepts a 'partial'(masked) interaction history. 
            nll = -torch.mean(xlog*xbatch_reverse_masked,dim=1) # We only care about guessing unobserved interactionsl
            score_pred = self.critic.forward(xbatch, nll, mask) # Note that first argument 
                                                                    # should be the original(unmasked) matrix.
            # Calculate NDCG@100.
            ndcg = NDCG_binary_at_k_batch(xlog.detach().numpy(),xbatch_reverse_masked,100,xbatch_masked)
            ndcg = torch.tensor(ndcg.reshape(-1,1),dtype=torch.float32)
            print('NDCG mean :',torch.mean(ndcg))
            

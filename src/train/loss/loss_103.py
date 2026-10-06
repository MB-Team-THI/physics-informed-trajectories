import torch
import torch.nn as nn
from src.train.loss.loss import loss
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist


def masked_argmax(mask,values):
    idx = torch.argmax(values[mask])
    return 

def similarity_func(a,b,sim_type='eucl',max=1000.0):
    if sim_type == 'eucl':
        d = ((a - b) ** 2).sum(dim=1)
        s = 1.0/(1.0+d)
    elif sim_type == 'eucl_max':
        d = ((a - b) ** 2).sum(dim=1)
        d = torch.clamp(d,max=max)
        s = 1.0-d/max
    elif sim_type == 'cos':
        s = torch.sum(a*b,dim=1)/(torch.norm(a)*torch.norm(b))
    elif sim_type == 'dot':
        s = torch.squeeze(torch.bmm(a[:,None,:],b[:,:,None]))
    else:
        d = ((a - b) ** 2).sum(dim=1)
        s = 1/(1+d)

    return s

def similarity_func2(a,b,sim_type='eucl',max=1000.0):
    if sim_type == 'eucl':
        d = ((a[:,None,:] - b[None,:,:]) ** 2).sum(dim=2)
        s = 1.0/(1.0+d)
    elif sim_type == 'eucl_max':
        d = ((a[:,None,:] - b[None,:,:]) ** 2).sum(dim=2)
        d = torch.clamp(d,max=max)
        s = 1.0-d/max
    elif sim_type == 'cos':
        s = torch.sum(a[:,None,:]*b[None,:,:],dim=2)/(torch.norm(a)*torch.norm(b))
    elif sim_type == 'dot':
        s = torch.mm(a,b.T)
    else:
        d = ((a[:,None,:] - b[None,:,:]) ** 2).sum(dim=2)
        s = 1/(1+d)

    return s

def dissimilarity_func(a,b,sim_type='eucl'):
    if sim_type == 'eucl':
        d = ((a - b) ** 2).sum(dim=1)
    elif sim_type == 'cos':
        s = torch.sum(a*b,dim=1)/(torch.norm(a)*torch.norm(b))
        d = s/2+0.5
    else:
        d = ((a - b) ** 2).sum(dim=1)

    return d

def dissimilarity_func2(a,b,sim_type='eucl'):
    if sim_type == 'eucl':
        d = ((a[:,None,:] - b.unsqueeze(0).expand(a.shape[0],-1,-1)) ** 2).sum(dim=2)
    elif sim_type == 'cos':
        s = torch.sum(a[:,None,:]*b[None,:,:],dim=2)/(torch.norm(a)*torch.norm(b))
        d = s/2+0.5
    else:
        d = ((a[:,None,:] - b[None,:,:]) ** 2).sum(dim=2)

    return d



class loss_103(loss):
    def __init__(self,
                idx = 103,
                name = 'Double SWAV + Weighted Recon Loss',
                description = 'Double SWAV given: anchor, pp, pn +  Weighted Recon Loss',
                input_ = '',
                output = '1D',
                ) -> None:
        super().__init__(idx,name,description,input_,output)
        self.epsilon = 0.05
        self.sinkhorn_iterations = 3
        self.world_size = -1
    
    def hard_metric_loss(self,
                    z_a=None,
                    z_pp=None, 
                    z_pn=None, 
                    z_nn=None, 
                    l2=0.1, 
                    beta_pp=1.0, 
                    beta_pn=1.0, 
                    beta_nn=1.0,
                    negatives_a_pp_pn=None,
                    negatives_nn=None,
                    pn_equal_pp = False,
                    hardest_sampling=True,
                    tau=0.1,
                    sim_type='eucl',
                    margin_nn=1.0,
                    margin_pn=1.0):

        d_pn = dissimilarity_func(z_a,z_pn,sim_type=sim_type)
        d_pp = dissimilarity_func(z_a,z_pp,sim_type=sim_type)

        z_all = torch.cat((z_a,z_pp,z_pn,z_nn))
        negatives_all = torch.cat((negatives_a_pp_pn,negatives_a_pp_pn,negatives_a_pp_pn,negatives_nn),dim=1)
        d_all = dissimilarity_func2(z_a,z_all,sim_type=sim_type)

        if hardest_sampling:
            d_all[torch.le(d_all, d_pn[:,None]) | torch.bitwise_not(negatives_all)] = float('inf')
            idx_max = torch.argmin(d_all,dim=1)[:,None]
        else:
            mask = (torch.gt(d_all, d_pn[:,None]) & negatives_all & torch.lt(d_all, d_pn[:,None]+margin_nn))
            idx_vect = torch.arange(0,d_all.shape[1],dtype=torch.int64,device="cuda")
            idx_max = torch.zeros(z_a.shape[0],dtype=torch.int64,device="cuda")
            for i in range(z_a.shape[0]):
                idx_temp = idx_vect[mask[i,:]]
                if len(idx_temp) ==0:
                    idx_max[i] = z_a.shape[0]+z_pp.shape[0]+z_pn.shape[0]+i
                else:
                    idx_max[i] = idx_temp[torch.randint(len(idx_temp),(1,))]
            idx_max = idx_max[:,None]
        
        d_nn = d_all.gather(1,idx_max)[:,0]
        z_nn_inside = z_all[idx_max[:,0],:]


        if pn_equal_pp:
            l_nn            = F.relu(d_pn-d_nn+margin_pn)
            l_pn            = F.relu(d_pp-d_nn+margin_pn)
        else:
            l_nn            = F.relu(d_pn-d_nn+margin_nn)
            l_pn            = F.relu(d_pp-d_pn+margin_pn)

        loss = beta_nn*l_nn + beta_pn*l_pn
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.mean(torch.norm(z_a,dim=1)) + torch.mean(torch.norm(z_nn_inside,dim=1)) + torch.mean(torch.norm(z_pn,dim=1)) + torch.mean(torch.norm(z_pp,dim=1)))/4
        return loss

    def hard_proto_metric_loss(self,
                    z_a=None,
                    z_pp=None, 
                    z_pn=None, 
                    z_nn=None,
                    prototypes=None, 
                    l2=0.1, 
                    beta_pp=1.0, 
                    beta_pn=1.0, 
                    beta_nn=1.0,
                    negatives_a_pp_pn=None,
                    negatives_nn=None,
                    pn_equal_pp = False,
                    hardest_sampling=True,
                    tau=0.1,
                    sim_type='eucl',
                    margin_nn=1.0,
                    margin_pn=1.0):

        d_pn = dissimilarity_func(z_a,z_pn,sim_type=sim_type)
        d_pp = dissimilarity_func(z_a,z_pp,sim_type=sim_type)


        # Find closest prototypes to anchor, pp, pn
        d_a_proto = dissimilarity_func2(z_a,prototypes,sim_type=sim_type)
        d_pp_proto = dissimilarity_func2(z_pp,prototypes,sim_type=sim_type)
        d_pn_proto = dissimilarity_func2(z_pn,prototypes,sim_type=sim_type)
        max_idx_a_proto = torch.argmin(d_a_proto,dim=1)
        max_idx_pp_proto = torch.argmin(d_pp_proto,dim=1)
        max_idx_pn_proto = torch.argmin(d_pn_proto,dim=1)

        negative_prototypes = torch.ones(d_a_proto.shape[0], d_a_proto.shape[1],dtype=torch.bool, device="cuda")
        negative_prototypes.scatter_(1, max_idx_a_proto.unsqueeze(1), 0)
        negative_prototypes.scatter_(1, max_idx_pp_proto.unsqueeze(1), 0)
        negative_prototypes.scatter_(1, max_idx_pn_proto.unsqueeze(1), 0)

        # Gather all instances and prototypes 
        z_all = torch.cat((z_a,z_pp,z_pn,z_nn,prototypes))
        negatives_all = torch.cat((negatives_a_pp_pn,negatives_a_pp_pn,negatives_a_pp_pn,negatives_nn,negative_prototypes),dim=1)
        d_all = dissimilarity_func2(z_a,z_all,sim_type=sim_type)

        if hardest_sampling:
            d_all[torch.le(d_all, d_pn[:,None]) | torch.bitwise_not(negatives_all)] = float('inf')
            idx_max = torch.argmin(d_all,dim=1)[:,None]
        else:
            mask = (torch.gt(d_all, d_pn[:,None]) & negatives_all & torch.lt(d_all, d_pn[:,None]+margin_nn))
            idx_vect = torch.arange(0,d_all.shape[1],dtype=torch.int64,device="cuda")
            idx_max = torch.zeros(z_a.shape[0],dtype=torch.int64,device="cuda")
            for i in range(z_a.shape[0]):
                idx_temp = idx_vect[mask[i,:]]
                if len(idx_temp) ==0:
                    idx_max[i] = z_a.shape[0]+z_pp.shape[0]+z_pn.shape[0]+i
                else:
                    idx_max[i] = idx_temp[torch.randint(len(idx_temp),(1,))]
            idx_max = idx_max[:,None]
        
        d_nn = d_all.gather(1,idx_max)[:,0]
        z_nn_inside = z_all[idx_max[:,0],:]

        if pn_equal_pp:
            l_nn            = F.relu(d_pn-d_nn+margin_pn)
            l_pn            = F.relu(d_pp-d_nn+margin_pn)
        else:
            l_nn            = F.relu(d_pn-d_nn+margin_nn)
            l_pn            = F.relu(d_pp-d_pn+margin_pn)

        loss = beta_nn*l_nn + beta_pn*l_pn
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.mean(torch.norm(z_a,dim=1)) + torch.mean(torch.norm(z_nn_inside,dim=1)) + torch.mean(torch.norm(z_pn,dim=1)) + torch.mean(torch.norm(z_pp,dim=1)))/4
        return loss
    
    def ptototype_loss(self,
        z,
        prototypes,
        tau=0.1,
        sim_type='eucl',
        learning_style='vq'):
        

        # Softmax apporach: 
        if learning_style=='soft_log':
            s = similarity_func2(z,prototypes,sim_type=sim_type)
            sft = torch.softmax(dim=1)
            max_idx_vect = sft(s)
            s_positive = torch.exp(s/tau*max_idx_vect.T)
            s_all = torch.sum(torch.exp(s/tau),dim=1)
            loss = -torch.log(s_positive/(s_all))
        elif learning_style=='hard_log':
            # Loss for encoder
            s = similarity_func2(z,prototypes,sim_type=sim_type)
            max_idx = torch.argmax(s,dim=1)
            s_positive = torch.exp(s[torch.arange(s.shape[0]),max_idx]/tau)
            s_all = torch.sum(torch.exp(s/tau),dim=1)
            loss = -torch.log(s_positive/(s_all))
            encodings = torch.zeros(s.shape[0], s.shape[1], device="cuda")
            encodings.scatter_(1, max_idx.unsqueeze(1), 1)
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        elif learning_style=='vq':
            d = dissimilarity_func2(z,prototypes,sim_type=sim_type)
            max_idx = torch.argmin(d,dim=1)
            loss = d[torch.arange(d.shape[0]),max_idx]
            encodings = torch.zeros(d.shape[0], d.shape[1], device="cuda")
            encodings.scatter_(1, max_idx.unsqueeze(1), 1)
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        elif learning_style=='soft_vq':
            s = similarity_func2(z,prototypes,sim_type=sim_type)
            sft = torch.softmax(dim=1)
            max_idx_vect = sft(s)
            max_idx = torch.argmax(max_idx_vect,dim=1)
            loss = d[torch.arange(d.shape[0]),max_idx]
            loss_diff = torch.mean(1-max_idx_vect[torch.arange(d.shape[0]),max_idx])
            encodings = max_idx_vect
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return torch.mean(loss),perplexity


    def reconstruction_loss_weighted(self,
                                    x,x_pred,
                                    beta_traj=1.0,
                                    beta_infra=1.0,
                                    beta_back_infra=1.0,
                                    beta_back_traj=1.0):
        traj_base = torch.clone(x[:,0,:,:])
        mask_traj = traj_base>0.0
        traj_pred = torch.clone(x_pred[:,0,:,:])
        traj_pred = torch.mul(traj_pred,mask_traj)

        mask_back_traj = traj_base==0.0
        back_traj = torch.clone(x_pred[:,0,:,:])
        back_traj = torch.mul(back_traj,mask_back_traj)
        
        infra_base = torch.clone(x[:,1,:,:])
        mask_infra = infra_base>0.0
        infra_pred = torch.clone(x_pred[:,1,:,:])
        infra_pred = torch.mul(infra_pred,mask_infra)

        mask_back_infra = infra_base==0.0
        back_infra = torch.clone(x_pred[:,1,:,:])
        back_infra = torch.mul(back_infra,mask_back_infra)

        l_traj = ((traj_base - traj_pred) ** 2).sum()/mask_traj.sum()
        l_infra = ((infra_base - infra_pred) ** 2).sum()/mask_infra.sum()
        l_back_traj = (back_traj ** 2).sum()/mask_back_traj.sum()
        l_back_infra = (back_infra **2).sum()/mask_back_infra.sum()
        l = beta_traj * l_traj + beta_infra * l_infra + beta_back_traj * l_back_traj + beta_back_infra * l_back_infra
        return l
    
    def forward(self,
                data,
                batch_idx,
                proto_freeze_batch_idx, 
                alpha_recon=0.0, 
                alpha_INFRAmetric=0.0,
                alpha_MERGEmetric=1.0,
                alpha_INFRAproto=0.0,
                alpha_MERGEproto=10.0,
                alpha_INFRAcommit=0.0,
                alpha_MERGEcommit=0.0,
                alpha_MERGE_proto_metric = 1.0,
                alpha_INFRA_proto_metric =1.0,
                beta_traj=5.0,
                beta_infra=5.0,
                beta_back_infra=10.0,
                beta_back_traj=20.0,
                sim_type='eucl'):
        """
        Computes loss for each batch.
        """
        if batch_idx <= proto_freeze_batch_idx:
            #alpha_MERGEproto = 0.0
            #alpha_MERGEcommit = 0.0
            #alpha_INFRAproto = 0.0
            #alpha_INFRAcommit = 0.0
            loss_infra_proto = torch.Tensor([0.0]).cuda()
            loss_merge_proto = torch.Tensor([0.0]).cuda()
            loss_infra_commit = torch.Tensor([0.0]).cuda()
            loss_merge_commit = torch.Tensor([0.0]).cuda()
            perplexity = torch.Tensor([0.0]).cuda()
            if alpha_INFRAmetric == 0.0:
                loss_infra_metric = torch.Tensor([0.0]).cuda()
            else:
                loss_infra_metric = self.hard_metric_loss(z_a=data['z_infra_a'],z_pp=data['z_infra_pp'],z_pn=data['z_infra_pn'],z_nn=data['z_infra_nn'],negatives_a_pp_pn=data['negatives_a_pp_pn'],negatives_nn=data['negatives_nn'],pn_equal_pp=True)
            if alpha_MERGEmetric == 0.0:
                loss_merge_metric = torch.Tensor([0.0]).cuda()
            else:
                loss_merge_metric = self.hard_metric_loss(z_a=data['z_merge_a'],z_pp=data['z_merge_pp'],z_pn=data['z_merge_pn'],z_nn=data['z_merge_nn'],negatives_a_pp_pn=data['negatives_a_pp_pn'],negatives_nn=data['negatives_nn'],hardest_sampling=False)
        else:
            loss_infra_proto = torch.Tensor([0.0]).cuda()
            loss_merge_proto = torch.Tensor([0.0]).cuda()
            loss_infra_commit = torch.Tensor([0.0]).cuda()
            loss_merge_commit = torch.Tensor([0.0]).cuda()
            perplexity = torch.Tensor([0.0]).cuda()
            if alpha_INFRAmetric == 0.0:
                loss_infra_metric = torch.Tensor([0.0]).cuda()
            else:
                loss_infra_metric = self.hard_proto_metric_loss(z_a=data['z_infra_a'],z_pp=data['z_infra_pp'],z_pn=data['z_infra_pn'],z_nn=data['z_infra_nn'],negatives_a_pp_pn=data['negatives_a_pp_pn'],negatives_nn=data['negatives_nn'],prototypes=data['prototypes_infra'].detach(),pn_equal_pp=True)
            if alpha_MERGEmetric == 0.0:
                loss_merge_metric = torch.Tensor([0.0]).cuda()
            else:
                loss_merge_metric = self.hard_proto_metric_loss(z_a=data['z_merge_a'],z_pp=data['z_merge_pp'],z_pn=data['z_merge_pn'],z_nn=data['z_merge_nn'],negatives_a_pp_pn=data['negatives_a_pp_pn'],negatives_nn=data['negatives_nn'],prototypes=data['prototypes_merge'].detach())
            if alpha_INFRAproto == 0.0:
                loss_infra_proto = torch.Tensor([0.0]).cuda()
            else:
                loss_infra_proto = self.ptototype_loss(data['z_infra_a'].detach(),data['prototypes_infra'],learning_style='vq')
            if alpha_MERGEproto == 0.0:
                loss_merge_proto = torch.Tensor([0.0]).cuda()
                perplexity = torch.Tensor([0.0]).cuda()
            else:
                loss_merge_proto,perplexity = self.ptototype_loss(data['z_merge_a'].detach(),data['prototypes_merge'],learning_style='soft_vq')
        
        ''''    
        if batch_idx <= proto_freeze_batch_idx:
            loss_infra_proto = torch.Tensor([0.0]).cuda()
            loss_merge_proto = torch.Tensor([0.0]).cuda()
            loss_infra_commit = torch.Tensor([0.0]).cuda()
            loss_merge_commit = torch.Tensor([0.0]).cuda()
            perplexity = torch.Tensor([0.0]).cuda()
            

            if alpha_INFRAmetric == 0.0:
                loss_infra_metric = torch.Tensor([0.0]).cuda()
            else:
                loss_infra_metric = self.hard_metric_loss(z_a=data['z_infra_a'],z_pp=data['z_infra_pp'],z_pn=data['z_infra_pn'],z_nn=data['z_infra_nn'],negatives_a_pp_pn=data['negatives_a_pp_pn'],negatives_nn=data['negatives_nn'],pn_equal_pp=True)
            
            if alpha_MERGEmetric == 0.0:
                loss_merge_metric = torch.Tensor([0.0]).cuda()
            else:
                loss_merge_metric = self.hard_metric_loss(z_a=data['z_merge_a'],z_pp=data['z_merge_pp'],z_pn=data['z_merge_pn'],z_nn=data['z_merge_nn'],negatives_a_pp_pn=data['negatives_a_pp_pn'],negatives_nn=data['negatives_nn'])
        
        else:
            if alpha_INFRAmetric == 0.0:
                loss_infra_metric = torch.Tensor([0.0]).cuda()
            else:
                loss_infra_metric = self.hard_proto_metric_loss(z_a=data['z_infra_a'],z_pp=data['z_infra_pp'],z_pn=data['z_infra_pn'],z_nn=data['z_infra_nn'],negatives_a_pp_pn=data['negatives_a_pp_pn'],negatives_nn=data['negatives_nn'],prototypes=data['prototypes_infra'],pn_equal_pp=True)
            
            if alpha_MERGEmetric == 0.0:
                loss_merge_metric = torch.Tensor([0.0]).cuda()
            else:
                loss_merge_metric = self.hard_proto_metric_loss(z_a=data['z_merge_a'],z_pp=data['z_merge_pp'],z_pn=data['z_merge_pn'],z_nn=data['z_merge_nn'],negatives_a_pp_pn=data['negatives_a_pp_pn'],negatives_nn=data['negatives_nn'],prototypes=data['prototypes_merge'])
            
            #required??
            # Proto Loss
            if alpha_INFRAproto == 0.0:
                loss_infra_proto = torch.Tensor([0.0]).cuda()
            else:
                loss_infra_proto = self.ptototype_loss(data['z_infra_a'].detach(),data['prototypes_infra'],learning_style='hard_log')
                
            if alpha_MERGEproto == 0.0:
                loss_merge_proto = torch.Tensor([0.0]).cuda()
                perplexity = torch.Tensor([0.0]).cuda()
            else:
                loss_merge_proto,perplexity = self.ptototype_loss(data['z_merge_a'].detach(),data['prototypes_merge'],learning_style='hard_log')

            # Commitment
            if alpha_INFRAproto == 0.0:
                loss_infra_commit = torch.Tensor([0.0]).cuda()
            else:
                loss_infra_commit = self.ptototype_loss(data['z_infra_a'],data['prototypes_infra'].detach(),learning_style='hard_log')
                
            if alpha_MERGEcommit == 0.0:
                loss_merge_commit = torch.Tensor([0.0]).cuda()
            else:
                loss_merge_commit,_ = self.ptototype_loss(data['z_merge_a'],data['prototypes_merge'].detach(),learning_style='hard_log')
        '''

        # RECON
        if alpha_recon == 0.0:
            reconstruction_loss = torch.Tensor([0.0]).cuda()
        else:
            reconstruction_loss = self.reconstruction_loss_weighted(data['image_target_a'],
                                                                data['image_out_a'],
                                                                beta_traj=beta_traj, 
                                                                beta_infra=beta_infra, 
                                                                beta_back_traj=beta_back_traj, 
                                                                beta_back_infra=beta_back_infra)
                                    
        '''
        reconstruction_loss += self.reconstruction_loss_weighted(data['image_target_pp'],
                                                                data['image_out_pp'],
                                                                beta_traj=beta_traj, 
                                                                beta_infra=beta_infra, 
                                                                beta_back_traj=beta_back_traj, 
                                                                beta_back_infra=beta_back_infra)
        reconstruction_loss += self.reconstruction_loss_weighted(data['image_target_pn'],
                                                                data['image_out_pn'],
                                                                beta_traj=beta_traj, 
                                                                beta_infra=beta_infra, 
                                                                beta_back_traj=beta_back_traj, 
                                                                beta_back_infra=beta_back_infra)
        reconstruction_loss /= 3.0
        '''                

        loss = alpha_INFRAmetric*loss_infra_metric + alpha_MERGEmetric*loss_merge_metric  + alpha_recon*reconstruction_loss + alpha_INFRAproto*loss_infra_proto + alpha_MERGEproto*loss_merge_proto + alpha_INFRAcommit*loss_infra_commit + alpha_MERGEcommit*loss_merge_commit + 1000*(512.0-perplexity)/512.0
        return loss, alpha_recon*reconstruction_loss, alpha_INFRAmetric*loss_infra_metric, alpha_MERGEmetric*loss_merge_metric, alpha_INFRAproto*loss_infra_proto, alpha_MERGEproto*loss_merge_proto, alpha_INFRAcommit*loss_infra_commit, alpha_MERGEcommit*loss_merge_commit,perplexity
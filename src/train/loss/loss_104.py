import torch
import torch.nn as nn
from src.train.loss.loss import loss
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist


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

def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes mse loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: invariance loss (mean squared error).
    """

    return F.mse_loss(z1, z2)


def variance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes variance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: variance regularization loss.
    """

    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    return std_loss


def covariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: covariance regularization loss.
    """

    N, D = z1.size()

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)

    diag = torch.eye(D, device=z1.device)
    cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
    return cov_loss


class loss_104(loss):
    def __init__(self,
                idx = 104,
                name = 'Double SWAV + Weighted Recon Loss',
                description = 'Double SWAV given: anchor, pp, pn +  Weighted Recon Loss',
                input_ = '',
                output = '1D',
                loss_type = 'swav',
                alpha_recon = 10.0,
                alpha_infra = 1.0,
                alpha_merge = 1.0,
                beta_traj = 5.0,
                beta_infra = 5.0,
                beta_back_infra=10.0,
                beta_back_traj=20.0,
                beta_pp = 1.0,
                beta_pn = 1.0,
                beta_nn = 1.0,
                hardest_sampling = False,
                tau = 0.1,
                margin_nn = 1.0,
                margin_pn = 1.0,
                l2 = 0.01,
                sim_type = 'eucl',
                lambda_barlow = 0.0051,
                beta_sim = 25.0,
                beta_var = 25.0,
                beta_cov = 1.0,

                ) -> None:
        super().__init__(idx,name,description,input_,output)
        self.epsilon = 0.05
        self.sinkhorn_iterations = 3
        self.world_size = -1
        self.loss_type=loss_type
        self.alpha_recon=alpha_recon
        self.alpha_infra=alpha_infra
        self.alpha_merge=alpha_merge
        self.beta_traj=beta_traj
        self.beta_infra=beta_infra
        self.beta_back_infra=beta_back_infra
        self.beta_back_traj = beta_back_traj
        self.beta_pp = beta_pp
        self.beta_pn = beta_pn
        self.beta_nn = beta_nn
        self.hardest_sampling = hardest_sampling
        self.tau = tau
        self.margin_nn = margin_nn
        self.margin_pn = margin_pn
        self.l2 = l2 
        self.sim_type = sim_type
        self.lambda_barlow = lambda_barlow
        self.beta_sim  = beta_sim
        self.beta_var = beta_var
        self.beta_cov = beta_cov
               
    
    def hard_metric_loss(self,
                    z_a=None,
                    z_pp=None, 
                    z_pn=None, 
                    z_nn=None, 
                    negatives_a_pp_pn=None,
                    negatives_nn=None,
                    pn_equal_pp = False):

        d_pn = dissimilarity_func(z_a,z_pn,sim_type=self.sim_type)
        d_pp = dissimilarity_func(z_a,z_pp,sim_type=self.sim_type)

        z_all = torch.cat((z_a,z_pp,z_pn,z_nn))
        negatives_all = torch.cat((negatives_a_pp_pn,negatives_a_pp_pn,negatives_a_pp_pn,negatives_nn),dim=1)
        d_all = dissimilarity_func2(z_a,z_all,sim_type=self.sim_type)

        if self.hardest_sampling:
            d_all[torch.le(d_all, d_pn[:,None]) | torch.bitwise_not(negatives_all)] = float('inf')
            idx_max = torch.argmin(d_all,dim=1)[:,None]
        else:
            mask = (torch.gt(d_all, d_pn[:,None]) & negatives_all & torch.lt(d_all, d_pn[:,None]+self.margin_nn))
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
            l_nn            = F.relu(d_pn-d_nn+self.margin_pn)
            l_pn            = F.relu(d_pp-d_nn+self.margin_pn)
        else:
            l_nn            = F.relu(d_pn-d_nn+self.margin_nn)
            l_pn            = F.relu(d_pp-d_pn+self.margin_pn)

        loss = self.beta_nn*l_nn + self.beta_pn*l_pn
        loss = torch.mean(loss)
        if self.l2 != 0:
            loss += self.l2 * (torch.mean(torch.norm(z_a,dim=1)) + torch.mean(torch.norm(z_nn_inside,dim=1)) + torch.mean(torch.norm(z_pn,dim=1)) + torch.mean(torch.norm(z_pp,dim=1)))/4
        return loss

    def soft_metric_loss_dis(self,
                    z_a=None,
                    z_pp=None, 
                    z_pn=None, 
                    z_nn=None, 
                    negatives_a_pp_pn=None,
                    negatives_nn=None,
                    pn_equal_pp = False):

        d_pn = dissimilarity_func(z_a,z_pn,sim_type=self.sim_type)
        d_pp = dissimilarity_func(z_a,z_pp,sim_type=self.sim_type)

        z_all = torch.cat((z_a,z_pp,z_pn,z_nn))
        negatives_all = torch.cat((negatives_a_pp_pn,negatives_a_pp_pn,negatives_a_pp_pn,negatives_nn),dim=1)
        d_all = dissimilarity_func2(z_a,z_all,sim_type=self.sim_type)
    
        if self.hardest_sampling:
            d_all[torch.le(d_all, d_pn[:,None]) | torch.bitwise_not(negatives_all)] = float('inf')
            idx_max = torch.argmin(d_all,dim=1)[:,None]
        else:
            mask = (torch.gt(d_all, d_pn[:,None]) & negatives_all & torch.lt(d_all, d_pn[:,None]+self.margin_nn))
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
            l_nn            = torch.log(1+torch.exp(1/self.tau*(d_pn-d_nn+self.margin_nn)))
            l_pn            = torch.log(1+torch.exp(1/self.tau*(d_pp-d_nn+self.margin_pn)))
        else:
            #l_nn            = -torch.log((torch.exp(1/tau*d_nn))/(torch.exp(1/tau*d_nn)+torch.exp(1/tau*(d_pn+margin_nn))))#F.relu(d_pn                       + margin_nn - d_nn)  
            #l_pn            = -torch.log((torch.exp(1/tau*d_pn))/(torch.exp(1/tau*d_pn)+torch.exp(1/tau*(d_pp+margin_pn))))#F.relu(d_pp_margin                + margin_pn - d_pn)
            l_nn            = torch.log(1+torch.exp(1/self.tau*(d_pn-d_nn+self.margin_nn)))
            l_pn            = torch.log(1+torch.exp(1/self.tau*(d_pp-d_pn+self.margin_pn)))
        #l_pp            = (d_in_pp*margin_pp - d_pp)**2  TODO: put that to traj space?
        loss = self.beta_nn*l_nn + self.beta_pn*l_pn
        loss = self.tau*torch.mean(loss)
        if self.l2 != 0:
            loss += self.l2 * (torch.mean(torch.norm(z_a,dim=1)) + torch.mean(torch.norm(z_nn_inside,dim=1)) + torch.mean(torch.norm(z_pn,dim=1)) + torch.mean(torch.norm(z_pp,dim=1)))/4

        return loss

    def soft_metric_loss(self,
                    z_a=None,
                    z_pp=None, 
                    z_pn=None, 
                    z_nn=None, 
                    negatives_a_pp_pn=None,
                    negatives_nn=None,
                    pn_equal_pp = False):

        s_pn = similarity_func(z_a,z_pn,sim_type=self.sim_type)
        s_pp = similarity_func(z_a,z_pp,sim_type=self.sim_type)

        z_all = torch.cat((z_a,z_pp,z_pn,z_nn))
        negatives_all = torch.cat((negatives_a_pp_pn,negatives_a_pp_pn,negatives_a_pp_pn,negatives_nn),dim=1)
        s_all = similarity_func2(z_a,z_all,sim_type=self.sim_type)

       
        if self.hardest_sampling:
            s_all[torch.ge(s_all, s_pn[:,None]) | torch.bitwise_not(negatives_all)] = 0
            idx_max = torch.argmax(s_all,dim=1)[:,None]
        else:
            mask = (torch.lt(s_all, s_pn[:,None]) & negatives_all)
            idx_vect = torch.arange(0,s_all.shape[1],dtype=torch.int64,device="cuda")
            idx_max = torch.zeros(z_a.shape[0],dtype=torch.int64,device="cuda")
            for i in range(z_a.shape[0]):
                idx_temp = idx_vect[mask[i,:]]
                if len(idx_temp) ==0:
                    idx_max[i] = z_a.shape[0]+z_pp.shape[0]+z_pn.shape[0]+i
                else:
                    idx_max[i] = idx_temp[torch.randint(len(idx_temp),(1,))]
            idx_max = idx_max[:,None]
        
        s_nn = s_all.gather(1,idx_max)[:,0]
        z_nn_inside = z_all[idx_max[:,0],:]
        if pn_equal_pp:
            l_nn            = -torch.log((torch.exp(1/self.tau*s_pn))/(torch.exp(1/self.tau*s_pn)+torch.exp(1/self.tau*(s_nn+self.margin_nn))))#F.relu(d_pn                       + margin_nn - d_nn)  
            l_pn            = -torch.log((torch.exp(1/self.tau*s_pp))/(torch.exp(1/self.tau*s_pp)+torch.exp(1/self.tau*(s_nn+self.margin_nn))))#F.relu(d_pp_margin                + margin_pn - d_pn)
        else:
            l_nn            = -torch.log((torch.exp(1/self.tau*s_pn))/(torch.exp(1/self.tau*s_pn)+torch.exp(1/self.tau*(s_nn+self.margin_nn))))#F.relu(d_pn                       + margin_nn - d_nn)  
            l_pn            = -torch.log((torch.exp(1/self.tau*s_pp))/(torch.exp(1/self.tau*s_pp)+torch.exp(1/self.tau*(s_pn+self.margin_pn))))#F.relu(d_pp_margin                + margin_pn - d_pn)
        #l_pp            = (d_in_pp*margin_pp - d_pp)**2  TODO: put that to traj space?
        loss = self.beta_nn*l_nn + self.beta_pn*l_pn
        loss = self.tau*torch.mean(loss)
        if self.l2 != 0:
            loss += self.l2 * (torch.mean(torch.norm(z_a,dim=1)) + torch.mean(torch.norm(z_nn_inside,dim=1)) + torch.mean(torch.norm(z_pn,dim=1)) + torch.mean(torch.norm(z_pp,dim=1)))/4
        return loss
    
    def swav_loss(self, output):
        #output (b,2,laten_dim)
        n_examples = output.shape[0]

        # ============ swav loss ... ============
        loss = 0
        for i in np.arange(n_examples):
            with torch.no_grad():
                out = output[i,:,:].detach()
                q = self.distributed_sinkhorn(out)

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(n_examples), i):
                x = output[v,:,:]/self.tau
                subloss -= torch.mean(
                    torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (n_examples - 1)
        loss /= n_examples
        if torch.any(torch.isnan(loss)):
            print("test")

        return loss

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] #* self.world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        #dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            #dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        if torch.any(torch.isnan(Q)):
            print("test")
        return Q.t()

    def barlow_twins_loss(self,
                out_1,
                out_2):
        batch_size = out_1.size(0)
        D = out_1.size(1)
        # cross-correlation matrix
        c = torch.mm(out_1.T, out_2) / (batch_size * self.world_size)
        if self.world_size > 1:
            dist.all_reduce(c)
        # loss
        c_diff = (c - torch.eye(D, device="cuda")).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_barlow
        loss = c_diff.sum()

        return loss

    def vicreg_loss(self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Computes VICReg's loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
            sim_loss_weight (float): invariance loss weight.
            var_loss_weight (float): variance loss weight.
            cov_loss_weight (float): covariance loss weight.
        Returns:
            torch.Tensor: VICReg loss.
        """

        sim_loss = invariance_loss(z1, z2)
        var_loss = variance_loss(z1, z2)
        cov_loss = covariance_loss(z1, z2)

        loss = self.beta_sim * sim_loss + self.beta_var * var_loss + self.beta_cov * cov_loss
        return loss

    def reconstruction_loss_weighted(self,
                                    x,x_pred):
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
        l = self.beta_traj * l_traj + self.beta_infra * l_infra + self.beta_back_traj * l_back_traj + self.beta_back_infra * l_back_infra
        return l
    
    def _forward_hard_metric(self,data):
        """
        Computes loss for each batch.
        """
        # METRIC INFRA
        if self.alpha_infra == 0.0:
            loss_infra = torch.Tensor([0.0]).cuda()
        else:
            loss_infra = self.hard_metric_loss(z_a=data['z_infra_a'],z_pp=data['z_infra_pp'],z_pn=data['z_infra_pn'],z_nn=data['z_infra_nn'],negatives_a_pp_pn=data['negatives_a_pp_pn'],negatives_nn=data['negatives_nn'],pn_equal_pp=True)
        
        # METRIC MERGE
        if self.alpha_merge == 0.0:
            loss_merge = torch.Tensor([0.0]).cuda()
        else:
            loss_merge = self.hard_metric_loss(z_a=data['z_merge_a'],z_pp=data['z_merge_pp'],z_pn=data['z_merge_pn'],z_nn=data['z_merge_nn'],negatives_a_pp_pn=data['negatives_a_pp_pn'],negatives_nn=data['negatives_nn'])

        # RECON
        if self.alpha_recon == 0.0:
            reconstruction_loss = torch.Tensor([0.0]).cuda()
        else:
            reconstruction_loss = self.reconstruction_loss_weighted(data['image_target_a'],data['image_out_a'])
  
        loss = self.alpha_infra*loss_infra + self.alpha_merge*loss_merge  + self.alpha_recon*reconstruction_loss
        return loss, self.alpha_recon*reconstruction_loss, self.alpha_infra*loss_infra, self.alpha_merge*loss_merge

    def _forward_soft_metric_dis(self,data):
        """
        Computes loss for each batch.
        """
        # METRIC INFRA
        if self.alpha_infra == 0.0:
            loss_infra = torch.Tensor([0.0]).cuda()
        else:
            loss_infra = self.soft_metric_loss_dis(z_a=data['z_infra_a'],z_pp=data['z_infra_pp'],z_pn=data['z_infra_pn'],z_nn=data['z_infra_nn'],negatives_a_pp_pn=data['negatives_a_pp_pn'],negatives_nn=data['negatives_nn'],pn_equal_pp=True)
        
        # METRIC MERGE
        if self.alpha_merge == 0.0:
            loss_merge = torch.Tensor([0.0]).cuda()
        else:
            loss_merge = self.soft_metric_loss_dis(z_a=data['z_merge_a'],z_pp=data['z_merge_pp'],z_pn=data['z_merge_pn'],z_nn=data['z_merge_nn'],negatives_a_pp_pn=data['negatives_a_pp_pn'],negatives_nn=data['negatives_nn'])

        # RECON
        if self.alpha_recon == 0.0:
            reconstruction_loss = torch.Tensor([0.0]).cuda()
        else:
            reconstruction_loss = self.reconstruction_loss_weighted(data['image_target_a'],data['image_out_a'])
  
        loss = self.alpha_infra*loss_infra + self.alpha_merge*loss_merge  + self.alpha_recon*reconstruction_loss
        return loss, self.alpha_recon*reconstruction_loss, self.alpha_infra*loss_infra, self.alpha_merge*loss_merge

    def _forward_soft_metric(self,data):
        """
        Computes loss for each batch.
        """
        # METRIC INFRA
        if self.alpha_infra == 0.0:
            loss_infra = torch.Tensor([0.0]).cuda()
        else:
            loss_infra = self.soft_metric_loss(z_a=data['z_infra_a'],z_pp=data['z_infra_pp'],z_pn=data['z_infra_pn'],z_nn=data['z_infra_nn'],negatives_a_pp_pn=data['negatives_a_pp_pn'],negatives_nn=data['negatives_nn'],pn_equal_pp=True)
        
        # METRIC MERGE
        if self.alpha_merge == 0.0:
            loss_merge = torch.Tensor([0.0]).cuda()
        else:
            loss_merge = self.soft_metric_loss(z_a=data['z_merge_a'],z_pp=data['z_merge_pp'],z_pn=data['z_merge_pn'],z_nn=data['z_merge_nn'],negatives_a_pp_pn=data['negatives_a_pp_pn'],negatives_nn=data['negatives_nn'])

        # RECON
        if self.alpha_recon == 0.0:
            reconstruction_loss = torch.Tensor([0.0]).cuda()
        else:
            reconstruction_loss = self.reconstruction_loss_weighted(data['image_target_a'],data['image_out_a'])
  
        loss = self.alpha_infra*loss_infra + self.alpha_merge*loss_merge  + self.alpha_recon*reconstruction_loss
        return loss, self.alpha_recon*reconstruction_loss, self.alpha_infra*loss_infra, self.alpha_merge*loss_merge
    
    def _forward_swav(self,data):
        """
        Computes loss for each batch.
        """

        # SWAV
        if self.alpha_infra == 0.0:
            loss_infra = torch.Tensor([0.0]).cuda()
        else:
            batch_infra_1 = torch.stack((data['cXz_infra_a'],data['cXz_infra_pp']))
            batch_infra_2 = torch.stack((data['cXz_infra_a'],data['cXz_infra_pn']))
            loss_infra = self.beta_pp * self.swav_loss(batch_infra_1) + self.beta_pn * self.swav_loss(batch_infra_2)
        
        if self.alpha_merge == 0.0:
            loss_merge = torch.Tensor([0.0]).cuda()
        else:
            batch_merge_1 = torch.stack((data['cXz_merge_a'],data['cXz_merge_pp']))
            batch_merge_2 = torch.stack((data['cXz_merge_a'],data['cXz_merge_pn']))
            loss_merge = self.beta_pp * self.swav_loss(batch_merge_1) + self.beta_pn * self.swav_loss(batch_merge_2)
        
        # RECON
        if self.alpha_recon == 0.0:
            reconstruction_loss = torch.Tensor([0.0]).cuda()
        else:
            reconstruction_loss = self.reconstruction_loss_weighted(data['image_target_a'],data['image_out_a'])
  
        loss = self.alpha_infra*loss_infra + self.alpha_merge*loss_merge  + self.alpha_recon*reconstruction_loss
        return loss, self.alpha_recon*reconstruction_loss, self.alpha_infra*loss_infra, self.alpha_merge*loss_merge

    def _forward_barlow_twins(self,data):
        """
        Computes loss for each batch.
        """

        # SWAV
        if self.alpha_infra == 0.0:
            loss_infra = torch.Tensor([0.0]).cuda()
        else:
            loss_infra = self.beta_pp * self.barlow_twins_loss(data['z_infra_proj_b_norm_a'],data['z_infra_proj_b_norm_pp']) + self.beta_pn * self.barlow_twins_loss(data['z_infra_proj_b_norm_a'],data['z_infra_proj_b_norm_pn'])
        
        if self.alpha_merge == 0.0:
            loss_merge = torch.Tensor([0.0]).cuda()
        else:
            loss_merge = self.beta_pp * self.barlow_twins_loss(data['z_merge_proj_b_norm_a'],data['z_merge_proj_b_norm_pp']) + self.beta_pn * self.barlow_twins_loss(data['z_merge_proj_b_norm_a'],data['z_merge_proj_b_norm_pn'])
 
        # RECON
        if self.alpha_recon == 0.0:
            reconstruction_loss = torch.Tensor([0.0]).cuda()
        else:
            reconstruction_loss = self.reconstruction_loss_weighted(data['image_target_a'],data['image_out_a'])
  
        loss = self.alpha_infra*loss_infra + self.alpha_merge*loss_merge  + self.alpha_recon*reconstruction_loss
        return loss, self.alpha_recon*reconstruction_loss, self.alpha_infra*loss_infra, self.alpha_merge*loss_merge
    
    def _forward_vicreg(self,data):
        """
        Computes loss for each batch.
        """

        # SWAV
        if self.alpha_infra == 0.0:
            loss_infra = torch.Tensor([0.0]).cuda()
        else:
            loss_infra = self.beta_pp * self.vicreg_loss(data['z_infra_proj_a'],data['z_infra_proj_pp']) + self.beta_pn * self.vicreg_loss(data['z_infra_proj_a'],data['z_infra_proj_pn'])
        
        if self.alpha_merge == 0.0:
            loss_merge = torch.Tensor([0.0]).cuda()
        else:
            loss_merge = self.beta_pp * self.vicreg_loss(data['z_merge_proj_a'],data['z_merge_proj_pp']) + self.beta_pn * self.vicreg_loss(data['z_merge_proj_a'],data['z_merge_proj_pn'])
 
        # RECON
        if self.alpha_recon == 0.0:
            reconstruction_loss = torch.Tensor([0.0]).cuda()
        else:
            reconstruction_loss = self.reconstruction_loss_weighted(data['image_target_a'],data['image_out_a'])
  
        loss = self.alpha_infra*loss_infra + self.alpha_merge*loss_merge  + self.alpha_recon*reconstruction_loss
        return loss, self.alpha_recon*reconstruction_loss, self.alpha_infra*loss_infra, self.alpha_merge*loss_merge
    
    def update_hypers(self,
                    alpha_recon=None,
                    alpha_infra=None,
                    alpha_merge=None,
                    beta_traj=None,
                    beta_infra=None,
                    beta_back_infra=None,
                    beta_back_traj=None,
                    beta_pp=None,
                    beta_pn=None,
                    beta_nn=None,
                    hardest_sampling=None,
                    tau=None,
                    margin_nn=None,
                    margin_pn=None,
                    l2=None,
                    sim_type=None,
                    lambda_barlow=None,
                    beta_sim=None,
                    beta_var=None,
                    beta_cov=None):
        if not(alpha_recon is None):
            self.alpha_recon = alpha_recon
        if not(alpha_infra is None):
            self.alpha_infra = alpha_infra
        if not(alpha_merge is None):
            self.alpha_merge = alpha_merge
        if not(beta_traj is None):
            self.beta_traj=beta_traj
        if not(beta_infra is None):
            self.beta_infra=beta_infra
        if not(beta_back_infra is None):
            self.beta_back_infra=beta_back_infra
        if not(beta_back_traj is None):
            self.beta_back_traj = beta_back_traj
        if not(beta_pp is None):
            self.beta_pp = beta_pp
        if not(beta_pn is None):
            self.beta_pn = beta_pn
        if not(beta_nn is None):
            self.beta_nn = beta_nn
        if not(hardest_sampling is None):
            self.hardest_sampling = hardest_sampling
        if not(tau is None):
            self.tau = tau
        if not(margin_nn is None):
            self.margin_nn = margin_nn
        if not(margin_pn is None):
            self.margin_pn = margin_pn
        if not(l2 is None):
            self.l2 = l2
        if not( sim_type is None):
            self.sim_type = sim_type
        if not( lambda_barlow is None):
            self.lambda_barlow = lambda_barlow
        if not( beta_sim is None):
            self.beta_sim  = beta_sim
        if not( beta_var is None):
            self.beta_var = beta_var
        if not( beta_cov is None):
            self.beta_cov = beta_cov

    def forward(self,data,**kwargs):
        self.update_hypers(**kwargs)
        
        if self.loss_type == 'hard_metric':
            return self._forward_hard_metric(data)
        elif self.loss_type == 'soft_metric_dis':
            return self._forward_soft_metric_dis(data)
        elif self.loss_type == 'soft_metric':
            return self._forward_soft_metric(data)
        elif self.loss_type == 'swav':
            return self._forward_swav(data)
        elif self.loss_type == 'barlow_twins':
            return self._forward_barlow_twins(data)
        elif self.loss_type == 'vicreg':
            return self._forward_vicreg(data)
        else:
            return self._forward_hard_metric(data)
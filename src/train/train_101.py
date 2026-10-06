import logging
from os import device_encoding
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.train.train import Training
from src.utils.average_meter import AverageMeter
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision
from einops import rearrange, repeat

def prep_data(input_data, cuda):
    """
    Takes a batch of tuplets and converts them into Pytorch variables 
    and puts them on GPU if available.
    """
    input_data_out = dict((k, Variable(v)) for k,v in input_data.items())
    input_data = input_data_out
    
    if cuda:
        input_data_out = dict((k, v.cuda()) for k,v in input_data.items())
    return input_data_out
# Following code is from lucidrains sample_vectors and kmeans
def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def kmeans(samples, num_clusters, num_iters = 10, use_cosine_sim = False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        #if use_cosine_sim:
        #    dists = samples @ means.t()
        #else:
        diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
        dists = -(diffs ** 2).sum(dim = -1)

        buckets = dists.max(dim = -1).indices
        bins = torch.bincount(buckets, minlength = num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        #if use_cosine_sim:
        #    new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins
class train_101(Training):
    def __init__(self,
                 idx=101,
                 crops_for_assignment=None,
                 nmb_crops=None,
                 temperature=0.1,
                 freeze_prototypes_niters=600,
                 epsilon=0.05,
                 queue=None,
                 sinkhorn_iterations=3,
                 **kwargs) -> None:
        super().__init__()
        if nmb_crops is None:
            nmb_crops = [2]
        if crops_for_assignment is None:
            crops_for_assignment = [0, 1]
        self.description = "Double SWAV + RECON"
        self.temperature = temperature
        self.freeze_prototypes_niters = freeze_prototypes_niters
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations #--> loss def?
        self.queue = queue

    def run_training(self, model, dataloader_train, loss_fc, optimizer, _,dataloader_test):
        self._train(model, dataloader_train, loss_fc, optimizer,dataloader_test)

    def _train(self, model, dataloader_train, loss_fc, optimizer,dataloader_test):
        epochs = dataloader_train.epochs
        pbar = tqdm(total=int(epochs * len(dataloader_train.dataset) /
                              dataloader_train.batch_size),
                    desc="init training...".center(50))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        writer = SummaryWriter()
        dataloader_test = dataloader_test()
        z_test = torch.zeros((len(dataloader_test.dataset)+512,model.z_dim_m))
        labels_test = [10]*(len(dataloader_test.dataset)+512)
        print_N = 16
        dummy = torch.zeros((print_N,1,100,100))
        dummy2 = torch.zeros((print_N*2,1,100,100))
        batch_pass_total = 0
        for epoch in range(epochs):
            loss_record = AverageMeter()
            loss_record_recon = AverageMeter()
            loss_record_infra = AverageMeter()
            loss_record_merge = AverageMeter()
            loss_record_merge_proto = AverageMeter()
            loss_record_infra_proto = AverageMeter()
            loss_record_merge_commit = AverageMeter()
            loss_record_infra_commit = AverageMeter()
            loss_record_perplexity = AverageMeter()
            batch_pass = 0
            dataloader = dataloader_train(epoch, rank=0)
            for batch_idx, input_data in enumerate(dataloader,start=epoch * len(dataloader)):
                # Save the projection
                
                
                
                
                #input_data = prep_data(input_data, device)

                # normalize the prototypes
                #with torch.no_grad():
                #    model.normalize_prototypes()

                output_a  = model(input_data[0][0].cuda(),(input_data[0][1].cuda(),input_data[0][2]),return_infra_z=True, return_infra_prototypes=True, return_merge_prototypes=True)
                output_pp = model(input_data[1][0].cuda(),(input_data[1][1].cuda(),input_data[1][2]),no_recon=True,return_infra_z=True, return_infra_prototypes=True, return_merge_prototypes=True)
                output_pn = model(input_data[2][0].cuda(),(input_data[2][1].cuda(),input_data[2][2]),no_recon=True,return_infra_z=True, return_infra_prototypes=True, return_merge_prototypes=True)
                output_nn = model(input_data[3][0].cuda(),(input_data[3][1].cuda(),input_data[3][2]),no_recon=True,return_infra_z=True, return_infra_prototypes=True, return_merge_prototypes=True)
                # output_all = {'cXz_infra_a'  :output_a['cXz_infra'],
                #             'cXz_infra_pp' :output_pp['cXz_infra'],
                #             'cXz_infra_pn' :output_pn['cXz_infra'],
                #             'cXz_infra_nn' :output_nn['cXz_infra'],
                #             'cXz_merge_a'  :output_a['cXz_merge'],
                #             'cXz_merge_pp' :output_pp['cXz_merge'],
                #             'cXz_merge_pn' :output_pn['cXz_merge'],
                #             'cXz_merge_nn' :output_nn['cXz_merge'],
                #             'image_target_a':input_data[0][3].cuda(),
                #             'image_out_a'   :output_a['x'],
                #             'negatives_a_pp_pn':input_data[0][4].cuda(),
                #             'negatives_nn':input_data[0][5].cuda()}
                output_all = {'z_infra_a'  :output_a['z_infra'],
                            'z_infra_pp' :output_pp['z_infra'],
                            'z_infra_pn' :output_pn['z_infra'],
                            'z_infra_nn' :output_nn['z_infra'],
                            'z_merge_a'  :output_a['z'],
                            'z_merge_pp' :output_pp['z'],
                            'z_merge_pn' :output_pn['z'],
                            'z_merge_nn' :output_nn['z'],
                            'prototypes_infra'  :output_a['prototypes_infra'],
                            'prototypes_merge'  :output_a['prototypes_merge'],
                            'image_target_a':input_data[0][3].cuda(),
                            'image_out_a'   :output_a['x'],
                            'negatives_a_pp_pn':input_data[4].cuda(),
                            'negatives_nn':input_data[5].cuda()}
                batch_pass += 1
                batch_pass_total += 1
                if batch_idx % 600 == 0:
                    for batch_idx_test,proj_data in enumerate(dataloader_test):
                        image_in = proj_data[0].cuda()
                        traj_in = (proj_data[1].cuda(),proj_data[2])
                        output_test  = model(image_in,traj_in)
                        z_test[(batch_idx_test)*dataloader_test.batch_size:(batch_idx_test)*dataloader_test.batch_size+output_test['z'].shape[0],:] = output_test['z'].detach().cpu()
                        labels_test[(batch_idx_test)*dataloader_test.batch_size:(batch_idx_test)*dataloader_test.batch_size+output_test['z'].shape[0]] = [int(val) for val in proj_data[4]]
                    z_test[-512:,] = output_a['prototypes_merge']
                    writer.add_embedding(z_test, metadata=labels_test,global_step=batch_idx)
                if batch_idx % 100 == 0:
                    temp = output_all['image_out_a'].cpu()
                    image_in        = torch.cat((input_data[0][0][:print_N,0,:,:][:,None,:,:],dummy,dummy),axis=1)
                    image_print_out = torch.cat((temp[:print_N,0,:,:][:,None,:,:],dummy,temp[:print_N,1,:,:][:,None,:,:]),axis=1)
                    temp2 = model.decode(output_a['prototypes_merge'][:print_N*2,:]).cpu()
                    image_print_proto = torch.cat((temp2[:print_N*2,0,:,:][:,None,:,:],dummy2,temp2[:print_N*2,1,:,:][:,None,:,:]),axis=1)
                    image_print_tgt = torch.cat((input_data[0][3][:print_N,0,:,:][:,None,:,:],dummy,input_data[0][3][:print_N,1,:,:][:,None,:,:]),axis=1)
                    writer.add_images('images_tgt', image_print_tgt, batch_idx)
                    writer.add_images('images_out', image_print_out, batch_idx)
                    writer.add_images('images_in', image_in, batch_idx)
                    writer.add_images('images_proto', image_print_proto, batch_idx)
                
                optimizer.zero_grad()
                
                loss,loss_recon,loss_infra,loss_merge,loss_infra_proto,loss_merge_proto,loss_infra_commit,loss_merge_commit,perplexity = loss_fc(output_all,batch_pass_total,self.freeze_prototypes_niters)

                if batch_pass_total == self.freeze_prototypes_niters:
                    #do kmeans here
                    means, _= kmeans(samples=torch.cat((output_all['z_merge_a'],output_all['z_merge_pp'],output_all['z_merge_pn'],output_all['z_merge_nn'])),num_clusters=output_all ['prototypes_merge'].shape[0])
                    model.encoder.merge_prototypes.data.copy_(means)
                    del means

                # ============ backward and optim step ... ============
                
                

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                
                optimizer.step()

                pbar.update(1)
                loss_record.update(loss.item(), output_a['z'].size(0))
                loss_record_recon.update(loss_recon.item(), output_a['z'].size(0))
                loss_record_infra.update(loss_infra.item(), output_a['z'].size(0))
                loss_record_merge.update(loss_merge.item(), output_a['z'].size(0))
                loss_record_merge_proto.update(loss_merge_proto.item(), output_a['z'].size(0))
                loss_record_infra_proto.update(loss_infra_proto.item(), output_a['z'].size(0))
                loss_record_merge_commit.update(loss_merge_proto.item(), output_a['z'].size(0))
                loss_record_infra_commit.update(loss_infra_proto.item(), output_a['z'].size(0))
                loss_record_perplexity.update(perplexity.item(), output_a['z'].size(0))
                log_msg = "Epoch:{:2}/{}  Iter:{:3}/{} Avg Loss: {:6.3f}".format(
                    epoch + 1, epochs, batch_pass, len(dataloader),
                    round(loss_record.avg, 3)).center(50)

                pbar.set_description(log_msg)
                writer.add_scalar("0 Train Loss", loss, batch_idx)
                writer.add_scalar("1 Train Loss - AVG", loss_record.avg, batch_idx)
                writer.add_scalar("2 Infra Loss - AVG", loss_record_infra.avg, batch_idx)
                writer.add_scalar("5 Merge Loss - AVG", loss_record_merge.avg, batch_idx)
                writer.add_scalar("8 Recon Loss - AVG", loss_record_recon.avg, batch_idx)
                writer.add_scalar("6 Merge Proto Loss - AVG", loss_record_merge_proto.avg, batch_idx)
                writer.add_scalar("3 Infra Proto Loss - AVG", loss_record_infra_proto.avg, batch_idx)
                writer.add_scalar("7 Merge Commit Loss - AVG", loss_record_merge_commit.avg, batch_idx)
                writer.add_scalar("4 Infra Commit Loss - AVG", loss_record_infra_commit.avg, batch_idx)
                writer.add_scalar("9 Merge Perplexity - AVG", loss_record_perplexity.avg, batch_idx)
                logging.info(log_msg)



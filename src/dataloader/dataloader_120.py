import random
from scipy.io import loadmat
from glob import glob
import os

import numpy as np
import torch
import torchnet as tnt
import math

from torch.nn.utils.rnn import pad_sequence


class dataloader_120(object):
    def __init__(
            self,
            idx=120,
            dataset=None,
            batch_size=None,
            epochs=None,
            num_workers=0,
            num_gpus=1,
            shuffle=False,
            epoch_size=None,
            transformation=None,
            transformation3D=None,
            representation='TODO',
            test=False,
            grid_chosen=None,
            features_dim=3,
            channels=3,
            name='TODO',
            description='with time'
    ):
        self.dataset = dataset[0]
        self.epoch_size = epoch_size if epoch_size is not None else len(
            self.dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers,
        self.epochs = epochs
        self.features_dim = features_dim
        self.test = test
        self.num_gpus = num_gpus
        self.channels = channels

    def _load_function(self, idx):
        idx = idx % len(self.dataset)
        sample = self.dataset[idx]
        return sample

    def _collate_fun(self, batch):
        hist_objs = []
        hist_objs_seq_len = []
        hist_obj_lens = []
        pred_objsx = []
        pred_objsy = []
        pred_objst = []
        pred_objsv = []
        pred_objspsi = []
        pred_objs_a_lon = []
        pred_objs_a_lat = []

        pred_objs_seq_len = []
        pred_obj_lens = []
        multi_channel_images = torch.zeros(len(batch), self.channels, self.dataset.bbox_pixel[0],
                                           self.dataset.bbox_pixel[1])
        file_names = []
        obj_decoder_in = []
        center_collected = []
        orientation = []
        for idx, elems in enumerate(batch):
            #TODO move to getitem
            multi_channel_images[idx, :, :, :] = elems['image']
            file_names.append(elems['filename'])
            center_collected.append(elems['center'])
            orientation.append(elems['orientation'])
            hist_objs.append(elems['traj_hist_obj'])
            hist_objs_seq_len += elems['traj_hist_obj_seq_lens']
            hist_obj_lens.append(elems['traj_hist_obj'].shape[0])
            if elems['traj_pred_obj'].shape[1] > 30:
                pred_objsx.append(elems['traj_pred_obj'][:, :30, 0].unsqueeze(2))
                pred_objsy.append(elems['traj_pred_obj'][:, :30, 1].unsqueeze(2))
                pred_objst.append(elems['traj_pred_obj'][:, :30, 2].unsqueeze(2))
                if self.features_dim == 5:
                    pred_objsv.append(elems['traj_pred_obj'][:, :30, 4].unsqueeze(2))
                    pred_objspsi.append(elems['traj_pred_obj'][:, :30, 3].unsqueeze(2))
            else:
                pred_objsx.append(elems['traj_pred_obj'][:, :, 0].unsqueeze(2))
                pred_objsy.append(elems['traj_pred_obj'][:, :, 1].unsqueeze(2))
                pred_objst.append(elems['traj_pred_obj'][:, :, 2].unsqueeze(2))
                if self.features_dim == 5:
                    pred_objsv.append(elems['traj_pred_obj'][:, :, 4].unsqueeze(2))
                    pred_objspsi.append(elems['traj_pred_obj'][:, :, 3].unsqueeze(2))
                    pred_objs_a_lon.append(elems['traj_pred_obj'][:, :, 5].unsqueeze(2))
                    pred_objs_a_lat.append(elems['traj_pred_obj'][:, :, 6].unsqueeze(2))
            pred_objs_seq_len += elems['traj_pred_obj_seq_lens']
            pred_obj_lens.append(elems['traj_pred_obj'].shape[0])
            obj_decoder_in.append(elems['obj_decoder_in'])
        # padding to handle different number of objects
        hist_objs = torch.nn.utils.rnn.pad_sequence(hist_objs,
                                                    batch_first=True)  # batch_size x max_obj x max_obj_seq_len x feature_dim
        pred_objsx = torch.nn.utils.rnn.pad_sequence(pred_objsx, batch_first=True,
                                                     padding_value=torch.nan)  # batch_size x max_obj x max_obj_seq_len x feature_dim
        pred_objsy = torch.nn.utils.rnn.pad_sequence(pred_objsy, batch_first=True,
                                                     padding_value=torch.nan)  # batch_size x max_obj x max_obj_seq_len x feature_dim
        pred_objst = torch.nn.utils.rnn.pad_sequence(pred_objst, batch_first=True,
                                                     padding_value=torch.nan)  # batch_size x max_obj x max_obj_seq_len x feature_dim
        if self.features_dim == 5:
            pred_objsv = torch.nn.utils.rnn.pad_sequence(pred_objsv, batch_first=True,
                                                         padding_value=torch.nan)  # batch_size x max_obj x max_obj_seq_len x feature_dim
            pred_objspsi = torch.nn.utils.rnn.pad_sequence(pred_objspsi, batch_first=True,
                                                           padding_value=torch.nan)  # batch_size x max_obj x max_obj_seq_len x feature_dim
            pred_objs_a_lon = torch.nn.utils.rnn.pad_sequence(pred_objs_a_lon, batch_first=True,
                                                           padding_value=torch.nan)  # batch_size x max_obj x max_obj_seq_len x feature_dim
            pred_objs_a_lat = torch.nn.utils.rnn.pad_sequence(pred_objs_a_lat, batch_first=True,
                                                           padding_value=torch.nan)  # batch_size x max_obj x max_obj_seq_len x feature_dim
        obj_decoder_in = torch.nn.utils.rnn.pad_sequence(obj_decoder_in, batch_first=True)
        hist_objs_seq_len = [ele for ele in hist_objs_seq_len]
        hist_object_lengths_sum = torch.cumsum(torch.Tensor(hist_obj_lens), 0).int()
        hist_object_lengths_sum = torch.cat([torch.Tensor([0]),
                                             hist_object_lengths_sum]).int()  # num of objects per scene , cumsum of the number of objects in the scene
        pred_objs_seq_len = [ele for ele in pred_objs_seq_len]
        # valid_values = pred_objspsi[~torch.isnan(pred_objspsi)]

        # Check if there are any valid values
        # if valid_values.numel() > 0:
        #     real_min_value = valid_values.min()
        #     print("Lowest real value:", real_min_value)
        # else:
        #     print("All values are NaN")
        pres_object_lengths_sum = torch.cumsum(torch.Tensor(pred_obj_lens), 0).int()
        pres_object_lengths_sum = torch.cat([torch.Tensor([0]), pres_object_lengths_sum]).int()
        center_collected = torch.Tensor(np.array(center_collected))
        orientation = torch.Tensor(orientation)
        conditions_acc = []
        conditions_goal_point = []
        conditions_v = []
        for sample in batch:
            cond = sample["conditions"]
            t = torch.tensor(cond["acc"])
            t = t.unsqueeze(-1)
            conditions_acc.append(t)
            t = torch.stack(cond["goal_point"])
            conditions_goal_point.append(t)
            cond_v = torch.tensor(cond["v"])
            cond_v = cond_v.unsqueeze(-1)
            conditions_v.append(cond_v)


        padded_acc = pad_sequence(conditions_acc, batch_first=True, padding_value=0)
        padded_v = pad_sequence(conditions_v, batch_first=True, padding_value=0)


        padded_goal_point = pad_sequence(conditions_goal_point, batch_first=True, padding_value=0)
        full_trajectory_pred = "TODO"
        # full_trajectory_hist = hist_objs[:,:,:,:2]
        # full_trajectory_pred = torch.cat([pred_objsx, pred_objsy], dim=3)
        # full_trajectory = torch.cat([full_trajectory_hist, full_trajectory_pred], dim=2)
        # nan_count = torch.isnan(full_trajectory).sum()#b,n,50,2
        # lateral_accel = torch.diff(full_trajectory, dim=2).norm(dim=-1).mean(dim=-1)
        # lateral_accel = torch.nan_to_num(lateral_accel)
        # Classify vehicles based on thresholds
        vehicle_class = 1 #torch.bucketize(lateral_accel, torch.tensor([1.0, 2.5]))  # 0: calm, 1: moderate, 2: aggressive
        out_dict = {'images': multi_channel_images, 'hist_objs': hist_objs, 'hist_obj_lens': hist_obj_lens,
                    'hist_objs_seq_len': hist_objs_seq_len,
                    'pred_objsx': pred_objsx, 'pred_objsy': pred_objsy, 'pred_objst': pred_objst,
                    'pred_objsv': pred_objsv, 'pred_objspsi': pred_objspsi,'pred_objs_a_lon':pred_objs_a_lon,'pred_objs_a_lat':pred_objs_a_lat,
                    'pred_obj_lens': pred_obj_lens, 'pred_objs_seq_len': pred_objs_seq_len,
                    'obj_decoder_in': obj_decoder_in,
                    'cond_goal_point': padded_goal_point, 'cond_acc': padded_acc,'cond_v':padded_v,
                    'hist_object_lengths_sum': hist_object_lengths_sum,
                    'vehicle_class': vehicle_class,
                    'pres_object_lengths_sum': pres_object_lengths_sum, 'file_names': file_names,
                    'center_collected': center_collected, 'orientation': orientation}
        return out_dict

    def get_iterator(self, epoch, gpu_idx):
        self.rand_seed1 = epoch

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
                                              load=self._load_function)

        sampler = torch.utils.data.distributed.DistributedSampler(
            tnt_dataset,
            num_replicas=self.num_gpus,
            shuffle=self.shuffle,
            rank=gpu_idx)
        sampler.set_epoch(epoch)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=self._collate_fun,
                                           num_workers=self.num_workers[0],
                                           sampler=sampler, pin_memory=True)
        return data_loader

    def __call__(self, epoch=0, rank=0):
        return self.get_iterator(epoch, rank)

    def __len__(self):
        return math.ceil((len(self.dataset) / self.batch_size) / self.num_gpus)

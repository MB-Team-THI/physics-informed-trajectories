from scipy.io import loadmat
import torch
import numpy


from src.utils.rot_points import rot_points
from src.utils.crop_numpy_meter import crop_numpy_meter
from src.utils.get_rectangle import get_rectangle_multiple_positions
from src.utils.in_field_of_view import in_field_of_view
from src.utils.in_range import in_range


def _load_sample_multiple_channels(self, filename):
    mat_temp = loadmat(filename,
                       variable_names=["center", "bbox", "map_image", "data_info",
                                       "orient", "ego", "obj_list", "channel_order"],
                       verify_compressed_data_integrity=False)
    orientation = float(-mat_temp['orient'][0])
    
    # META===============================================================================================
    # GENERAL-------------------------------------------------------------------------------------------
    channel_order = []
    [channel_order.extend(el) for el in mat_temp['channel_order'][0]]
    # MAPS-----------------------------------------------------------------------------------------------
    map_info = mat_temp['map_image']

    # DYNAMICS============================================================================================
    # EGO--------------------------------------------------------------------------------------------------
    ego_full = mat_temp['ego']
    time_idx_temp = ego_full[7, :].astype(int) - 1
    ego_temp = ego_full[:, (time_idx_temp >= self.hist_seq_first) & (
        time_idx_temp <= self.hist_seq_last)]
    if self.orientation == 'ego':
        ego_temp[0:2, :] = rot_points(ego_temp[0:2, :], orientation)
    ego_out_hist = torch.from_numpy(ego_temp[0:3, :]).t()

    ego_temp = ego_full[:, (time_idx_temp >= self.pred_seq_first) & (
        time_idx_temp <= self.pred_seq_last)]
    if self.orientation == 'ego':
        ego_temp[0:2, :] = rot_points(ego_temp[0:2, :], orientation)
    ego_out_pred = torch.from_numpy(ego_temp[0:3, :]).t()
    
    ego_timestamp = torch.from_numpy(ego_full[2, :]).t()

    # OBJ--------------------------------------------------------------------------------------------------
    #TODO idea: create list of object-lists and save the object-lists per channel
    obj_temp = mat_temp['obj_list']
    obj_out_hist = []
    obj_out_pred = []
    obj_channel_src = []
    obj_timestamp = []
    obj_v = []
    obj_v_lat = []
    obj_v_lon = []
    obj_a_lat = []
    obj_a_lon = []
    if obj_temp.size != 0:
        for obj in obj_temp[0]:
            time_idx_temp = obj[7, :].astype(int) - 1
            
            steps_period = (time_idx_temp >= self.hist_seq_first) & (time_idx_temp <= self.hist_seq_last)
            # Object data
            src_channel_idx = obj[10, steps_period] - 1
            obj_channel_src.append(torch.from_numpy(src_channel_idx).t()) 
            obj_timestamp.append(torch.from_numpy(obj[2, steps_period]).t())
            obj_v.append(torch.from_numpy(obj[4, steps_period]).t())
            obj_a_lat.append(torch.from_numpy(obj[6, steps_period]).t())
            obj_a_lon.append(torch.from_numpy(obj[5, steps_period]).t())
            obj_v_lat.append(torch.from_numpy(obj[13, steps_period]).t())
            obj_v_lon.append(torch.from_numpy(obj[12, steps_period]).t())
            
            # History
            obj_inner = obj[:, steps_period]
            if obj_inner.size == 0:
                continue
            if "fieldofview" in self.augmentation_type:
                indicator = in_field_of_view(obj_inner, ego_full, self.augmentation_meta['range'], self.augmentation_meta['angle_range'])
                obj_inner = obj_inner[:, indicator]
            elif "range" in self.augmentation_type:
                indicator = in_range(obj_inner, ego_full, self.augmentation_meta['range'])
                obj_inner = obj_inner[:, indicator]
            if obj_inner.size == 0:
                continue
            if self.orientation == 'ego':
                obj_inner[0:2, :] = rot_points(obj_inner[0:2, :], orientation)
            obj_out_hist.append(torch.from_numpy(obj_inner).t())

            # Prediction
            obj_inner = obj[:, (time_idx_temp >= self.pred_seq_first) & (
                time_idx_temp <= self.pred_seq_last)]
            if "fieldofview" in self.augmentation_type:
                indicator = in_field_of_view(
                    obj_inner, ego_full, self.augmentation_meta['range'], self.augmentation_meta['angle_range'])
                obj_inner = obj_inner[:, indicator]
            elif "range" in self.augmentation_type:
                indicator = in_range(obj_inner, ego_full,
                                     self.augmentation_meta['range'])
                obj_inner = obj_inner[:, indicator]
            if obj_inner.size == 0:
                obj_out_pred.append(torch.from_numpy(obj_inner).t())
                continue
            if self.orientation == 'ego':
                obj_inner[0:2, :] = rot_points(obj_inner[0:2, :], orientation)
            obj_out_pred.append(torch.from_numpy(obj_inner).t())

    # TODO introduce may seq length
    obj_out_hist_seq_lens = [obj_temp.shape[0] for obj_temp in obj_out_hist]
    obj_out_hist = torch.nn.utils.rnn.pad_sequence(obj_out_hist, batch_first=True)

    obj_out_pred_seq_lens = [obj_temp.shape[0] for obj_temp in obj_out_pred]
    obj_out_pred = torch.nn.utils.rnn.pad_sequence(obj_out_pred, batch_first=True)

    return {'channel_order':channel_order,
            'hist_traj_objs': obj_out_hist, 'hist_traj_objs_lens': obj_out_hist_seq_lens,
            'pred_traj_objs': obj_out_pred, 'pred_traj_objs_lens': obj_out_pred_seq_lens,
            'hist_traj_ego': ego_out_hist,  'hist_traj_ego_lens': ego_out_hist.shape[0],
            'pred_traj_ego': ego_out_pred,  'pred_traj_ego_lens': ego_out_pred.shape[0],
            'obj_channel_src': obj_channel_src, 
            'obj_timestamp': obj_timestamp, 'ego_timestamp': ego_timestamp,
            'obj_v': obj_v, 'obj_v_lat': obj_v_lat, 'obj_v_lon': obj_v_lon,
            'obj_a_lat': obj_a_lat, 'obj_a_lon': obj_a_lon,
            'map_image': map_info}

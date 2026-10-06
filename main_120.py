from datetime import date, datetime
from experiment import Experiment
from torchvision import transforms

# Dynamic model
model_to_use = "constant_turn_rate"
feature_dim = 7
feature_dim_data = 5
onlyEgo = True
channels = 1
inference = True

# dataset number
# dataset hypers
dataset_train = [{
    'name': 'argoverse',
    'augmentation_type': None,
    'augmentation_meta': {
        'range': 25.0,
        'angle_range': 30.0
    },
    'hist_seq_first': 0,
    'hist_seq_last': 19,
    'pred_seq_last': 49,
    'orientation': 'ego',
    'representation_type': 'image_multichannel_vector',
    'mode': 'train',
    'bbox_meter': [60.0, 60.0],
    'bbox_pixel': [240, 240],
    'center_meter': [20.0, 30.0],
    'only_edges': False,
    'features_dim': feature_dim,
    'channels': channels
}]

dataset_val = [{
    'name': 'argoverse',
    'augmentation_type': None,
    'augmentation_meta': {
        'range': 25.0,
        'angle_range': 30.0
    },
    'hist_seq_first': 0,
    'hist_seq_last': 19,
    'pred_seq_last': 49,
    'orientation': 'ego',
    'representation_type': 'image_multichannel_vector',
    'mode': 'val',
    'bbox_meter': [60.0, 60.0],
    'bbox_pixel': [240, 240],
    'center_meter': [20.0, 30.0],
    'only_edges': False,
    'features_dim': feature_dim,
    'channels': channels
}]

# The test dataset ground truth is black box and only available via the evaluation server
dataset_test = [{
    'name': 'argoverse',
    'augmentation_type': None,
    'augmentation_meta': {
        'range': 25.0,
        'angle_range': 30.0
    },
    'hist_seq_first': 0,
    'hist_seq_last': 19,
    'pred_seq_last': 49,
    'orientation': 'ego',
    'representation_type': 'image_multichannel_vector',
    'mode': 'val',
    'bbox_meter': [60.0, 60.0],
    'bbox_pixel': [240, 240],
    'center_meter': [20.0, 30.0],
    'only_edges': False,
    'features_dim': feature_dim,
    'channels': channels
}]

# dataloader number
# dataloader hypers

train_dataloader = {
    'idx': 120,
    'batch_size': 16,
    'epochs': 25,
    'num_workers': 0,
    'shuffle': True,
    'representation': 'image_multichannel_vector',
    'features_dim': feature_dim_data,
    'channels': channels
}

val_dataloader = {
    'idx': 120,
    'batch_size': 1,
    'num_workers': 0,
    'shuffle': False,
    'representation': 'image_multichannel_vector',
    'features_dim': feature_dim_data,
    'channels': channels
}

test_dataloader = {
    'idx': 120,
    'batch_size': 1,
    'num_workers': 0,
    'shuffle': False,
    'representation': 'image_multichannel_vector',
    'features_dim': feature_dim_data,
    'channels': channels
}
# model number
# model hypers

#model = {'idx':0,'model_depth':18, 'projector_dim': [1024, 2048]}
model = {
    'idx': 730,
    'encoderI_type': 'ResNet-18',
    'encoderT_type': 'Transformer-Encoder',
    'merge_type': 'Transformer',
    'decoder_type': 'LSTM',
    'z_dim_t': 64,
    'z_dim_i': 64,
    'zm_dim_in': 64,
    'zm_dim_out': 512,
    'image_size': 240,
    'channels': channels,
    'traj_size': feature_dim_data,
    'm_depth': 5,
    'm_heads': 8,
    'use_infra': True,
    'use_infra_merge': True,
    'dynamic_model': model_to_use,
    'onlyEgo': onlyEgo,
    'inference': inference
}

# eval number
# eval hypers
evaluation = {
    'idx': 761,
    'visualize': False,
    'onlyEgo': onlyEgo,
    'dynamic_model': model_to_use
}

# optimis er numer
# optimiser hypers
optimiser = {'idx': 1, 'lr': 0.0001, 'weight_decay': 0, 'betas': (0.9, 0.999)}

# scheduler number
# scheduler hypers
scheduler = None

# loss number
# loass hypers
loss = [{
    'idx': 120
}, {
    'idx': 121,
    'thres_dynamic_1': 8,
    'thres_dynamic_2': 8
}]


name_experiment = "paper_pen1_22M_onlygoal"

meta_info = {
    "name": name_experiment,
    "description": "Driver behaviour in the traffic"
}

# train hypers
training = {
    'idx': 120,
    "num_gpus": 1,
    'dataset_dict': dataset_train[0],
    'summary_name': meta_info['name'],
    'dynamic_model': model_to_use,
    'frequency': 10
}

## Main file settings


if __name__ == '__main__':
    experiment = Experiment(meta_info, dataset_train, dataset_val,
                            dataset_test, train_dataloader, val_dataloader,
                            test_dataloader, model, training, evaluation,
                            optimiser, scheduler, loss)
    experiment.train()

from re import L
from src.model.model import model 
from cv2 import transform
#from model import model 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from src.model.model_123 import  model_123
#from model_101 import model_101 
import random
from einops import rearrange, reduce



 

class model_122(model):
    def __init__(self,
                micro_step = 3,
                encoderI_type=None, 
                encoderT_type=None, 
                merge_type=None, 
                decoder_type=None,
                encoderI_args=None,
                encoderT_args=None, 
                merge_args=None,
                z_dim_t=64,
                z_dim_i=64,
                zm_dim_in=64,
                zm_dim_out=128,
                image_size=200,
                channels=5,
                traj_size=3,
                use_infra=True,
                use_infra_merge = True,
                dynamic_model = True,
                m_depth=6,
                m_heads=8,
                idx = 120,
                name = 'ScenariioBetaAutoRegressive',
                size = None,
                n_params = None,
                input_ = 'image_multichannel_vector',
                output = 'vector for objections predicted',
                task = 'Representaion Learning',
                description = 'Learning behaviour from latent space'
                ):
        
        super().__init__(idx,name,size,n_params,input_,output,task,description)
        
        self.image_size = image_size
        self.zm_dim_in = zm_dim_in
        self.zm_dim_out = zm_dim_out
        self.dynamic_model = dynamic_model
        self.micro_steps = micro_step

        self.trajectory_forecaster = model_123(encoderI_type=encoderI_type,
            encoderT_type=encoderT_type,
            merge_type=merge_type,
            decoder_type=decoder_type,
            encoderI_args=encoderI_args,
            encoderT_args=encoderT_args,
            z_dim_t=z_dim_t,
            z_dim_i=z_dim_i,
            zm_dim_in=self.zm_dim_in,
            zm_dim_out=self.zm_dim_out,
            m_heads= m_heads,
            m_depth=m_depth,
            image_size=self.image_size,
            channels=channels,
            traj_size=traj_size,
            use_infra=use_infra,
            use_infra_merge=use_infra_merge,
            dynamic_model=dynamic_model
            )   

    @staticmethod
    def _process_dynamic_model(output,gTruthX,x_traj_pred_obj_len,pres_object_lengths_sum,target_len):
        X = output['X']
        Y = output['Y']

        X_reshaped = torch.empty((gTruthX.shape[0],gTruthX.shape[1],target_len,1),device=gTruthX.device)
        Y_reshaped = torch.empty((gTruthX.shape[0],gTruthX.shape[1],target_len,1),device=gTruthX.device)

        for unp in range(gTruthX.shape[0]):
            X_reshaped[unp,0:x_traj_pred_obj_len[unp],:,:] = X[pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1],:].unsqueeze(2)  
            Y_reshaped[unp,0:x_traj_pred_obj_len[unp],:,:] = Y[pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1],:].unsqueeze(2)
        return X_reshaped,Y_reshaped
    
    
    @staticmethod
    def _closed_loop(x_traj,output,gTruthX,x_traj_pred_obj_len,pres_object_lengths_sum,target_len):
        x_traj_hist_x = torch.empty_like(x_traj[0][:,:,:,0],device=x_traj[0].device)
        x_traj_hist_y = torch.empty_like(x_traj[0][:,:,:,1],device=x_traj[0].device)
        X_reshaped,Y_reshaped = model_122._process_dynamic_model(output,gTruthX,x_traj_pred_obj_len,pres_object_lengths_sum,target_len)


        x_traj_hist_x[:,:,:20-target_len] = x_traj[0][:,:,target_len:,0] 
        x_traj_hist_y[:,:,:20-target_len] = x_traj[0][:,:,target_len:,1] 
        x_traj_hist_x[:,:,20-target_len:] = X_reshaped.squeeze(3)
        x_traj_hist_y[:,:,20-target_len:] = Y_reshaped.squeeze(3)

        history_compute = torch.cat((x_traj_hist_x[:,:,:,None],x_traj_hist_y[:,:,:,None]),dim=3)
        x_traj = [history_compute,x_traj[1]]
        
        batch_wise_decoder_input = torch.cat((x_traj[0][:,:,-1,0].unsqueeze(2),x_traj[0][:,:,-1,1].unsqueeze(2)),dim=2)
        return x_traj,batch_wise_decoder_input
    
    
    def forward(self,
                x_image=None,
                x_traj=None,
                x_traj_len=None,
                batch_wise_object_lengths_sum=None,
                batch_wise_decoder_input=None,
                target_length=None,
                x_traj_pred_obj_len=None,
                pres_object_lengths_sum=None,
                gTruthX=None):
        target_length = target_length//self.micro_steps
        output = torch.empty
        for idx in range(self.micro_steps):
            output = self.trajectory_forecaster(x_image,
                                        x_traj,
                                        x_traj_len=x_traj_len,
                                        batch_wise_object_lengths_sum=batch_wise_object_lengths_sum,
                                        batch_wise_decoder_input=batch_wise_decoder_input,
                                        target_length=target_length,)
            x_reshaped_temp,y_reshaped_temp = model_122._process_dynamic_model(output,gTruthX,x_traj_pred_obj_len,pres_object_lengths_sum,target_length)
            x_traj,batch_wise_decoder_input = model_122._closed_loop(x_traj,output,gTruthX,x_traj_pred_obj_len,pres_object_lengths_sum,target_length)

            if idx == 0:
                v_x_reshaped = torch.empty((output['vx'].shape[0],target_length*self.micro_steps),device=x_reshaped_temp.device)
                v_y_reshaped = torch.empty((output['vy'].shape[0],target_length*self.micro_steps),device=x_reshaped_temp.device)

                x_reshaped = torch.empty((x_reshaped_temp.shape[0],x_reshaped_temp.shape[1],target_length*self.micro_steps,1),device=x_reshaped_temp.device)
                y_reshaped = torch.empty((y_reshaped_temp.shape[0],y_reshaped_temp.shape[1],target_length*self.micro_steps,1),device=y_reshaped_temp.device)
            v_x_reshaped[:,idx*target_length:(idx+1)*target_length] = output['vx']
            v_y_reshaped[:,idx*target_length:(idx+1)*target_length] = output['vy']
            x_reshaped[:,:,idx*target_length:(idx+1)*target_length,:] = x_reshaped_temp
            y_reshaped[:,:,idx*target_length:(idx+1)*target_length,:] = y_reshaped_temp
        output_dynamics = {}
        output_dynamics['vx'] = v_x_reshaped
        output_dynamics['vy'] = v_y_reshaped
     
        return x_reshaped,y_reshaped,output_dynamics
    


def generate_model(**model_params):
    return model_122(**model_params)


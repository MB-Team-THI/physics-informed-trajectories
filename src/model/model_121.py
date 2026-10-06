from re import L
from src.model.model import model 
from cv2 import transform
#from model import model 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from src.model.model_120 import  model_120
#from model_101 import model_101 
import random
from einops import rearrange, reduce



 

class model_121(model):
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
                onlyEgo = False,
                inference = False,
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
        self.onlyEgo = False#onlyEgo
        self.inference = inference
        self.trajectory_forecaster = model_120(encoderI_type=encoderI_type,
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
            dynamic_model=dynamic_model,
            inference=inference
            )   

    @staticmethod
    def _process_dynamic_model(output,gTruthX,x_traj_pred_obj_len,pres_object_lengths_sum,target_len,inference=False,features=3):
        X = output['X']
        Y = output['Y']
        T = output['T']
        if features == 5:
            V = output['v']
            PSI = output['psi']
            PSI_DOT = output['psi_dot']
            AX = output['ax']
        X_reshaped = torch.empty((gTruthX.shape[0],gTruthX.shape[1],target_len,1),device=gTruthX.device)
        Y_reshaped = torch.empty((gTruthX.shape[0],gTruthX.shape[1],target_len,1),device=gTruthX.device)
        T_reshaped = torch.empty((gTruthX.shape[0],gTruthX.shape[1],target_len,1),device=gTruthX.device)
        if features == 5:
            V_reshaped = torch.empty((gTruthX.shape[0],gTruthX.shape[1],target_len,1),device=gTruthX.device)
            PSI_reshaped = torch.empty((gTruthX.shape[0],gTruthX.shape[1],target_len,1),device=gTruthX.device)
            PSI_DOT_reshaped = torch.empty((gTruthX.shape[0],gTruthX.shape[1],target_len,1),device=gTruthX.device)
            AX_reshaped = torch.empty((gTruthX.shape[0],gTruthX.shape[1],target_len,1),device=gTruthX.device)

        for unp in range(gTruthX.shape[0]):
            X_reshaped[unp,0:x_traj_pred_obj_len[unp],:,:] = X[pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1],:].unsqueeze(2)  
            Y_reshaped[unp,0:x_traj_pred_obj_len[unp],:,:] = Y[pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1],:].unsqueeze(2)
            T_reshaped[unp,0:x_traj_pred_obj_len[unp],:,:] = T[pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1],:].unsqueeze(2)
            if features == 5:
                V_reshaped[unp,0:x_traj_pred_obj_len[unp],:,:] = V[pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1],:].unsqueeze(2)
                PSI_reshaped[unp,0:x_traj_pred_obj_len[unp],:,:] = PSI[pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1],:].unsqueeze(2)  
                PSI_DOT_reshaped[unp,0:x_traj_pred_obj_len[unp],:,:] = PSI_DOT[pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1],:].unsqueeze(2)          
                AX_reshaped[unp,0:x_traj_pred_obj_len[unp],:,:] = AX[pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1],:].unsqueeze(2)          

                        
        if features == 5:
            return X_reshaped,Y_reshaped,T_reshaped,V_reshaped,PSI_reshaped,PSI_DOT_reshaped,AX_reshaped
        else: 
            return X_reshaped,Y_reshaped,T_reshaped,None, None, None, None
    
    
    def _closed_loop(self,x_traj,output,gTruthX,x_traj_pred_obj_len,pres_object_lengths_sum,target_len,inference=False):
        x_traj_hist_x = torch.empty_like(x_traj[0][:,:,:,0],device=x_traj[0].device)
        x_traj_hist_y = torch.empty_like(x_traj[0][:,:,:,1],device=x_traj[0].device)
        x_traj_hist_t = torch.empty_like(x_traj[0][:,:,:,2],device=x_traj[0].device)
        if self.dynamic_model == 'constant_turn_rate':
            x_traj_hist_v = torch.empty_like(x_traj[0][:,:,:,3],device=x_traj[0].device)
            x_traj_hist_psi = torch.empty_like(x_traj[0][:,:,:,4],device=x_traj[0].device)
            features = 5
        else:
            features = 3


        X_reshaped,Y_reshaped,T_reshaped,V_reshaped,PSI_reshaped,PSI_DOT_reshaped,AX_reshaped = model_121._process_dynamic_model(output,gTruthX,
                                                                                                                                 x_traj_pred_obj_len,
                                                                                                                                 pres_object_lengths_sum,
                                                                                                                                 target_len,self.inference,
                                                                                                                                 features)
 
        x_traj_hist_x[:,:,:20-target_len] = x_traj[0][:,:,target_len:,0] 
        x_traj_hist_y[:,:,:20-target_len] = x_traj[0][:,:,target_len:,1] 
        x_traj_hist_x[:,:,20-target_len:] = X_reshaped.squeeze(3)
        x_traj_hist_y[:,:,20-target_len:] = Y_reshaped.squeeze(3)
        x_traj_hist_t[:,:,:20-target_len] = x_traj[0][:,:,target_len:,2]
        if self.dynamic_model == 'constant_turn_rate':
            x_traj_hist_v[:,:,:20-target_len] = x_traj[0][:,:,target_len:,3] 
            x_traj_hist_psi[:,:,:20-target_len] = x_traj[0][:,:,target_len:,4] 
            x_traj_hist_v[:,:,20-target_len:] = V_reshaped.squeeze(3)
            x_traj_hist_psi[:,:,20-target_len:] = PSI_reshaped.squeeze(3)


        if not inference:
            x_traj_hist_t[:,:,20-target_len:] = T_reshaped.squeeze(3)
        else:
            for t in range(target_len):
                #addition_time = torch.arange(x_traj[0][:,:,target_len,2],(x_traj[0][:,:,target_len:,2]+(0.1*target_len)),target_len+1)
                #addition_time = addition_time.to(x_traj[0].device)
                x_traj_hist_t[:,:,t+target_len] = x_traj_hist_t[:,:,t+target_len-1]+ 0.1*torch.ones_like(x_traj[0][:,:,target_len,2],device=x_traj[0].device)
            
            
        if self.dynamic_model == 'constant_turn_rate':
            history_compute = torch.cat((x_traj_hist_x[:,:,:,None],x_traj_hist_y[:,:,:,None],x_traj_hist_t[:,:,:,None],x_traj_hist_v[:,:,:,None],
                                         x_traj_hist_psi[:,:,:,None]),dim=3)

        else:
            history_compute = torch.cat((x_traj_hist_x[:,:,:,None],x_traj_hist_y[:,:,:,None],x_traj_hist_t[:,:,:,None]),dim=3)
        x_traj = [history_compute,x_traj[1]]
        if self.dynamic_model == 'constant_turn_rate':
            batch_wise_decoder_input = torch.cat((x_traj[0][:,:,-1,0].unsqueeze(2),x_traj[0][:,:,-1,1].unsqueeze(2),
                                                x_traj[0][:,:,-1,2].unsqueeze(2),x_traj[0][:,:,-1,3].unsqueeze(2),x_traj[0][:,:,-1,4].unsqueeze(2)),dim=2)
        else:
            batch_wise_decoder_input = torch.cat((x_traj[0][:,:,-1,0].unsqueeze(2),x_traj[0][:,:,-1,1].unsqueeze(2),
                                                x_traj[0][:,:,-1,2].unsqueeze(2)),dim=2)
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
                gTruthX=None,
                gTruthT=None):
        target_length = target_length//self.micro_steps
        output = torch.empty
        for idx in range(self.micro_steps):
            output = self.trajectory_forecaster(x_image,
                                        x_traj,
                                        x_traj_len=x_traj_len,
                                        batch_wise_object_lengths_sum=batch_wise_object_lengths_sum,
                                        batch_wise_decoder_input=batch_wise_decoder_input,
                                        target_length=target_length,)
            if self.dynamic_model == 'constant_turn_rate':
                features = 5
            else:
                features = 3

            x_reshaped_temp,y_reshaped_temp,t_reshaped_temp,v_reshaped_temp,psi_reshaped_temp,psi_dot_reshaped_temp,ax_reshaped_temp = model_121._process_dynamic_model(output,
                                                                                                                                 gTruthX,
                                                                                                                                 x_traj_pred_obj_len,
                                                                                                                                 pres_object_lengths_sum,
                                                                                                                                 target_length,
                                                                                                                                 self.inference,features)
            if self.micro_steps>1:
                x_traj,batch_wise_decoder_input = self._closed_loop(x_traj,output,gTruthX,x_traj_pred_obj_len,pres_object_lengths_sum,
                                                                    target_length,self.inference)


            if idx == 0:
                if self.dynamic_model == 'decoupled_dynamic':
                    v_x_reshaped = torch.empty((output['vx'].shape[0],target_length*self.micro_steps),device=x_reshaped_temp.device)
                    v_y_reshaped = torch.empty((output['vy'].shape[0],target_length*self.micro_steps),device=x_reshaped_temp.device)
                else:
                    v_reshaped = torch.empty((output['v'].shape[0],target_length*self.micro_steps),device=x_reshaped_temp.device)
                    psi_reshaped = torch.empty((output['psi'].shape[0],target_length*self.micro_steps),device=x_reshaped_temp.device)
                    ax_reshaped = torch.empty((output['ax'].shape[0],target_length*self.micro_steps),device=x_reshaped_temp.device)
                    psi_dot_reshaped = torch.empty((output['psi_dot'].shape[0],target_length*self.micro_steps),device=x_reshaped_temp.device)

                    
                if self.onlyEgo:
                    x_reshaped = torch.empty((x_reshaped_temp.shape[0],1,target_length*self.micro_steps,1),device=x_reshaped_temp.device)
                    y_reshaped = torch.empty((y_reshaped_temp.shape[0],1,target_length*self.micro_steps,1),device=y_reshaped_temp.device)
                    t_reshaped = torch.empty((t_reshaped_temp.shape[0],1,target_length*self.micro_steps,1),device=t_reshaped_temp.device)
     
                else:
                    x_reshaped = torch.empty((x_reshaped_temp.shape[0],x_reshaped_temp.shape[1],target_length*self.micro_steps,1),device=x_reshaped_temp.device)
                    y_reshaped = torch.empty((y_reshaped_temp.shape[0],y_reshaped_temp.shape[1],target_length*self.micro_steps,1),device=y_reshaped_temp.device)
                    t_reshaped = torch.empty((t_reshaped_temp.shape[0],t_reshaped_temp.shape[1],target_length*self.micro_steps,1),device=t_reshaped_temp.device)

            
            if self.dynamic_model == 'decoupled_dynamic':
                v_x_reshaped[:,idx*target_length:(idx+1)*target_length] = output['vx']
                v_y_reshaped[:,idx*target_length:(idx+1)*target_length] = output['vy']
            else:
                v_reshaped[:,idx*target_length:(idx+1)*target_length] = output['v']
                psi_reshaped[:,idx*target_length:(idx+1)*target_length] = output['psi']
                ax_reshaped[:,idx*target_length:(idx+1)*target_length] = output['ax']
                psi_dot_reshaped[:,idx*target_length:(idx+1)*target_length] = output['psi_dot']
  
            if self.onlyEgo:
                for unp in range(x_reshaped_temp.shape[0]):
                    x_reshaped[unp,0,idx*target_length:(idx+1)*target_length,:] = x_reshaped_temp[unp,x_traj_pred_obj_len[unp]-1,:,:]
                    y_reshaped[unp,0,idx*target_length:(idx+1)*target_length,:] = y_reshaped_temp[unp,x_traj_pred_obj_len[unp]-1,:,:]
                    t_reshaped[unp,0,idx*target_length:(idx+1)*target_length,:] = t_reshaped_temp[unp,x_traj_pred_obj_len[unp]-1,:,:]
            else:
                x_reshaped[:,:,idx*target_length:(idx+1)*target_length,:] = x_reshaped_temp
                y_reshaped[:,:,idx*target_length:(idx+1)*target_length,:] = y_reshaped_temp
                t_reshaped[:,:,idx*target_length:(idx+1)*target_length,:] = t_reshaped_temp

        output_dynamics = {}
        if self.dynamic_model == 'decoupled_dynamic':

            output_dynamics['vx'] = v_x_reshaped
            output_dynamics['vy'] = v_y_reshaped
        else:
            
            output_dynamics['v'] = v_reshaped
            output_dynamics['psi'] = psi_reshaped
            output_dynamics['psi_dot'] = psi_dot_reshaped
            output_dynamics['ax'] = ax_reshaped
     
        return x_reshaped,y_reshaped,t_reshaped,output_dynamics
    


def generate_model(**model_params):
    return model_121(**model_params)


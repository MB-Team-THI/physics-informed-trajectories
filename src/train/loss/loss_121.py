import torch
from src.train.loss.loss import loss

class loss_121(loss):
    def __init__(self,
                idx = 121,
                name = 'Delta loss',
                description = 'Used to control the rate of the outputs, eg. jerk, acceleration, steering rate, etc',
                input_ = 'dynamic outputs from the model and thresholds',
                output = 'loss',
                thres_dynamic_1=None,
                thres_dynamic_2=None
                ) -> None:
        super().__init__(idx,name,description,input_,output)
        self.thres_dynamic_1 = thres_dynamic_1
        self.thres_dynamic_2 = thres_dynamic_2
    def forward(self,output=None, dynamic_model = None):
        
        if dynamic_model == 'decoupled_dynamic':
            vx = output['vx']
            vy = output['vy']
            vx_dot = torch.diff(vx)/0.1 # 0.1 is the sample rate change this to a param later
            vy_dot = torch.diff(vy)/0.1 # 0.1 is the sample rate change this to a param later
            vx_dot = torch.abs(vx_dot)
            vy_dot = torch.abs(vy_dot)
            max_delta_1 = torch.max(vx_dot,axis=1)
            max_delta_2 = torch.max(vy_dot,axis=1)
            # Loss based on the maximum lateral and longitudinal acceleration allowed
            thre_boolean_dynamic_1 = vx_dot > self.thres_dynamic_1
            thre_boolean_dynamic_2 = vy_dot > self.thres_dynamic_2

            loss_vx_dot = torch.sum(vx_dot[thre_boolean_dynamic_1])
            loss_vy_dot = torch.sum(vy_dot[thre_boolean_dynamic_2])

            loss_dynamic = loss_vx_dot + loss_vy_dot

        elif dynamic_model == 'constant_turn_rate':
            # To do
            # Include loss for psi
            
            v = output['v']
            psi = output['psi']
            psi_dot = output['psi_dot']
            ax = output['ax']
            ay = v * psi_dot
            
            
            ax_dot = torch.diff(ax)/0.1 # 0.1 is the sample rate change this to a param later
            ay_dot = torch.diff(ay)/0.1 # 0.1 is the sample rate change this to a param later
            ax_dot = torch.abs(ax_dot)
            ay_dot = torch.abs(ay_dot)
            max_delta_ax_dot_1 = torch.max(ax_dot,axis=1)
            max_delta_ax_dot_2 = torch.max(ay_dot,axis=1)
            # Loss based on the maximum lateral and longitudinal acceleration allowed
            thre_boolean_dynamic_1 = ax_dot > 4
            thre_boolean_dynamic_2 = ay_dot > 4

            loss_ax_dot = torch.sum(ax_dot[thre_boolean_dynamic_1])
            loss_ay_dot = torch.sum(ay_dot[thre_boolean_dynamic_2])
            
            
            max_delta_1 = torch.max(ax,axis=1)
            max_delta_2 = torch.max(ay,axis=1)
            

            # Loss based on the maximum lateral and longitudinal acceleration allowed
            thre_boolean_dynamic_1 = ax > self.thres_dynamic_1
            thre_boolean_dynamic_2 = ay > self.thres_dynamic_2
            #thre_boolean_dynamic_3 = psi > 0.78
            loss_vx_dot = torch.sum(ax[thre_boolean_dynamic_1])
            loss_vy_dot = torch.sum(ay[thre_boolean_dynamic_2])
            #loss_psi = torch.sum(psi[thre_boolean_dynamic_3])


            loss_dynamic = loss_vx_dot + loss_vy_dot #+loss_ax_dot + loss_ay_dot# + loss_psi   +
        
        elif dynamic_model == 'deep_kinematic_model':
            
            theta = output['theta']
            a = output['a']

            max_delta_1 = torch.max(theta,axis=1)
            max_delta_2 = torch.max(a,axis=1)

            # Loss based on the maximum lateral and longitudinal acceleration allowed
            threshold_boolean_theta = theta > self.thres_dynamic_1
            thre_boolean_a = a > self.thres_dynamic_2

            loss_theta = torch.sum(theta[threshold_boolean_theta])
            loss_a = torch.sum(a[thre_boolean_a])


            loss_dynamic = loss_theta + loss_a 
            
        return loss_dynamic,max_delta_1[0],max_delta_2[0]


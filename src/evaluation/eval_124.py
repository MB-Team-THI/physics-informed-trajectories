import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.evaluation.eval import eval
from src.utils.average_meter import AverageMeter
from src.utils.get_fde import get_fde
from src.utils.get_ade import get_ade
import cv2
from src.utils.rot_points import rot_points
from einops import rearrange
from src.utils.get_cmap import get_cmap
import logging
import matplotlib.pyplot as plt
from src.utils.get_img_from_fig import get_img_from_fig
import os
import imageio
from tqdm import tqdm
# from argoverse.evaluation.competition_util import generate_forecasting_h5
# from argoverse.evaluation import eval_forecasting
class eval_124(eval):
    def __init__(self, 
                 idx=124,
                 name='MSE prediction accuracy',
                 input_='y_true,y_pred',
                 output='acc.',
                 visualize=False,
                 onlyEgo=False,
                 dynamic_model='decoupled_dynamic',
                 description='Calculates MSE,ADE,FDE for all vehicles in the batch',
                  ):
        super().__init__(idx,
                    name,
                    input_,
                    output,
                    description)
        self.visualize = visualize
        self.onlyEgo = onlyEgo
        self.dynamic_model = dynamic_model

    def __call__(self, model=None, dataloader_test=None, device=None,dataset_dict=None):
        return self._evaluate(model, dataloader_test, device,dataset_dict)

    def _evaluate(self, model, dataloader_test, device,dataset_dict):
        """
        Evaluates accuracy on linear model trained on upstream ssl model
        """
        def mse_eval(out_1 = None, out_2 = None):
            return F.mse_loss(out_1, out_2, reduction = 'mean')
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss_record = AverageMeter()
        ade_record = AverageMeter()
        fde_record = AverageMeter()
        save_results = False
        output_path = "/home/schenker3/Desktop/SCENARIO-NET/output_path/"
        model.eval()
        if next(model.parameters()).device is not device:
            model.to(device)
        output_all = {}
        g_all = {}
        p_all = {}
        batch_pass = 0
        max_id=0
        for batch_idx, sample in enumerate(tqdm(dataloader_test(epoch=0))):
            x_image = sample['images']
            x_image = x_image.to(device)
            x_traj = [sample['hist_objs'].to(device),sample['hist_obj_lens']]
            x_traj_len = sample['hist_objs_seq_len']
            x_traj_pred_obj_len = sample['pred_obj_lens']
            x_traj_pred_len = sample['pred_objs_seq_len']
            pres_object_lengths_sum = sample['pres_object_lengths_sum']
            obj_length_padded = sample['hist_object_lengths_sum']
            batch_wise_decoder_input = sample['obj_decoder_in'].to(device)
            gTruthX = sample['pred_objsx'].to(device)
            gTruthY = sample['pred_objsy'].to(device)
            gTruthT = sample['pred_objst'].to(device, non_blocking=True)

            target_len = 20
            output = model(x_image=x_image,x_traj=x_traj,x_traj_len=x_traj_len,batch_wise_object_lengths_sum=obj_length_padded,
                    batch_wise_decoder_input=batch_wise_decoder_input,target_length=target_len)
            X = output['X']
            Y = output['Y']
            T = output['T']

            X_reshaped = torch.empty((gTruthX.shape[0],1,target_len,gTruthX.shape[3]),device=gTruthX.device)
            Y_reshaped = torch.empty((gTruthX.shape[0],1,target_len,gTruthX.shape[3]),device=gTruthX.device)
            T_reshaped = torch.empty((gTruthX.shape[0],1,target_len,gTruthX.shape[3]),device=gTruthX.device)


            gTruthX_ego = torch.empty((gTruthX.shape[0],1,gTruthX.shape[2],gTruthX.shape[3]),device=gTruthX.device)
            gTruthY_ego = torch.empty((gTruthX.shape[0],1,gTruthX.shape[2],gTruthX.shape[3]),device=gTruthX.device)


            for unp in range(x_image.shape[0]):
                X_reshaped[unp,:,:,:] = X[pres_object_lengths_sum[unp+1]-1,:].unsqueeze(1)  
                Y_reshaped[unp,:,:,:] = Y[pres_object_lengths_sum[unp+1]-1,:].unsqueeze(1) 
                if not save_results:
                    gTruthX_ego[unp,:,:,:] = gTruthX[unp,x_traj_pred_obj_len[unp]-1,:,:]
                    gTruthY_ego[unp,:,:,:] = gTruthY[unp,x_traj_pred_obj_len[unp]-1,:,:]
                    T_reshaped[unp,:,:,:] = T[pres_object_lengths_sum[unp+1]-1,:].unsqueeze(1) 
            file_names = sample['file_names']
            center_collected = sample['center_collected']
            orientation = sample['orientation']
            predicted_traj = torch.cat((X_reshaped,Y_reshaped),dim=3)
            predicted_traj = predicted_traj.squeeze()
            gTruth_traj = torch.cat((gTruthX_ego,gTruthY_ego),dim=3)
            gTruth_traj = gTruth_traj.squeeze()
            # for b_idx, (pred_traj,gr_traj) in enumerate(zip(predicted_traj,gTruth_traj)):
            #     center = center_collected[b_idx]
                

            #     single = pred_traj.detach().cpu().numpy()
            #     single = rearrange(single,'tp f->f tp')
            #     single = rot_points(single, -orientation[b_idx])
            #     single = rearrange(single,'f tp->tp f')
            #     single[:,0] += center[0].numpy()
            #     single[:,1] += center[1].numpy() 
            #     single = rearrange(single,'p f->1 p f')
            #     if len(gr_traj)>0:
            #         gr = gr_traj.detach().cpu().numpy()
            #         gr = rearrange(gr,'tp f->f tp')
            #         gr = rot_points(gr, -orientation[b_idx])
            #         gr = rearrange(gr,'f tp->tp f')
            #         gr[:,0] += center[0].numpy()
            #         gr[:,1] += center[1].numpy()   
            #    # gr = rearrange(gr,'p f->1 p f')

                # seq_id = int(file_names[b_idx].split('/')[-1][:-4])
                # # if max_id<seq_id:
                # #     max_id=seq_id
                # # #print('\r'+str(seq_id),end="\n")
                # # multi_pred = []
                # # for _ in range(6):
                # #     multi_pred.append(single)
                # # multi_pred = np.array(multi_pred)
                # p_all[seq_id] = single
                # if len(gr_traj)>0:

                #     g_all[seq_id] = gr

            maskXnot = torch.isnan(gTruthX_ego)
            maskYnot = torch.isnan(gTruthY_ego)
            # if dataset_dict['name'] == 'lyft':
            #     gTruthT *=1e-9
            if save_results:

                file_names = sample['file_names']
                center_collected = sample['center_collected']
                orientation = sample['orientation']
                predicted_traj = torch.cat((X_reshaped,Y_reshaped),dim=3)
                predicted_traj = predicted_traj.squeeze()
                for b_idx, pred_traj in enumerate(predicted_traj):
                    center = center_collected[b_idx]
                   

                    single = pred_traj.detach().cpu().numpy()
                    single = rearrange(single,'tp f->f tp')
                    single = rot_points(single, -orientation[b_idx])
                    single = rearrange(single,'f tp->tp f')
                    single[:,0] += center[0].numpy()
                    single[:,1] += center[1].numpy()                    


                    seq_id = int(file_names[b_idx].split('/')[-1][:-4])
                    if max_id<seq_id:
                        max_id=seq_id
                    #print('\r'+str(seq_id),end="\n")
                    multi_pred = []
                    for _ in range(6):
                        multi_pred.append(single)
                    multi_pred = np.array(multi_pred)
                    output_all[seq_id] = multi_pred
                batch_pass+=1

            else:
                if gTruthX_ego.shape[2]>0:
                    X_mse = mse_eval(X_reshaped[~maskXnot],gTruthX_ego[~maskXnot])
                    Y_mse = mse_eval(Y_reshaped[~maskYnot],gTruthY_ego[~maskYnot])
                    loss = X_mse + Y_mse 

                    # ade = get_ade(X_reshaped, Y_reshaped, gTruthX_ego, gTruthY_ego, maskXnot, maskYnot)
                    # fde = get_fde(X_reshaped, Y_reshaped, gTruthX_ego, gTruthY_ego, x_traj_pred_len, maskXnot, maskYnot,self.onlyEgo)
                    # ade_record.update(ade.item(), x_image.size(0))
                    # fde_record.update(fde.item(), x_image.size(0))
                    loss_record.update(loss.item(), x_image.size(0))
                    logging.info("Validation loss: {}".format(loss_record.avg))
                    # logging.info("Validation ADE: {}".format(ade_record.avg))
                    # logging.info("Validation FDE: {}".format(fde_record.avg))

                else:
                    ade=10000
                    fde=10000
                    loss=10000

                with torch.no_grad():
                    if self.visualize:
                        save_path = os.path.join(os.getcwd(),'visImage')
                        save_path_gif = os.path.join(os.getcwd(),'visImage_gif')
                        if not os.path.exists(save_path_gif):
                            os.makedirs(save_path_gif)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        if type(dataset_dict) is list:
                            dataset_dict = dataset_dict[0]
                        resolution = np.array(dataset_dict['bbox_pixel'])/np.array(dataset_dict['bbox_meter'])
                        seq_len = dataset_dict['hist_seq_last']
                        center = dataset_dict['center_meter']
                        for idx,image in enumerate(x_image):
                            image = image[0,:,:].cpu().numpy()
                            traj_hist = x_traj[0][idx,:,:,:].cpu().numpy()
                            gTruthX_plot = gTruthX_ego[idx,:,:,:].cpu().numpy()
                            gTruthY_plot = gTruthY_ego[idx,:,:,:].cpu().numpy()
                            X_reshaped_plot = X_reshaped[idx,:,:,:].cpu().numpy()
                            Y_reshaped_plot = Y_reshaped[idx,:,:,:].cpu().numpy()
                            dynamic_1_plot = dynamic_1_reshaped[idx,:,:,:].cpu().numpy()
                            dynamic_2_plot = dynamic_2_reshaped[idx,:,:,:].cpu().numpy()
                            dynamic_3_plot = dynamic_3_reshaped[idx,:,:,:].cpu().numpy()
                            x_hist = np.expand_dims(traj_hist[x_traj_pred_obj_len[idx]-1,:,0],axis=0)
                            y_hist = np.expand_dims(traj_hist[x_traj_pred_obj_len[idx]-1,:,1],axis=0)

                            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

                            x_traj_len_local = x_traj_len[obj_length_padded[idx+1]]
                            x_traj_pred_len_local = x_traj_pred_len[obj_length_padded[idx+1]]
                            cmap_objec = get_cmap(X_reshaped_plot.shape[0])

                            fig_dy_x = plt.figure()
                            fig_dy_y = plt.figure()
                            dy_x_ax1 = fig_dy_x.add_subplot(111)
                            dy_y_ax1 = fig_dy_y.add_subplot(111)
                            dy_x_ax1.set_ylim([-4,4])
                            dy_y_ax1.set_ylim([-4,4])
                            #dy_x_ax1.legend()
                            #dy_y_ax1.legend()
                            dy_x_ax1.set_title('a_x')
                            dy_y_ax1.set_title('a_y')
                            for id,(x,y,x_temp_hist,y_temp_hist,gt_x,gt_y,dy_x,dy_y,dy_z) in \
                            enumerate(zip(X_reshaped_plot,
                                        Y_reshaped_plot,
                                        x_hist,
                                        y_hist,
                                        gTruthX_plot,
                                        gTruthY_plot,
                                        dynamic_1_plot,
                                        dynamic_2_plot,
                                        dynamic_3_plot)):

                            

                                x = x[:x_traj_pred_len_local,:]
                                y = y[:x_traj_pred_len_local,:] 

                                x_temp_hist = x_temp_hist[:x_traj_len_local,]
                                y_temp_hist = y_temp_hist[:x_traj_len_local,]
                                
                                gt_x = gt_x[:x_traj_pred_len_local,:]
                                gt_y = gt_y[:x_traj_pred_len_local,:]

                                dy_x = dy_x[:x_traj_pred_len_local,:]
                                dy_y = dy_y[:x_traj_pred_len_local,:]
                                if self.dynamic_model == 'constant_turn_rate':
                                    dy_z = dy_z[:x_traj_pred_len_local,:]

                                dy_x = dy_x[1:,:]
                                dy_y = dy_y[1:,:]
                                if self.dynamic_model == 'constant_turn_rate':
                                    dy_z = dy_z[1:,:]

                                
                                if dy_x.shape[0]!=1 and dy_y.shape[0]!=1:
                                    if self.dynamic_model == 'decoupled_dynamic':
                                        dy_x = np.diff(dy_x.squeeze())/0.1
                                        dy_y = np.diff(dy_y.squeeze())/0.1
                                        dy_x = np.expand_dims(dy_x,axis=1)
                                        dy_y = np.expand_dims(dy_y,axis=1)
                                        dy_x_ax1.plot(list(range(len(gt_x)-2)),dy_x,color=cmap_objec(id),linewidth=1)
                                        dy_y_ax1.plot(list(range(len(gt_y)-2)),dy_y,color=cmap_objec(id),linewidth=1)
                                    elif self.dynamic_model == 'constant_turn_rate':
                                        #ay = dy_x*dy_y
                                        ax = dy_z# dy_x*np.cos(dy_y)
                                        ay = dy_x*np.sin(dy_y)
                                        dy_x_ax1.plot(list(range(len(gt_x)-1)),ax.squeeze(),color=cmap_objec(id),linewidth=1)
                                        dy_y_ax1.plot(list(range(len(gt_y)-1)),ay,color=cmap_objec(id),linewidth=1)
                                        

                                line_cor = [(int(x_temp),int(y_temp)) for x_temp,y_temp in zip((gt_x+center[0])*resolution[0],dataset_dict['bbox_pixel'][1]-(gt_y+center[1])*resolution[1])]
                                line_cor = np.array(line_cor)
                                line_cor = line_cor.reshape(-1,1,2)
                                image = cv2.polylines(image, [line_cor], False, (0,1,0), 1)                   

                                line_cor = [(int(x_temp),int(y_temp)) for x_temp,y_temp in zip((x_temp_hist+center[0])*resolution[0],dataset_dict['bbox_pixel'][1]-(y_temp_hist+center[1])*resolution[1])]
                                line_cor = np.array(line_cor)
                                line_cor = line_cor.reshape(-1,1,2)

                                image = cv2.polylines(image, [line_cor], False, cmap_objec(id), 3)   


                            
                                line_cor = [(int(x_temp),int(y_temp)) for x_temp,y_temp in zip((x+center[0])*resolution[0],dataset_dict['bbox_pixel'][1]-(y+center[1])*resolution[1])]
                                line_cor = np.array(line_cor)
                                line_cor = line_cor.reshape(-1,1,2)

                                image = cv2.polylines(image, [line_cor], False, (0,0,1), 1)

                            #plt.show()
                            #image_to_display = rearrange(image,'H W C->C H W')
                            dy_x_image = get_img_from_fig(fig_dy_x)
                            dy_x_image = cv2.resize(dy_x_image, (image.shape[0],image.shape[1]), interpolation = cv2.INTER_AREA)  
                            dy_y_image = get_img_from_fig(fig_dy_y)
                            dy_y_image = cv2.resize(dy_y_image, (image.shape[0],image.shape[1]), interpolation = cv2.INTER_AREA)
                            dy_x_image = dy_x_image
                            dy_y_image = dy_y_image
                            # clear the figure
                            plt.close(fig_dy_x)
                            dy_x_ax1.cla()
                            plt.close(fig_dy_y)
                            dy_y_ax1.cla()

                            #image_to_display = image
                            image_to_display = np.concatenate((image*255.0,dy_x_image,dy_y_image),axis=1)
                            cv2.imwrite(os.path.join(save_path,str(batch_idx)+'_'+str(idx)+'.png'),image_to_display)
                            del image_to_display
                            if idx>5:
                                break

                        # GIF writer
                        for idx,image in enumerate(x_image):
                                
                            image = image[0,:,:].cpu().numpy()
                            traj_hist = x_traj[0][idx,:,:,:].cpu().numpy()
                            gTruthX_plot = gTruthX_ego[idx,:,:,:].cpu().numpy()
                            gTruthY_plot = gTruthY_ego[idx,:,:,:].cpu().numpy()
                            X_reshaped_plot = X_reshaped[idx,:,:,:].cpu().numpy()
                            Y_reshaped_plot = Y_reshaped[idx,:,:,:].cpu().numpy()

                            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    
                            x_pred_temp = X_reshaped_plot
                            y_pred_temp = Y_reshaped_plot
                            x_hist_temp = np.expand_dims(traj_hist[x_traj_pred_obj_len[idx]-1,:,0],axis=0)
                            y_hist_temp = np.expand_dims(traj_hist[x_traj_pred_obj_len[idx]-1,:,1],axis=0)
                            gtruth_x_temp = gTruthX_plot
                            gtruth_y_temp = gTruthY_plot
                            frames = []
                            ímage_old = image.copy()
                            image_base = image.copy()
                            for id in range(x_hist_temp.shape[1]+x_pred_temp.shape[1]):
                                if id < x_hist_temp.shape[1]:
                                    ímage_old = image.copy()

                                    x_temp = (x_hist_temp[:,id] + center[0])*resolution[0]
                                    y_temp = dataset_dict['bbox_pixel'][1]-(y_hist_temp[:,id]+center[1])*resolution[1]
                                    for cid,(circle_x_temp,circle_y_temp) in enumerate(zip(x_temp.astype(int),y_temp.astype(int))):
                                        #alpha = 0.75
                                        #beta = 1 - alpha
                                        #image_draw = cv2.addWeighted(image_base,alpha,ímage_old,beta,0)
                                        cv2.circle(ímage_old,(circle_x_temp,circle_y_temp),3,cmap_objec(cid),-1)
                                    frames.append(ímage_old)
                                else:
                                    image_base_draw = image.copy()
                                    cv2.putText(image_base_draw, 'GT', (220, 220), cv2.FONT_HERSHEY_PLAIN, 1, (1,0,0), 2)
                                    cv2.putText(image_base_draw, 'Pred', (220, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0,1,0), 2)

                                    x_temp = (x_pred_temp[:,id-x_hist_temp.shape[1],:]+ center[0])*resolution[0]
                                    y_temp = dataset_dict['bbox_pixel'][1]-(y_pred_temp[:,id-x_hist_temp.shape[1],:]+center[1])*resolution[1]
                                    for circle_x_temp,circle_y_temp in zip(x_temp.astype(int),y_temp.astype(int)):
                                        if circle_x_temp[0]>0 and circle_y_temp[0]>0:
                                            cv2.circle(image_base_draw,(circle_x_temp[0],circle_y_temp[0]),3,(0,1,0),-1)
                                    x_gTruth_temp = (gtruth_x_temp[:,id-x_hist_temp.shape[1],:]+ center[0])*resolution[0]
                                    y_gTruth_temp = dataset_dict['bbox_pixel'][1]-(gtruth_y_temp[:,id-x_hist_temp.shape[1],:]+center[1])*resolution[1]
                                    for circle_x_temp,circle_y_temp in zip(x_gTruth_temp.astype(int),y_gTruth_temp.astype(int)):
                                        if circle_x_temp[0]>0 and circle_y_temp[0]>0:
                                            cv2.circle(image_base_draw,(circle_x_temp[0],circle_y_temp[0]),3,(1,0,0),-1)
                                    frames.append(image_base_draw)
                            save_gif_name = os.path.join(save_path_gif,str(batch_idx)+'_'+str(idx)+'.gif')
                            with imageio.get_writer(save_gif_name, mode="I") as writer:
                                for frame in frames:
                                    writer.append_data(frame)
                            writer.close()
        if not save_results:   
            # generate_forecasting_h5(p_all,output_path)
            print("Validation loss: {}, Validation ADE: {}, Validation FDE: {}".format(loss_record.avg,0,0))

            # eval_forecasting.get_displacement_errors_and_miss_rate(p_all,g_all,1,30,2.0)

            return loss_record.avg,0,0
        else:
            print(max_id)
            generate_forecasting_h5(output_all,output_path)


import torch
import wandb
import numpy as np
from tqdm import tqdm
from math import sin, pi
from torch import nn
from pprint import pprint
import pandas
from torchvision import transforms
from MedAI_code_segmentation_evaluation import IOU_class01, calculate_metrics_torch
# from My_losses import *
#TODO: delete the unecessarly model.train after the phase loop
# TODO: Implement saving a checkpoint
def saving_checkpoint(epoch,model,optimizer,val_loss,test_loss,
                      val_mIOU,test_mIOU, colab_dir, model_name, save_generator_checkpoints = False):

    checkpoint = {
        'epoch': epoch + 1,
        'description': "add your description",
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'Validation Loss': val_loss,
        'Test Loss': test_loss,
        'IOU Polyp test': test_mIOU,
        'IOU Polyp val': val_mIOU
    }
    if save_generator_checkpoints:
        torch.save(checkpoint,
                   colab_dir + '/checkpoints/gen_highest_loss_' + model_name + '.pt')
    else:
        torch.save(checkpoint,
               colab_dir + '/checkpoints/highest_IOU_' + model_name + '.pt')

    print("finished saving checkpoint")

def show2(generated_images, X, generated_mask,true_mask, phase, index, save,save_all=(False,-1),limit=5):
    if phase[-1].isnumeric():
        if int(phase[-1]) != 1:
            return

    original_imgs = X
    if not generated_mask.shape == original_imgs.shape:
        generated_mask = generated_mask.repeat(1, 3, 1, 1)
    if not true_mask.shape == original_imgs.shape:
        true_mask = true_mask.repeat(1, 3, 1, 1)
    if not generated_images.shape == original_imgs.shape:
        generated_images.unsqueeze_(1)  # (N,H,W) ==> (N,1,H,W)
        generated_images = generated_images.repeat(1, 3, 1, 1)  # (N,1,H,W) ==> (N,3,H,W)

    # inference mode configuration
    inference_mode = False
    total_images_so_far = save_all[1]
    if save_all[0]:  # True if we are in inference phase
        inference_mode = True
        limit=-1 #no limit
    # End inference mode configuration

    toPIL = transforms.ToPILImage()
    for i, img in enumerate(generated_images):
        if (i == limit): return
        generated_img = img.clone().detach().cpu()
        original_img = original_imgs[i].clone().detach().cpu()
        mask_img = generated_mask[i].clone().detach().cpu()
        true_mask_img = true_mask[i].clone().detach().cpu()
        imgs_cat = torch.cat((original_img, generated_img), 2)
        masks_cat = torch.cat((true_mask_img, mask_img), 2)
        img = torch.cat((imgs_cat, masks_cat), 1)
        img = toPIL(img)  # .numpy().transpose((1, 2, 0))
        if inference_mode:
            image_numbering = str(index+i+ (total_images_so_far-len(generated_images)))
        else:
            image_numbering = str(index) + '_' + str(i)
        img.save('./generatedImages_' + phase + '/' + image_numbering + 'generated.png')

def show(generated_imgs, original_imgs,masks, phase, index, save):
    # if not isinstance(torch_img,list):
    #     torch_img = [torch_img]
    if not masks.shape == original_imgs.shape:
        masks = masks.repeat(1,3,1,1)
    if not generated_imgs.shape == original_imgs.shape:
        generated_imgs.unsqueeze_(1) # (N,H,W) ==> (N,1,H,W)
        generated_imgs=generated_imgs.repeat(1,3,1,1) # (N,1,H,W) ==> (N,3,H,W)

    toPIL = transforms.ToPILImage()
    for i, img in enumerate(generated_imgs):
        if (i == 5): return
        generated_img = img.clone().detach().cpu()
        original_img = original_imgs[i].clone().detach().cpu()
        mask_img = masks[i].clone().detach().cpu()
        img = torch.cat((original_img, generated_img,mask_img), 2)
        img = toPIL(img)  # .numpy().transpose((1, 2, 0))
        img.save('./generatedImages_'+phase+'/' + str(index) + '_' + str(i) + 'generated.png')
        # plt.imshow(img)
        # if save:
        #     plt.savefig('./generatedImages/'+str(index)+'_'+str(i)+'generated.jpg')
        #     plt.clf()
        # else:
        #     plt.show()
        #     plt.clf()
        # print(img)


def ExpandingManifold_training_loop(num_epochs, optimizer, lamda, model, loss_dic, data_loader_dic,
                         device,switch_epoch,colab_dir,
                         model_name,inference=False):
    best_loss = {k: 1000 for k in data_loader_dic.keys()}
    best_iou = {k: 0 for k in data_loader_dic.keys()}
    best_iou_epoch = -1
    loss_fn_sum = loss_dic['generator']
    generator_models = ['ExpandMani_unetsqueezed', 'ExpandMani_unetwithoutskip', 'ExpandMani_VAE']
    #this variable to track the performance of the generator
    # this number is created according to the best gen loss at
    # Denoising_trainCVC_testKvasir_Exp4_IncludeAugX_hue_avgV2_unet_Lraspp
    # best_train_generator_loss=0.005
    best_train_generator_loss=1000
    for epoch in range(0, num_epochs + 1):

        for phase in data_loader_dic.keys():
            if phase.find('test')>=0 and best_iou_epoch != epoch: #skip testing if no better iou val achieved
                continue
            if phase == 'train':
                model.train()
            else:
                model.eval()

            flag = True #flag for showing first batch
            total_train_images = 0
            # TODO: the Loss here are normalized using np.mean() to get the average loss across all images. However,
            # TODO: last epoch may have less images hence, mean is not accurate
            loss_batches = []
            loss_l2_batches = []
            loss_mask_batches = []
            loss_KL_batches = []
            iou_batches = np.array([])
            iou_background_batches = np.array([])
            # metrics_polyp = []
            # metrics_background = []

            all_true_maskes_torch = []
            all_pred_maskes_torch = []

            pbar = tqdm(data_loader_dic[phase], total=len(data_loader_dic[phase]))
            for X, intermediate, original_masks in pbar:
                z_vectors = None
                batch_size = len(X)
                total_train_images += batch_size

                X = X.to(device).float()
                intermediate = intermediate.to(device).float()  # intermediate is the mask with type of float
                original_masks = original_masks.to(device)#this is 2 channels mask

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                #Generate polyp images and masks
                    if model_name.find('GenSeg') >= 0:
                        results = model(X, phase, original_masks)
                        generated_images, generated_masks, original_masks = results
                    elif model_name.find('ExpandMani') >= 0:
                        cycles = 10
                        rate = (1+sin(cycles*epoch/num_epochs))/2
                        results = model(X, phase, original_masks, rate = rate)
                        if model_name.find('VAE')>=0:
                            #if it VAE we need to handle the z_mean, z_log_var to calculate KL
                            generated_images, generated_masks, original_masks,(z_mean, z_log_var) = results
                        else:
                            generated_images, generated_masks, original_masks = results
                    else:  # the old version code i.e., other than GenSeg_IncludeX models
                        generated_images = model[0](X)
                        generated_X = generated_images.clone().detach()
                        if epoch >= switch_epoch[1]:
                            generated_masks = model[1](generated_X)

                    loss_mask = torch.zeros((1)).int()
                    iou=0



                    #reconstruction loss |f-g| "MSELoss
                    loss_l2 = loss_fn_sum(generated_images, X) * lamda['l2']
                    # mask loss
                    bce = loss_dic['segmentor']
                    loss_mask = bce(generated_masks, original_masks)

                    # KL Divergence loss only for VAE
                    if model_name.find('VAE')>=0:
                        kl_div = -0.5 * torch.sum(1+ z_log_var - z_mean**2 - torch.exp(z_log_var), dim=1)
                        kl_div = kl_div.mean()/1000
                        loss = loss_l2 + loss_mask + kl_div
                    else:
                        loss = loss_l2 + loss_mask
                        kl_div = torch.zeros((1))

                    # iou = IOU_class01(original_masks, generated_masks)
                    # iou is numpy array for each image
                    iou = calculate_metrics_torch(true=original_masks,pred=generated_masks,metrics='jaccard')
                    iou_background = calculate_metrics_torch(true=original_masks,pred=generated_masks,
                                                             metrics='jaccard',ROI='background')
                    #store all the true and pred masks
                    if len(all_true_maskes_torch)==0:
                        all_true_maskes_torch = original_masks.clone().detach()
                        all_pred_maskes_torch = generated_masks.clone().detach()
                    else:
                        all_true_maskes_torch = torch.cat((all_true_maskes_torch,original_masks.clone().detach()))
                        all_pred_maskes_torch = torch.cat((all_pred_maskes_torch,generated_masks.clone().detach()))

                    loss_batches.append(loss.clone().detach().cpu().numpy())
                    loss_l2_batches.append(loss_l2.clone().detach().cpu().numpy())
                    loss_KL_batches.append(kl_div.clone().detach().cpu().numpy())
                    loss_mask_batches.append(loss_mask.clone().detach().cpu().numpy())
                    iou_batches=np.append(iou_batches,iou)
                    iou_background_batches=np.append(iou_background_batches,iou_background)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if flag:  # this flag
                    flag = False
                    _, true_mask = original_masks.max(dim=1, keepdim=True)
                    if epoch >= switch_epoch[1]:#stage 3
                        max, generated_mask = generated_masks.max(dim=1)
                        generated_mask = generated_mask.unsqueeze(dim=1)
                        show2(generated_images, X, generated_mask,true_mask, phase, index=100 + epoch, save=True)
                    else: #stage 1 and 2
                        generated_mask = torch.zeros(generated_images.shape)
                        show2(generated_images, X, generated_mask,true_mask, phase, index=100 + epoch, save=True)

                # update the progress bar
                pbar.set_postfix({phase + ' Epoch': str(epoch) + "/" + str(num_epochs - 1),
                                  'polypIOU': iou_batches.mean(),
                                  'best_val_iou': best_iou['val'],
                                  'Loss': np.mean(loss_batches),
                                  'L2': np.mean(loss_l2_batches),
                                  'BCE_loss': np.mean(loss_mask_batches),
                                  'KL_loss': np.mean(loss_KL_batches),
                                  'mIOU': np.mean((iou_batches + iou_background_batches) / 2)
                                  })

            # !!!! calculate metrics for all images. The results are dictionary
            mean_metrics_polyp = calculate_metrics_torch(all_true_maskes_torch, all_pred_maskes_torch,
                                                         reduction='mean',cloned_detached=True)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if phase=='train' and model_name in generator_models:
                # if the generator is getting better save a checkpoint for the generator
                generator_loss = np.mean(loss_l2_batches)

                if best_train_generator_loss > generator_loss:
                    print('saving a checkpoint for the best generator '
                          '\n previously={} and now={}'.format(best_train_generator_loss, generator_loss))
                    saving_checkpoint(epoch, model, optimizer,
                                      generator_loss, -1,
                                      generator_loss, -1,
                                      colab_dir, model_name)
                    best_train_generator_loss = generator_loss

            if phase != 'train':
                if best_iou_epoch == epoch:  # calculate background metrics for val and test if it is the best epoch
                    metrics_dic_background = calculate_metrics_torch(all_true_maskes_torch, all_pred_maskes_torch,
                                                                     reduction='mean', cloned_detached=True,
                                                                     ROI='background')
                    metrics_dic_polyp = mean_metrics_polyp

                    metrics_mMetrics_dic = {metric: (metrics_dic_polyp[metric] + metrics_dic_background[metric]) / 2
                                            for metric in metrics_dic_background.keys()}

                    print(phase, ':', metrics_dic_polyp)
                    wandb.run.summary["dict_{}".format(phase)] = metrics_dic_polyp

                    if inference:
                        file_name = colab_dir + "/results/bestGenerator_{}_summary_report.xlsx".format(phase)
                    else:
                        file_name = colab_dir + "/results/{}_summary_report.xlsx".format(phase)
                    pandas.DataFrame.from_dict({'Polyp': metrics_dic_polyp, 'Background': metrics_dic_background,
                                                'Mean': metrics_mMetrics_dic}).transpose().to_excel(file_name)

                if phase=='val':
                    #if Polyp mean is getting better
                    if mean_metrics_polyp['jaccard'] > best_iou[phase]:
                        wandb.run.summary["best_{}_iou".format(phase)] = np.mean(iou_batches)
                        wandb.run.summary["best_{}_iou_epoch".format(phase)] = epoch
                        best_iou[phase] = mean_metrics_polyp['jaccard'] #Jaccard/IOU of polyp
                        best_loss[phase] = np.mean(loss_batches)
                        best_iou_epoch = epoch
                        print('best val_iou')
                        if model_name not in generator_models:
                            print('Saving a checkpoint')
                            saving_checkpoint(epoch, model, optimizer,
                                              best_loss['val'], -1, #test_loss
                                              best_iou['val'],  -1, #test_mIOU
                                              colab_dir, model_name)
                            print('testing on a test set....\n')


            wandb.log({phase + "_loss": np.mean(loss_batches),
                       phase + "_L2": np.mean(loss_l2_batches),
                       phase + '_BCE_loss': np.mean(loss_mask_batches), phase + '_iou': np.mean(iou_batches),
                       phase + "_KL": np.mean(loss_KL_batches),
                       "best_val_loss": best_loss['val'],
                       'best_val_iou': best_iou['val'], phase + "_epoch": epoch},
                      step=epoch)


def show_filter(generated_masks, original_masks,kernel3D, phase, index, save):
    # if not isinstance(torch_img,list):
    #     torch_img = [torch_img]
    kernel3D = kernel3D.squeeze() # (1,3,H,W) ==> (3,H,W)
    if not original_masks.shape == kernel3D.shape:
        original_masks = original_masks.repeat(1,3,1,1)
    if not generated_masks.shape == kernel3D.shape:
        generated_masks.unsqueeze_(dim=1)
        generated_masks = generated_masks.repeat(1,3,1,1)
    kernel_img = kernel3D.clone().detach()
    # normalize kernel_img
    kernel_img = (kernel_img - torch.min(kernel_img))/(torch.max(kernel_img) - torch.min(kernel_img))
    kernel_img =kernel_img.cpu()
    toPIL = transforms.ToPILImage()
    for i, img in enumerate(generated_masks):
        if (i == 5): return
        generated_mask = img.clone().detach().cpu()
        original_mask = original_masks[i].clone().detach().cpu()

        img = torch.cat((original_mask, generated_mask,kernel_img), 2)
        img = toPIL(img)  # .numpy().transpose((1, 2, 0))
        img.save('./generatedImages_'+phase+'/' + str(index) + '_' + str(i) + 'generated.png')
        # plt.imshow(img)
        # if save:
        #     plt.savefig('./generatedImages/'+str(index)+'_'+str(i)+'generated.jpg')
        #     plt.clf()
        # else:
        #     plt.show()
        #     plt.clf()
        # print(img)

def cat_split(tensor_s):
    split=True
    if isinstance(tensor_s,list):#if list, means we need to concat
        return torch.cat(tensor_s,dim=0)
    else: # or split
        return tensor_s.chunk(chunks=2)


#if phase != 'train':
# Saving the best loss for the validation
# if np.mean(loss_batches) < best_loss[phase]:
#     print('best {} loss={} so far ...'.format(phase,np.mean(loss_batches)))
#     wandb.run.summary["best_{}_loss_epoch".format(phase)] = epoch
#     wandb.run.summary["best_{}_loss".format(phase)] = np.mean(loss_batches)
#     best_loss[phase] = np.mean(loss_batches)
#     if phase=='val':
#         print('better validation loss- saving checkpoint')
#         saving_checkpoint(epoch, model, optimizer,
#                           best_loss[phase], 0,
#                           0, 0,
#                           colab_dir, model_name)
# calculate summary results and store them in results
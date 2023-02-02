#9064466c4a4f16db52c1672e03ee3c52060a24e4 token
import torch.optim as optim
from requests import get
# import matplotlib.pyplot as plt
import wandb
import random
import time
import os
import numpy as np
# from Plotting import plot, plot_test
from torch.nn import functional as F
from MyDataloaders import *
# from Metrics import *
from models import MyModelV1, FCNModels, DeepLabModels, unet
import torch
from MyDataloaders_denoising import getLoadersBySetName
from models.GenSeg_Models import *
from models.unet_withoutskip import UNet as unet_withoutskip
from torch import nn
from Training import *
from torchvision import datasets
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader



wandb.login(key="38818beaffe50c5403d3f87b288d36d1b38372f8")
# from prettytable import PrettyTable
def initializWandb():
    transfer_learning = model_name.find('TL') >= 0
    wandbproject_name = "ExpandingManifold"
    wandb.init(
        project=wandbproject_name,
        entity="mss3331",
        name=experiment_name,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "architecture": model_name,
            "batch_size": batch_size,
            "lamda": lamda,
            "num_epochs": num_epochs,
            "shuffle": shuffle,
            "transfer_learning": transfer_learning,
            "dataset": experimentDatasets, })

def repreducibility():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getStateDict(checkpoint):
    for key,value in checkpoint.items():
        if key.find('state')>=0:#Skip state_dict, thought print other keys
            continue
        print(key,":",value)
    return checkpoint['state_dict']

def getModel(model_name):
    # identify which models for Gen Seg
    Gen_Seg_arch = model_name.split('_')[-2:]
    pretrained = model_name.find('TL') >= 0

    if model_name.find('unet-proposed')>=0:
        model = unet_proposed()
    elif model_name.find('Conventional') >= 0:
        if model_name.find('avgV2_blure') >= 0:
            model = GenSeg_IncludeX_Conventional_avgV2_blure(Gen_Seg_arch)
        elif model_name.find('avgV2_colorjitter') >= 0:
            model = GenSeg_IncludeX_Conventional_avgV2_colorjitter(Gen_Seg_arch)
        elif model_name.find('avgV2_hue') >= 0:
            model = GenSeg_IncludeX_Conventional_avgV2_hue(Gen_Seg_arch)
        elif model_name.find('avgV2_brightness') >= 0:
            model = GenSeg_IncludeX_Conventional_avgV2_brightness(Gen_Seg_arch)
        elif model_name.find('Conventional_colorjitter') >= 0:
            model = GenSeg_IncludeX_Conventional_colorjitter(Gen_Seg_arch)
        elif model_name.find('Conventional_blure') >= 0:
            model = GenSeg_IncludeX_Conventional_blure(Gen_Seg_arch)
        elif model_name.find('Conventional_hue') >= 0:
            model = GenSeg_IncludeX_Conventional_hue(Gen_Seg_arch)
        elif model_name.find('Conventional_brightness') >= 0:
            model = GenSeg_IncludeX_Conventional_brightness(Gen_Seg_arch)
    elif model_name.find('IncludeX')>=0:
        if model_name.find('_max')>=0:
            model = GenSeg_IncludeX_max(Gen_Seg_arch)
        elif model_name.find('_conv_')>=0:
            model = GenSeg_IncludeX_conv(Gen_Seg_arch)
        elif model_name.find('_convV2')>=0:
            model = GenSeg_IncludeX_convV2(Gen_Seg_arch)
        elif model_name.find('IncludeX_avg_')>=0:
            model = GenSeg_IncludeX_avg(Gen_Seg_arch)
        elif model_name.find('IncludeX_avgV2')>=0:
            model = GenSeg_IncludeX_avgV2(Gen_Seg_arch)
        elif model_name.find('_NoCombining')>=0:
            model = GenSeg_IncludeX_NoCombining(Gen_Seg_arch)
        elif model_name.find('ColorJitterGenerator_avgV2')>=0:
            model = GenSeg_IncludeX_ColorJitterGenerator_avgV2(Gen_Seg_arch)
        elif model_name.find('ColorJitterGeneratorTrainOnly_avgV2')>=0:
            model = GenSeg_IncludeX_ColorJitterGeneratorTrainOnly_avgV2(Gen_Seg_arch)
    elif model_name.find('IncludeAugX')>=0:
            if model_name.find('hue_avgV2')>=0:
                model = GenSeg_IncludeAugX_hue_avgV2(Gen_Seg_arch,transfer_learning=pretrained)
            elif model_name.find('gray_avgV2')>=0:
                model = GenSeg_IncludeAugX_gray_avgV2(Gen_Seg_arch)
    elif model_name.find('Vanilla') >= 0:
        model = GenSeg_Vanilla(Gen_Seg_arch,pretrained)
    elif model_name.find('ExpandMani')>=0:
        if model_name.find('unet_withoutskip')>=0:
            #out channels is 5 (2 for mask and 3 for generated images)
            model = unet_withoutskip(in_channels=3, out_channels=5,n_blocks=5,activation='relu',normalization='batch',conv_mode='same',dim=2)

    else:
        print('Model name unidentified')
        exit(-1)

    return model

def get_Dataloaders_dic(experimentDatasets):
    '''for now let us make it simple and fixed. Later we would use Yaml config to design dataset.
    for now the experiments as follows:
    1- Train/val/test: EndoSceneStill
    2- Train/val: CVC-clinicDB, test:Kvasir
    3- Train/val: Kvasir, test: CVC-clinicDB
    '''
    Dataloaders_dic = {}
    if experimentDatasets==None:
        print('experimentDatasets is not provided')
        exit(-1)

    if experimentDatasets=='CVC_EndoSceneStill':
        # EndoSceneStill train (C1), val (C2), test(C3)
        train_val_ratio = 0
        Dataloaders_dic['train'] = getLoadersBySetName('CVC_EndoSceneStill', 'data_C1', target_img_size,
                                          train_val_ratio=train_val_ratio, batch_size=batch_size, shuffle=shuffle)
        Dataloaders_dic['val'] = getLoadersBySetName('CVC_EndoSceneStill', 'data_C2', target_img_size,
                                                     batch_size=batch_size, shuffle=shuffle)
        Dataloaders_dic['test1'] = getLoadersBySetName('CVC_EndoSceneStill', 'data_C3', target_img_size,
                                                       batch_size=batch_size, shuffle=shuffle)
    elif experimentDatasets=='CVC_ClinicDB':
        # CVC train/val, Kvasir Test
        train_val_ratio = 0.8
        dataloasers = getLoadersBySetName('CVC_ClinicDB', 'data_C1',target_img_size, train_val_ratio=train_val_ratio,
                                          shuffle=shuffle, batch_size=batch_size)
        Dataloaders_dic['train'], Dataloaders_dic['val'] = dataloasers
        Dataloaders_dic['test1'] = getLoadersBySetName('Kvasir-SEG', 'data_C1', target_img_size, train_val_ratio=0,
                                                       batch_size=batch_size)
        Dataloaders_dic['test2'] = getLoadersBySetName('Kvasir-SEG', 'data_C2', target_img_size, train_val_ratio=0,
                                                        batch_size=batch_size)
        Dataloaders_dic['test3'] = getLoadersBySetName('Kvasir-SEG', 'data_C3', target_img_size, train_val_ratio=0,
                                                        batch_size=batch_size)
        Dataloaders_dic['test4'] = getLoadersBySetName('Kvasir-SEG', 'data_C4', target_img_size, train_val_ratio=0,
                                                        batch_size=batch_size)
        Dataloaders_dic['test5'] = getLoadersBySetName('Kvasir-SEG', 'data_C5', target_img_size, train_val_ratio=0,
                                                        batch_size=batch_size)
    elif  experimentDatasets=='CVC_ClinicDB_withoutTest':
        # CVC train/val, Kvasir Test
        train_val_ratio = 0.8
        dataloasers = getLoadersBySetName('CVC-ClinicDB', 'data_C1',target_img_size, train_val_ratio=train_val_ratio,
                                          shuffle=shuffle, batch_size=batch_size)
        Dataloaders_dic['train'], Dataloaders_dic['val'] = dataloasers
    elif experimentDatasets in ['CVC_ClinicDB_Brightness','CVC_ClinicDB_Brightness20'
                                ,'CVC_ClinicDB_flipping','CVC_ClinicDB_rotate'
                                ,'CVC_ClinicDB_shear']:
        # CVC train/val, Kvasir Test
        # train_val_ratio = 0.8
        # dataloasers = getLoadersBySetName('CVC_ClinicDB', 'data_C1',target_img_size, train_val_ratio=train_val_ratio,
        #                                   shuffle=shuffle, batch_size=batch_size)
        # Dataloaders_dic['train'], Dataloaders_dic['val'] = dataloasers
        Dataloaders_dic['train'] = getLoadersBySetName(experimentDatasets, 'data_C1', target_img_size, train_val_ratio=0,
                                                       batch_size=batch_size)
        Dataloaders_dic['val'] = getLoadersBySetName(experimentDatasets, 'data_C2', target_img_size, train_val_ratio=0,
                                                       batch_size=batch_size)
        Dataloaders_dic['test1'] = getLoadersBySetName('Kvasir-SEG', 'data_C1', target_img_size, train_val_ratio=0,
                                                       batch_size=batch_size)
        Dataloaders_dic['test2'] = getLoadersBySetName('Kvasir-SEG', 'data_C2', target_img_size, train_val_ratio=0,
                                                        batch_size=batch_size)
        Dataloaders_dic['test3'] = getLoadersBySetName('Kvasir-SEG', 'data_C3', target_img_size, train_val_ratio=0,
                                                        batch_size=batch_size)
        Dataloaders_dic['test4'] = getLoadersBySetName('Kvasir-SEG', 'data_C4', target_img_size, train_val_ratio=0,
                                                        batch_size=batch_size)
        Dataloaders_dic['test5'] = getLoadersBySetName('Kvasir-SEG', 'data_C5', target_img_size, train_val_ratio=0,
                                                        batch_size=batch_size)

        
    else:
        print("I didn't find so called experiment=",experimentDatasets)
        exit(-1)

    return Dataloaders_dic


if __name__ == '__main__':
    '''This main is created to do side experiments'''
    repreducibility()
    # experiment_name=get('http://172.28.0.2:9000/api/sessions').json()[0]['name'].split('.')[0]
    experiment_name= None
    # if inference in the title of the experiment it means we want to do inference
    # true, we need to load weights, set epoch to 0 and delete the training set
    inference = experiment_name.find('inference') >= 0
    learning_rate = 0.01
    input_channels = 3
    number_classes = 3  # output channels should be one mask for binary class
    switch_epoch = [50,150] # when to switch to the next training stage?
    # If true, it means I wanted to save the best generator parameters as well as the best segmentor parameters.
    save_generator_checkpoints = False
    run_in_colab = True

    root_dir = r"E:\Databases\dummyDataset\train"
    child_dir = "data_C1"
    imageDir = 'images_C1'
    maskDir = 'mask_C1'
    colab_dir = "."
    if run_in_colab:
        root_dir = "/content/CVC-ClinicDB"
        colab_dir = "/content/denoising-using-deeplearning"
    num_epochs = 300
    batch_size = 7
    shuffle = False
    lamda = {"l2":1,"grad":10} #L2 and Grad

    # ************** modify for full experiment *************
    # load_to_RAM = True

    resize_factor = 0.75
    target_img_size = (int(288*resize_factor), int(384*resize_factor))


    print("resize_factor={} and image size={}".format(resize_factor, target_img_size))
    # ************** modify for full experiment *************
    # [SegNet, SegNetGRU, SegNetGRU_Symmetric, SegNetGRU_Symmetric_columns,
    # SegNetGRU_Symmetric_columns_shared_EncDec, SegNetGRU_Symmetric_columns_UltimateShare,
    # SegNetGRU_Symmetric_columns_last2stages, SegNetGRU_Symmetric_columns_last2stages_Notshared_EncDec
    # SegNetGRU_Symmetric_columns_last2stages_Notshared_EncDec_smallerH, SegNetGRU_5thStage_only_not_shared,
    # SegNetGRU_4thStage_only_not_shared, SegNetGRU_Symmetric_last2stages_FromEncToDec]
    ########################### Deeplab versions ###################################
    # [Deeplap_resnet50, Deeplap_resnet101, FCN_resnet50, FCN_resnet101, Deeplabv3_GRU_ASPP_resnet50,
    # Deeplabv3_GRU_CombineChannels_resnet50, Deeplabv3_GRU_ASPP_CombineChannels_resnet50, Deeplabv3_LSTM_resnet50]
    ########################### unet model #####################################################
    ################### Proposed
    # [unet-proposed, GenSeg_IncludeX_max_unet_unet,GenSeg_IncludeX_max_unet_deeplab,
    # GenSeg_IncludeX_conv, GenSeg_IncludeX_avg, GenSeg_IncludeX_avgV2_unet_unet,
    # GenSeg_IncludeX_convV2_unet_unet, GenSeg_IncludeX_ColorJitterGenerator_avgV2_unet_unet,
    # GenSeg_IncludeX_ColorJitterGeneratorTrainOnly_avgV2_unet_unet,
    # GenSeg_IncludeAugX_hue_avgV2_unet_unet, GenSeg_IncludeAugX_gray_avgV2_unet_unet,
    # GenSeg_IncludeAugX_hue_avgV2_TL_unet_fcn]
    #
    #################### Conventional Segmentor models (i.e., online augmentation) with avgV2
    # [GenSeg_IncludeX_Conventional_avgV2_blure_unet, GenSeg_IncludeX_Conventional_avgV2_colorjitter_unet
    # GenSeg_IncludeX_Conventional_avgV2_hue_unet, GenSeg_IncludeX_Conventional_avgV2_brightness_unet]
    #################### Conventional Segmentor models (i.e., online augmentation) without avgV2 (i.e., Typical augmentation usage)
    #[GenSeg_IncludeX_Conventional_colorjitter_unet, GenSeg_IncludeX_Conventional_blure_unet,
    # GenSeg_IncludeX_Conventional_hue_unet, GenSeg_IncludeX_Conventional_brightness_unet]
    ################### Vanilla models (i.e., no generator and no augmentation) #######################
    #                 Transfere Learning for vanilla models are added except for Unet
    #['GenSeg_Vanilla_none_unet', GenSeg_Vanilla_none_fcn, GenSeg_Vanilla_none_deeplab]
    #[GenSeg_Vanilla_TL_fcn, GenSeg_Vanilla_TL_deeplab, GenSeg_Vanilla_TL_lraspp]

    ################## Expand Manifold models ##########################
        ############### Denosing-reconstruct auto encoder ############
        #[ExpandMani_unet_withoutskip
    model_name = "ExpandMani_unet_withoutskip"
    model = getModel(model_name)
    if model_name.find('GenSeg')>=0:
        switch_epoch=[-1,-1]
    if model_name.find('ExpandMani') >= 0:
        switch_epoch = [-1, -1]
    if model_name.find('Conventional')>=0 or model_name.find('Vanilla')>=0:
        #we don't have Generator here, hence, nothing to optimize
        lamda = {"l2": 0, "grad": 0}

    # experimentDatasets = (CVC_EndoSceneStill (train/val/test), CVC_ClinicDB,Kvasir-SEG,
    # CVC_ClinicDB_Brightness20, CVC_ClinicDB_flipping, CVC_ClinicDB_withoutTest )
    experimentDatasets = 'CVC_ClinicDB_withoutTest'

    # Start WandB recording
    initializWandb()
    print("Experiment name:",experiment_name)
    print("epochs {} batch size {}".format(num_epochs, batch_size))
############## This is an old code to create train/val/test
    # dataset_info = [(root_dir, child_dir, imageDir, maskDir, target_img_size)]#,
    #                 #("/content/trainData_EndoCV2021_5_Feb2021","data_C2","images_C2","mask_C2",target_img_size)]
    # dataloder_info = (train_val_ratio,batch_size, shuffle)
    # Dataloaders_dic = getDataloadersDic(dataset_info, dataloder_info)
    #
    # dataset_info = ("/content/trainData_EndoCV2021_5_Feb2021", child_dir, imageDir, maskDir, target_img_size)
    # dataloder_info = (0.01,batch_size, shuffle) # from 0:(0.01*datasize) will be for val the rest for test
    # Dataloaders_test_dic = getDataloadersDic(dataset_info, dataloder_info)
    # Dataloaders_dic['test']=Dataloaders_test_dic['val']
    #dataset_name = [Kvasir-SEG*5, CVC_ClinicDB*1 ,ETIS_Larib*1, EndoCV*5] 5= data_C1, data_C2 ... data_C5
    #               CVC_EndoSceneStill, CVC_ClinicDB_withoutTest

    Dataloaders_dic= get_Dataloaders_dic(experimentDatasets)


    print('datasets in total:',Dataloaders_dic.keys())
    for phase in Dataloaders_dic.keys():
        print(f"{phase} images:{len(Dataloaders_dic[phase].dataset)}")

    print("model name:", model_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    print("Training will be on:", device)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = {'generator':nn.MSELoss(), # this is generator loss,
               'segmentor':nn.BCEWithLogitsLoss()}

    # call the training loop,
    # make sure to pass correct checkpoint path, or none if starting with the training
    if inference:
        num_epochs=0
        lamda = {"l2":1,"grad":1}
        checkpoint = torch.load('./denoising-using-deeplearning/checkpoints/highest_IOU_{}.pt'.format(model_name))
        state_dict = getStateDict(checkpoint)
        model.load_state_dict(state_dict)
        Dataloaders_dic.pop('train')

    start = time.time()

    ExpandingManifold_training_loop(num_epochs, optimizer, lamda, model, loss_fn,
                  Dataloaders_dic, device, switch_epoch,colab_dir,
                         model_name)


    wandb.save(colab_dir + '/*.py')
    wandb.save(colab_dir + '/results/*')
    wandb.save(colab_dir + '/models/*')
    wandb.finish()

    total_time = time.time() - start
    print('-' * 50, '\nThe entire experiments completed in {:.0f}h {:.0f}m'.format(total_time // 60 ** 2,
                                                                                   (total_time % 60 ** 2) // 60))
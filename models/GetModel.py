from models.GenSeg_Models import *
from models.unet_withoutskip import UNet as unet_withoutskip
from models.VAEModels import VAE as VAE
from models.unet_withoutskip_withSqueeze import UNet_squeeze as UNet_squeeze
from models.ExpandMani_Models import ExpandMani_AE, ExpandMani_AE_AvgMaskGenSeg, ExpandMani_AE_SpatialInterpolate, ExpandMani_AE_SpatInterTVstyle, ExpandMani_AE_TVstyle

def getModelFrameWork(model_name):
    # identify which models for Gen Seg
    Gen_Seg_arch = model_name.split('_')[-2:]
    pretrained = model_name.find('TL') >= 0
    aug = None
    if model_name.find('hue') >= 0:
        aug = torchvision.transforms.ColorJitter(hue=0.05)
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
        if model_name =='ExpandMani_unetsqueezed':
            model = UNet_squeeze(in_channels=3, out_channels=5,n_blocks=5,activation='relu',normalization='batch',conv_mode='same',dim=2)
        elif model_name == 'ExpandMani_unetwithoutskip':
            #out channels is 5 (2 for mask and 3 for generated images)
            model = unet_withoutskip(in_channels=3, out_channels=5,n_blocks=5,activation='relu',normalization='batch',conv_mode='same',dim=2)
        elif model_name == 'ExpandMani_VAE':
            model = VAE()
        elif model_name.find('AvgMaskGenSeg')>=0:
            model = ExpandMani_AE_AvgMaskGenSeg(Gen_Seg_arch)
        elif model_name.find('ExpandMani_SpatialInterpolate')>=0:
            model = ExpandMani_AE_SpatialInterpolate(Gen_Seg_arch)
        elif model_name.find('ExpandMani_SpatInterTVstyle_')>=0:
            model = ExpandMani_AE_SpatInterTVstyle(Gen_Seg_arch,aug=aug)
        elif model_name.find('ExpandMani_SpatInterTVstyle80')>=0:
            model = ExpandMani_AE_SpatInterTVstyle(Gen_Seg_arch,aug=aug, rescale_rate=True)
        elif model_name.find('ExpandMani_TVstyle')>=0:
            model = ExpandMani_AE_TVstyle(Gen_Seg_arch,aug=aug)
        else:
            model = ExpandMani_AE(Gen_Seg_arch)


    else:
        print('Model name unidentified')
        exit(-1)

    return model

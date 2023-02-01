import torch, torchvision
from torch import nn
from models import MyModelV1, FCNModels, DeepLabModels, unet

'''This model should handle the following:
1- Load pretrained models
2- Flexibility at selecting the Generator and the Segmentor architectures
3- Flexibility at handling the original images and generated images with respect to the final mask
(i.e., type of voting mechanism for generating masks for train/val/test phases).
 The signature of forward: forward(X, phase)-> (generated_images, mask)'''
def getModel(model_name='unet',pretrianed=False, in_channels=3, out_channels=2,):
    if not isinstance(model_name,str):
        return model_name #it means that model_name=torchvision.transforms.Augmentation or nn.Identity or something else
    if model_name=='deeplab':
        model = DeepLabModels.Deeplabv3(num_classes=out_channels, pretrianed=pretrianed)
    elif model_name == 'fcn':
        model = FCNModels.FCN(num_classes=out_channels, pretrianed=pretrianed)
    elif model_name == 'lraspp':
        model = DeepLabModels.Lraspp(num_classes=out_channels, pretrianed=pretrianed)
    elif model_name == 'unet':
        model = unet.UNet(in_channels=in_channels,
              out_channels=out_channels,
              n_blocks=4,
              activation='relu',
              normalization='batch',
              conv_mode='same',
              dim=2)
    else:
        print('unknnown model for the Gen Seg models')
        exit(-1)

    return model

def catOrSplit(tensor_s, chunks=2):
    if isinstance(tensor_s,list):#if list, means we need to concat
        return torch.cat(tensor_s,dim=0)
    else: # or split
        return tensor_s.chunk(chunks)

class GenSeg_IncludeX(nn.Module):
    '''
    This is the base class for GenSeg_IncludeX class. Now we will create subclasses
    '''

    def __init__(self, Gen_Seg_arch=('unet','unet'), augmentation=None,transfer_learning=False):
        super().__init__()
        base = getModel(Gen_Seg_arch[0],out_channels=3,pretrianed=transfer_learning)
        self.augmentation = augmentation
        if isinstance(Gen_Seg_arch[0], str):  # it means Gen_Seg_arch[0]='unet' or 'deeplab' ... etc
            self.Generator = nn.Sequential(base, nn.Sigmoid())
        else:  # it means Gen_Seg_arch[0]=torchvision.transforms.GaussianBlur (i.e., Conventional Aug)
            self.Generator = base
        self.Segmentor = getModel(Gen_Seg_arch[1])

    def forward(self,X):
        generated_images = self.Generator(X)
        generated_images_clone = generated_images.clone().detach()
        if self.augmentation:
            augmented_images = self.augmentation(X)
            X_and_generated_images = catOrSplit([X, generated_images_clone, augmented_images])
        else:
            X_and_generated_images = catOrSplit([X, generated_images_clone])
        masks = self.Segmentor(X_and_generated_images)
        return generated_images, masks

class GenSeg_IncludeX_max(nn.Module):
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks
        if phase !='train':
            predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
            #the results would be (N,C,[orig gen],H,W)
            generated_X_masks_stacked = torch.stack((predicted_masks_gen, predicted_masks_X), dim=2)
            predicted_masks, _ = generated_X_masks_stacked.max(dim=2)
        else:#if it is train, double the number of labels to be (2*N,2,H,W)
            truth_masks = catOrSplit([truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks

class GenSeg_IncludeX_conv(nn.Module):
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)
        self.combine_masks_conv = nn.Conv2d(in_channels=4,out_channels=2,kernel_size=3,padding='same')

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
        #the results would be (N,2+2,H,W)
        generated_X_masks_cat = torch.cat((predicted_masks_gen, predicted_masks_X), dim=1)
        predicted_masks = self.combine_masks_conv(generated_X_masks_cat)#(N,4,H,W)=>(N,2,H,W)

        return generated_images, predicted_masks, truth_masks
class GenSeg_IncludeX_convV2(nn.Module):
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)
        self.combine_masks_conv = nn.Sequential(nn.Conv2d(in_channels=4,out_channels=1,kernel_size=3,padding='same'),
                                                nn.Sigmoid())

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
        #the results would be (N,2+2,H,W)
        generated_X_masks_cat = torch.cat((predicted_masks_gen, predicted_masks_X), dim=1)
        predicted_masks = self.combine_masks_conv(generated_X_masks_cat)#(N,4,H,W)=>(N,1,H,W)
        predicted_masks = predicted_masks.cat((1-predicted_masks),dim=1)#(N,1,H,W)=>(N,2,H,W)

        return generated_images, predicted_masks, truth_masks
class GenSeg_IncludeX_avg(nn.Module):
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
        #the results would be (N,C,[orig gen],H,W)
        generated_X_masks_stacked = torch.stack((predicted_masks_gen, predicted_masks_X), dim=2)
        predicted_masks = generated_X_masks_stacked.mean(dim=2)

        return generated_images, predicted_masks, truth_masks
class GenSeg_IncludeX_avgV2(nn.Module):
    #It is similar to GenSeg_IncludeX_max class, in which the segmentor trained on generated
    #image as if it is original images, meanwhile, the val and test average is applied
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        if phase != 'train':
            predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
            # the results would be (N,C,[orig gen],H,W)
            generated_X_masks_stacked = torch.stack((predicted_masks_gen, predicted_masks_X), dim=2)
            predicted_masks= generated_X_masks_stacked.mean(dim=2)
        else:  # if it is train, double the number of labels to be (2*N,2,H,W)
            truth_masks = catOrSplit([truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks

class GenSeg_IncludeX_ColorJitterGenerator_avgV2(nn.Module):
    # We increase the challenge for the Generator to reconstructed a corrupted images (i.e., augmented).
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        Gen_Seg_arch[0] = nn.Sequential( torchvision.transforms.ColorJitter(brightness=.2, hue=.05),
                                         getModel(Gen_Seg_arch[0],out_channels=3) )
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        if phase != 'train':
            predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
            # the results would be (N,C,[orig gen],H,W)
            generated_X_masks_stacked = torch.stack((predicted_masks_gen, predicted_masks_X), dim=2)
            predicted_masks= generated_X_masks_stacked.mean(dim=2)
        else:  # if it is train, double the number of labels to be (2*N,2,H,W)
            truth_masks = catOrSplit([truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks
class GenSeg_IncludeX_ColorJitterGeneratorTrainOnly_avgV2(nn.Module):
    # We increase the challenge for the Generator to reconstructed a corrupted images (i.e., augmented).
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        Gen_Seg_arch[0] = nn.Sequential( torchvision.transforms.ColorJitter(brightness=.2, hue=.3),
                                         getModel(Gen_Seg_arch[0],out_channels=3) )
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)#this one for train
        # this one doesn't augment the images before fedding it to Generator. Both cases use the Same Generator
        self.baseGenSeg_identity_model = self.baseGenSeg_model
        self.baseGenSeg_identity_model.Generator[0] = nn.Identity()

    def forward(self,X, phase, truth_masks):
        if phase == 'train': # augment images before fedding it to the Generator
            generated_images, predicted_masks = self.baseGenSeg_model(X)
        else: # Original images should be used as an input for the Generator
            generated_images, predicted_masks = self.baseGenSeg_identity_model(X)

        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        if phase != 'train':
            predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
            # the results would be (N,C,[orig gen],H,W)
            generated_X_masks_stacked = torch.stack((predicted_masks_gen, predicted_masks_X), dim=2)
            predicted_masks= generated_X_masks_stacked.mean(dim=2)
        else:  # if it is train, double the number of labels to be (2*N,2,H,W)
            truth_masks = catOrSplit([truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks
class GenSeg_IncludeX_NoCombining(nn.Module):
    #It is similar to GenSeg_IncludeX_max class, in which the segmentor trained on generated
    #image as if it is original images, meanwhile, the val and test average is applied
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        if phase != 'train':
            predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
            predicted_masks = predicted_masks_X
        else:  # if it is train, double the number of labels to be (2*N,2,H,W)
            truth_masks = catOrSplit([truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks
class GenSeg_IncludeAugX_hue_avgV2(nn.Module):
    #It is similar to GenSeg_IncludeX_max class, in which the segmentor trained on generated
    #image as if it is original images, meanwhile, the val and test average is applied
    def __init__(self, Gen_Seg_arch=('unet','unet'),transfer_learning=False):
        super().__init__()
        aug= torchvision.transforms.ColorJitter(hue=0.05)
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch, augmentation=aug,transfer_learning=transfer_learning)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        if phase != 'train':
            predicted_masks_X, predicted_masks_gen, predicted_masks_aug = catOrSplit(predicted_masks, chunks=3)
            # the results would be (N,C,[orig gen],H,W)
            generated_X_masks_stacked = torch.stack((predicted_masks_gen, predicted_masks_X), dim=2)
            predicted_masks= generated_X_masks_stacked.mean(dim=2)
        else:  # if it is train, triple the number of labels to be (3*N,C,H,W) => (Original,Gen,Aug)
            truth_masks = catOrSplit([truth_masks, truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks
class GenSeg_IncludeAugX_gray_avgV2(nn.Module):
    #It is similar to GenSeg_IncludeX_max class, in which the segmentor trained on generated
    #image as if it is original images, meanwhile, the val and test average is applied
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        aug= torchvision.transforms.Grayscale(num_output_channels=3)
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch, augmentation=aug)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        if phase != 'train':
            predicted_masks_X, predicted_masks_gen, predicted_masks_aug = catOrSplit(predicted_masks, chunks=3)
            # the results would be (N,C,[orig gen],H,W)
            generated_X_masks_stacked = torch.stack((predicted_masks_gen, predicted_masks_X), dim=2)
            predicted_masks= generated_X_masks_stacked.mean(dim=2)
        else:  # if it is train, triple the number of labels to be (3*N,C,H,W) => (Original,Gen,Aug)
            truth_masks = catOrSplit([truth_masks, truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks
# this is the default model. Including the original images with the corresponding mask is done in the
#training loop
def unet_proposed():
    generator = unet.UNet(in_channels=3,
                          out_channels=3,
                          n_blocks=4,
                          activation='relu',
                          normalization='batch',
                          conv_mode='same',
                          dim=2)
    generator = nn.Sequential(generator, nn.Sigmoid())
    segmentor = unet.UNet(in_channels=3,
                          out_channels=2,
                          n_blocks=4,
                          activation='relu',
                          normalization='batch',
                          conv_mode='same',
                          dim=2)
    return nn.ModuleList([generator, segmentor])


####################################################################################
###################  SOTA methods with augmentation ################################
class GenSeg_IncludeX_Conventional_avgV2_blure(nn.Module):
    #It is similar to GenSeg_IncludeX_max class, in which the segmentor trained on generated
    #image as if it is original images, meanwhile, the val and test average is applied
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        #the Generator here is simply bluring
        Gen_Seg_arch[0] = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        if phase != 'train':
            predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
            # the results would be (N,C,[orig gen],H,W)
            generated_X_masks_stacked = torch.stack((predicted_masks_gen, predicted_masks_X), dim=2)
            predicted_masks= generated_X_masks_stacked.mean(dim=2)
        else:  # if it is train, double the number of labels to be (2*N,2,H,W)
            truth_masks = catOrSplit([truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks

class GenSeg_IncludeX_Conventional_avgV2_colorjitter(nn.Module):
    #image as if it is original images, meanwhile, the val and test average is applied
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        #the Generator here is simply bluring
        Gen_Seg_arch[0] = torchvision.transforms.ColorJitter(brightness=.2, hue=0.05)
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        if phase != 'train':
            predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
            # the results would be (N,C,[orig gen],H,W)
            generated_X_masks_stacked = torch.stack((predicted_masks_gen, predicted_masks_X), dim=2)
            predicted_masks= generated_X_masks_stacked.mean(dim=2)
        else:  # if it is train, double the number of labels to be (2*N,2,H,W)
            truth_masks = catOrSplit([truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks

class GenSeg_IncludeX_Conventional_avgV2_hue(nn.Module):
    #image as if it is original images, meanwhile, the val and test average is applied
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        #the Generator here is simply bluring
        Gen_Seg_arch[0] = torchvision.transforms.ColorJitter(hue=.05)
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        if phase != 'train':
            predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
            # the results would be (N,C,[orig gen],H,W)
            generated_X_masks_stacked = torch.stack((predicted_masks_gen, predicted_masks_X), dim=2)
            predicted_masks= generated_X_masks_stacked.mean(dim=2)
        else:  # if it is train, double the number of labels to be (2*N,2,H,W)
            truth_masks = catOrSplit([truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks

class GenSeg_IncludeX_Conventional_avgV2_brightness(nn.Module):
    #image as if it is original images, meanwhile, the val and test average is applied
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        #the Generator here is simply bluring
        Gen_Seg_arch[0] = torchvision.transforms.ColorJitter(brightness=.2)
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        if phase != 'train':
            predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
            # the results would be (N,C,[orig gen],H,W)
            generated_X_masks_stacked = torch.stack((predicted_masks_gen, predicted_masks_X), dim=2)
            predicted_masks= generated_X_masks_stacked.mean(dim=2)
        else:  # if it is train, double the number of labels to be (2*N,2,H,W)
            truth_masks = catOrSplit([truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks

########################################################################################################
################## the following have only augmentation in the train (typical usage of Augmentation) #################
class GenSeg_IncludeX_Conventional_colorjitter(nn.Module):
    #image as if it is original images, meanwhile, the val and test average is applied
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        #the Generator here is simply bluring
        Gen_Seg_arch[0] = torchvision.transforms.ColorJitter(brightness=.2, hue=0.05)
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        if phase != 'train': #if val or test, validate only original images masks
            predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
            predicted_masks = predicted_masks_X
        else:  # if it is train, double the number of labels to be (2*N,2,H,W)
            truth_masks = catOrSplit([truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks

class GenSeg_IncludeX_Conventional_blure(nn.Module):
    #image as if it is original images, meanwhile, the val and test average is applied
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        #the Generator here is simply bluring
        Gen_Seg_arch[0] = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        if phase != 'train': #if val or test, validate only original images masks
            predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
            predicted_masks = predicted_masks_X
        else:  # if it is train, double the number of labels to be (2*N,2,H,W)
            truth_masks = catOrSplit([truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks

class GenSeg_IncludeX_Conventional_hue(nn.Module):
    #image as if it is original images, meanwhile, the val and test average is applied
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        #the Generator here is simply bluring
        Gen_Seg_arch[0] = torchvision.transforms.ColorJitter(hue=0.05)
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        if phase != 'train': #if val or test, validate only original images masks
            predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
            predicted_masks = predicted_masks_X
        else:  # if it is train, double the number of labels to be (2*N,2,H,W)
            truth_masks = catOrSplit([truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks

class GenSeg_IncludeX_Conventional_brightness(nn.Module):
    #image as if it is original images, meanwhile, the val and test average is applied
    def __init__(self, Gen_Seg_arch=(None,'unet')):
        super().__init__()
        #the Generator here is simply bluring
        Gen_Seg_arch[0] = torchvision.transforms.ColorJitter(brightness=.2)
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks

        if phase != 'train': #if val or test, validate only original images masks
            predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
            predicted_masks = predicted_masks_X
        else:  # if it is train, double the number of labels to be (2*N,2,H,W)
            truth_masks = catOrSplit([truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks

########################################################################################################
################## Vanilla SOTA models (i.e., no augmentation at all #################
class GenSeg_Vanilla(nn.Module):
    #image as if it is original images, meanwhile, the val and test average is applied
    def __init__(self, Gen_Seg_arch=(None,'unet'), pretrained=False):
        super().__init__()
        #the Generator here is simply bluring
        self.model = getModel(Gen_Seg_arch[1], pretrained)

    def forward(self,X, phase, truth_masks):
        predicted_masks = self.model(X)
        generated_images = X

        return generated_images, predicted_masks, truth_masks

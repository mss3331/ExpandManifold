from torch import nn
from models import MyModelV1, FCNModels, DeepLabModels, unet
from models.unet_withoutskip import UNet as unet_withoutskip
import torch
#from torch.nn import functional as F
def createTruthMask(unormalized_2channels_mask):
    '''The input here is real number, we want to convert it to [0,1] '''
    _, polyp = torch.max(unormalized_2channels_mask, dim=1, keepdim=True)
    background = 1-polyp
    normalized_mask = torch.cat([background, polyp], dim=1)
    return normalized_mask
def catOrSplit(tensor_s, chunks=2):
    if isinstance(tensor_s,list):#if list, means we need to concat
        return torch.cat(tensor_s,dim=0)
    else: # or split
        return tensor_s.chunk(chunks)

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

def getStateDict(checkpoint):
    for key,value in checkpoint.items():
        if key.find('state')>=0:#Skip state_dict, thought print other keys
            continue
        print(key,":",value)
    return checkpoint['state_dict']
def loadCheckPoint(model_name, freez=True):
    model_name = 'ExpandMani_'+model_name #unitwithoutskip ==> ExpandMani_unetwithoutskip
    checkpoint = torch.load('./ExpandManifold/checkpoints/highest_IOU_{}.pt'.format(model_name))
    state_dict = getStateDict(checkpoint)
    generator = getGenerator(model_name)
    generator.load_state_dict(state_dict)
    if freez:
        set_parameter_requires_grad(generator)
    return generator


class ExpandMani_AE(nn.Module):
    '''This model'''
    def __init__(self, Gen_Seg_arch, input=3, out = 2):
        super().__init__()
        self.generator_model = loadCheckPoint(Gen_Seg_arch[0])
        self.segmentor_model = getSegmentor(Gen_Seg_arch[1])

    def forward(self, x, phase, truth_masks, rate, z_vectors=None):
        '''z_vectors here is not needed but lefted as a dummy to be consistent with the AE that requires
        z_vectors'''

        #generate images according to z_prime
        with torch.set_grad_enabled(False):
            #to get the latent vector z
            generator_result = self.generator_model(x, phase, truth_masks, returnZ= True)
            _, _, _, z_vectors = generator_result
            # to generate images according to latent z.
            # The generator will generate images according to z_prime (calculated internally according to rate)
            generator_result = self.generator_model(x, phase, truth_masks, rate=rate, z_vectors= z_vectors)
            generated_images, generated_masks, truth_masks = generator_result
            #Here the generated_masks should have the either 0 or 1 so that BCE make sense ylog(y) + (1-y)log(1-y)
            generated_masks = createTruthMask(generated_masks)


        if phase=='train':
            '''if train then training input is increased 
             original images (x) + generated images along with corresponding masks
             '''
            x = catOrSplit([generated_images, x])
            truth_masks = catOrSplit([generated_masks, truth_masks])

        predicted_masks = self.segmentor_model(x)
        '''
        generated_images: From the decoder part of the generative model
        predicted_masks : From the segmentation model
        truth_masks: Is combination between generated_mask (if we trust the generative model to create a valid mask)
        and the orignial masks
        '''
        return generated_images, predicted_masks, truth_masks


def getSegmentor(model_name='unet',pretrianed=False, in_channels=3, out_channels=2,):
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

def getGenerator(model_name):
    model = None
    if model_name.find('ExpandMani_unetwithoutskip') >= 0:
        # out channels is 5 (2 for mask and 3 for generated images)
        model = unet_withoutskip(in_channels=3, out_channels=5, n_blocks=5, activation='relu', normalization='batch',
                                 conv_mode='same', dim=2)
    return model

class ExpandMani_AE_AvgMaskGenSeg(nn.Module):
    '''This model will average both masks of the generator and segmentor to generate final mask'''
    def __init__(self, Gen_Seg_arch, input=3, out = 2):
        super().__init__()
        self.generator_model = loadCheckPoint(Gen_Seg_arch[0])
        self.segmentor_model = getSegmentor(Gen_Seg_arch[1])

    def forward(self, x, phase, truth_masks, rate, z_vectors=None):
        '''z_vectors here is not needed but lefted as a dummy to be consistent with the AE that requires
        z_vectors'''

        #generate images according to z_prime
        with torch.set_grad_enabled(False):
            #to get the latent vector z
            generator_result = self.generator_model(x, phase, truth_masks, returnZ= True)
            generated_images, generated_masks, _, z_vectors = generator_result

        predicted_masks = self.segmentor_model(x)
        if phase.find('val')>=0 or phase.find('test')>=0:
            avg_mask = 0.5*predicted_masks + 0.5*generated_masks

        '''
        generated_images: From the decoder part of the generative model
        predicted_masks : From the segmentation model
        truth_masks: Is combination between generated_mask (if we trust the generative model to create a valid mask)
        and the orignial masks
        '''
        if phase=='train':
            return generated_images, predicted_masks, truth_masks
        else:
            return generated_images, avg_mask, truth_masks

class ExpandMani_AE_SpatialInterpolate(nn.Module):
    '''This model'''
    def __init__(self, Gen_Seg_arch, input=3, out = 2):
        super().__init__()
        self.generator_model = loadCheckPoint(Gen_Seg_arch[0])
        self.segmentor_model = getSegmentor(Gen_Seg_arch[1])

    def forward(self, x, phase, truth_masks, rate, z_vectors=None):
        '''z_vectors here is not needed but lefted as a dummy to be consistent with the AE that requires
        z_vectors'''

        #generate images according to z_prime
        with torch.set_grad_enabled(False):
            #to get the latent vector z
            generator_result = self.generator_model(x, phase, truth_masks, returnZ= True)
            generated_images, generated_masks, _, _ = generator_result

        if phase=='train':
            '''The input x should be original image and interpolation images 
               the truth_mask should be double for training
             '''
            generated_images = rate*x + (1-rate)* generated_images
            x = catOrSplit([generated_images, x])
            truth_masks = catOrSplit([truth_masks, truth_masks])

        predicted_masks = self.segmentor_model(x)
        '''
        generated_images: From the decoder part of the generative model
        predicted_masks : From the segmentation model
        truth_masks: Is combination between generated_mask (if we trust the generative model to create a valid mask)
        and the orignial masks
        '''
        return generated_images, predicted_masks, truth_masks
class ExpandMani_AE_SpatInterTVstyle(nn.Module):
    '''This model'''
    def __init__(self, Gen_Seg_arch,aug, input=3, out = 2):
        super().__init__()
        self.aug = aug
        self.generator_model = loadCheckPoint(Gen_Seg_arch[0])
        self.segmentor_model = getSegmentor(Gen_Seg_arch[1])

    def forward(self, x, phase, truth_masks, rate, z_vectors=None):
        '''z_vectors here is not needed but lefted as a dummy to be consistent with the AE that requires
        z_vectors'''

        with torch.set_grad_enabled(False):
            generator_result = self.generator_model(x, phase, truth_masks, rate)
            generated_images, generated_masks, truth_masks = generator_result

        input_images = catOrSplit([generated_images, x])
        if phase=='train':
            '''The input x should be original image and interpolation images 
               the truth_mask should be double for training
             '''
            generated_images = rate*x + (1-rate)* generated_images

            if self.aug:
                truth_masks = catOrSplit([truth_masks, truth_masks, truth_masks])
                x_aug = self.aug(x)
                input_images = catOrSplit([x, generated_images, x_aug])
            else:
                truth_masks = catOrSplit([truth_masks, truth_masks])

        predicted_masks = self.segmentor_model(input_images)

        if phase !='train':
            predicted_masks1, predicted_masks2 = catOrSplit(predicted_masks)
            # predicted_masks = (predicted_masks1+predicted_masks2)/2
            predicted_masks = predicted_masks2
        '''
        generated_images: From the decoder part of the generative model
        predicted_masks : From the segmentation model
        truth_masks: Is combination between generated_mask (if we trust the generative model to create a valid mask)
        and the orignial masks
        '''
        return generated_images, predicted_masks, truth_masks

class ExpandMani_AE_TVstyle(nn.Module):
    '''This model mimic GenSeg_IncludeX_avgV2 in which both the Gen and Seg is trained'''
    def __init__(self, Gen_Seg_arch,aug, input=3, out = 2):
        super().__init__()
        self.aug = aug
        self.generator_model = getGenerator('ExpandMani_unetwithoutskip')
        self.segmentor_model = getSegmentor(Gen_Seg_arch[1])

    def forward(self, x, phase, truth_masks, rate, z_vectors=None):
        '''z_vectors here is not needed but lefted as a dummy to be consistent with the AE that requires
        z_vectors'''

        #generate images and masks, predict masks
        generator_result = self.generator_model(x, phase, truth_masks)
        generated_images, generated_masks, truth_masks = generator_result

        generated_images_clone = generated_images.clone().detach()
        input_images = catOrSplit([x, generated_images_clone])
        if phase =='train':
            if self.aug:
                truth_masks = catOrSplit([truth_masks,truth_masks,truth_masks])
                x_aug = self.aug(x)
                input_images = catOrSplit([x, generated_images_clone, x_aug])
            else:
                truth_masks = catOrSplit([truth_masks,truth_masks])

        predicted_masks = self.segmentor_model(input_images)
        if phase !='train':
            # average the result
            predicted_masks1, predicted_masks2 = catOrSplit(predicted_masks)
            # predicted_masks = 0.5*predicted_masks1 + 0.5*predicted_masks2
            predicted_masks = predicted_masks1



        return generated_images, predicted_masks, truth_masks

# aug= torchvision.transforms.ColorJitter(hue=hue)
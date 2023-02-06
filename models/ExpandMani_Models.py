from torch import nn
import torch
from models.GetModel import getModelFrameWork
from models.GetModel import getModel as getSegmentor

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
    generator = getModelFrameWork(model_name)
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


        if phase=='train':
            '''if train then training input is increased 
             original images (x) + generated images along with corresponding masks'''
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
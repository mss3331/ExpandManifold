import torch
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
def deleteTail(x):
  # print(x)
  x = x.split('_mask')[0] # /EndoCV_5_.../C3_0110_mask.jpg --> /EndoCV_5_.../C3_0110
  x = x.split('.jpg')[0] #/EndoCV_5_.../C3_0110 --> /EndoCV_5_.../C3_0110
  # x = x.split('.')[0] # 0110.jpg --> C3_0110
  # print(x)
  return x
class SegDataset(Dataset):
    def __init__(self, parentDir, childDir , imageDir, maskDir, targetSize,augmentation, load_to_RAM= False):
        self.imageList = sorted(glob.glob(parentDir + '/data_'+childDir+'/' + imageDir + '_'+childDir+'/*'), key = deleteTail)
        # self.imageList.sort()
        self.maskList = sorted(glob.glob(parentDir + '/data_'+childDir+'/' + maskDir + '_'+childDir+'/*'), key = deleteTail)
        # self.maskList.sort()
        mismatch = 0
        for pair in zip(self.imageList, self.maskList):
            img_name, mask_name = pair
            img_name = img_name.split('_')[-1].split('.')[0]
            mask_name = mask_name.split('_')[-2]
            if img_name != mask_name:
                mismatch += 1
                # print('img={} mask={}'.format(img_name,mask_name))
        print('Number of mismatch for Data{} is {}'.format(childDir, mismatch))
        assert(mismatch==0)
        #At this stage we are sure that the mask corresponds to its mask
        self.targetSize = targetSize
        self.tensor_images = []
        self.tensor_masks = []
        self.load_to_RAM = load_to_RAM
        self.augmentation = augmentation

        if self.load_to_RAM:# load all data to RAM for faster fetching
            print("Loading dataset to RAM...")
            self.tensor_images = [self.get_tensor_image(image_path) for image_path in self.imageList]
            self.tensor_masks = [self.get_tensor_mask(mask_path) for mask_path in self.maskList]
            print("Finish loading dataset to RAM")

    def __getitem__(self, index):
        if self.load_to_RAM:#if images are loaded to the RAM copy them, otherwise, read them
            x = self.tensor_images[index]
            y = self.tensor_masks[index]
        else:
            x=self.get_tensor_image(self.imageList[index])
            y=self.get_tensor_mask(self.maskList[index])
        return x, y

    def __len__(self):
        return len(self.imageList)

    def get_tensor_image(self, image_path):
        '''this function get image path and return transformed tensor image'''
        preprocess = transforms.Compose([
            # transforms.Resize((384, 288), 2),
            transforms.Resize(self.targetSize, 2),
            self.augmentation,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
        X = Image.open(image_path).convert('RGB')
        X = preprocess(X)
        return X
    def get_tensor_mask(self, mask_path):
        trfresize = transforms.Resize(self.targetSize, 2)
        trftensor = transforms.ToTensor()
        yimg = Image.open(mask_path).convert('L')
        y1 = trftensor(trfresize(yimg))
        y1 = y1.type(torch.BoolTensor)
        y2 = torch.bitwise_not(y1)
        y = torch.cat([y2, y1], dim=0)
        # y.squeeze_()
        return y

#TTR is Train Test Ratio
def trainTestSplit(dataset, TTR):
    '''This function split train test randomely'''
    print("dataset is splitted randomely")
    dataset_size = len(dataset[0]) #dataset is tuble = (megaDataset_augmented, megaDataset_no_augmented)
    dataset_permutation = np.random.permutation(dataset_size)
    # print(dataset_permutation[:10])
    # trainDataset = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    # valDataset = torch.utils.data.Subset(dataset, range(int(TTR*len(dataset)), len(dataset)))
    #
    trainDataset = torch.utils.data.Subset(dataset[0], dataset_permutation[:int(TTR * dataset_size)] )
    valDataset = torch.utils.data.Subset(dataset[1],dataset_permutation[int(TTR * dataset_size):] )
    print("training indices first samples{}\n val indices first samples{}".format(trainDataset.indices[:5],valDataset.indices[:5]))
    # print(trainDataset.dataset[0])
    # exit(0)
    return trainDataset, valDataset
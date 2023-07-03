import torch
import glob
import numpy as np
from pprint import pprint
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import collections

def getLoadersBySetName(dataset_name, data_C,target_img_size,
                        train_val_ratio=0,batch_size=7,shuffle=False, randomSplit= False):
    #input should be dataset_name="CVC-ClinicDB",  data_C="data_C1", split_ratio=0.5
    #output dataloader1 OR dataloader1, dataloader2 if split_ratio != 0 (i.e., we want train/val dataloaders)
    # databases = {'EndoCV': "/content/trainData_EndoCV2021_5_Feb2021",
    #              'CVC_ClinicDB': '/content/CVC-ClinicDB',
    #              'CVC_ClinicDB_Brightness':'/content/CVC-ClinicDB-Brightness',
    #              'CVC_ClinicDB_flipping': '/content/CVC-ClinicDB-flipping',
    #              'Kvasir_Seg': '/content/Kvasir-SEG',
    #              'ETIS_Larib':'/content/ETIS-LaribPolypDB',
    #               'CVC_EndoSceneStill':'/content/CVC-EndoSceneStill'}
    databases = {}
    databases[dataset_name] = '/content/'+dataset_name
    #if the dataset_info is list of tuple, then it will be combined before splitting by the function
    #getDataloadersDic(). Good!!
    if not isinstance(data_C, list):#
        data_C = [data_C]

    dataset_info = []
    for child_dir in data_C:#Create a list of dataset_info
        C = child_dir.split('_')[-1]
        imageDir = 'images_'+C #images_C1 or C2 ... etc
        maskDir = 'mask_'+C
        dataset_info.append((databases[dataset_name],
                             child_dir, imageDir, maskDir, target_img_size))

    dataloder_info = (train_val_ratio, batch_size, shuffle)
    Dataloaders_dic = getDataloadersDic(dataset_info, dataloder_info, randomSplit)

    dataloader1 = Dataloaders_dic['train']
    dataloader2 = Dataloaders_dic['val']
    if train_val_ratio==0:
        return dataloader2

    return dataloader1, dataloader2


def getDataloadersDic(dataset_info, dataloder_info, randomSplit):
    datasets_list = []
    if not isinstance(dataset_info, list):
        dataset_info = [dataset_info]

    for i, info in enumerate(dataset_info):
        datasets_list.append(SegDataset(*info))

    dataset = ConcatDataset(datasets_list)
    train_val_ratio, batchSize, shuffle = dataloder_info
    trainDataset, valDataset = trainTestSplit(dataset, train_val_ratio, randomSplit)
    # -------------Dataloader configurations---------------------
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=shuffle, drop_last=False)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=shuffle, drop_last=False)
    dataloader_dic = {'train': trainLoader, 'val': valLoader}
    return dataloader_dic

class SegDatasetEnjectNoise(Dataset):
    '''This dataset is designed only to train autoencoder by injecting niose and delete part of the images'''

    def __init__(self, parentDir, dataset_name, imageDir, maskDir, targetSize, augmentation=None, load_to_RAM=False):
        self.imageList = sorted(glob.glob("/".join((parentDir, dataset_name, imageDir, '/*'))), key=deleteTail)
        self.maskList = sorted(glob.glob("/".join((parentDir, dataset_name, maskDir, '/*'))), key=deleteTail)

        mismatch = identifyMismatch(self.imageList, self.maskList)
        print('Number of mismatch for Data{} is {}'.format(dataset_name, mismatch))
        assert(mismatch==0)
        # At this stage we are sure that the mask corresponds to its mask
        self.targetSize = targetSize
        self.tensor_images = []
        self.tensor_masks = []
        self.load_to_RAM = load_to_RAM
        self.augmentation = augmentation
        if self.augmentation == None:
            self.augmentation = transforms.Resize(self.targetSize)


    def __getitem__(self, index):
        x,x_noised = self.get_tensor_image(self.imageList[index])
        y_dic = self.get_tensor_mask(self.maskList[index])
        bitwise_mask = y_dic['seg_target']
        single_mask = y_dic['seg_intermediate']
        #singleMask is one channel image mask. y is two channels bitwise mask for cross entropy loss
        return x, single_mask, bitwise_mask

    def __len__(self):
        return len(self.imageList)

    def get_tensor_image(self, image_path, mask):
        '''this function get image path and return transformed tensor image'''
        preprocess = transforms.Compose([
            # transforms.Resize((384, 288), 2),
            transforms.Resize(self.targetSize),
            self.augmentation,
            transforms.ToTensor()])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
        x = Image.open(image_path).convert('RGB')
        x = preprocess(x)
        noise = torch.randn_like(x)
        alpha = 0.96+ 0.04*torch.rand(1) #original image from 0.96 to 1
        x_noised = alpha * x + noise * (1 - alpha)

        polyp_scale = 3*torch.sum(mask)/(mask.nelement())
        #the size of erasing area range between half and full polyp size
        randomErase = transforms.RandomErasing(p=1,scale=(polyp_scale/16,polyp_scale/8))
        x_noised_erased = randomErase(x_noised)
        return x, x_noised_erased

    def get_tensor_mask(self, mask_path):
        trfresize = transforms.Resize(self.targetSize)
        trftensor = transforms.ToTensor()
        yimg = Image.open(mask_path).convert('L')
        y1 = trftensor(trfresize(yimg))
        mask = y1
        y1 = y1.type(torch.BoolTensor)
        y2 = torch.bitwise_not(y1)
        y = torch.cat([y2, y1], dim=0).float()
        mask_dic = {'seg_target': y, 'seg_intermediate': mask}
        # y.squeeze_()
        return mask_dic

class SegDataset(Dataset):
    def __init__(self, parentDir, dataset_name, imageDir, maskDir, targetSize, augmentation=None, load_to_RAM=False):
        self.imageList = sorted(glob.glob("/".join((parentDir, dataset_name, imageDir, '/*'))), key=deleteTail)
        self.maskList = sorted(glob.glob("/".join((parentDir, dataset_name, maskDir, '/*'))), key=deleteTail)

        mismatch = identifyMismatch(self.imageList, self.maskList)
        print('Number of mismatch for Data{} is {}'.format(dataset_name, mismatch))
        assert(mismatch==0)
        # At this stage we are sure that the mask corresponds to its mask
        self.targetSize = targetSize
        self.tensor_images = []
        self.tensor_masks = []
        self.load_to_RAM = load_to_RAM
        self.augmentation = augmentation
        if self.augmentation == None:
            self.augmentation = transforms.Resize(self.targetSize)

        if self.load_to_RAM:  # load all data to RAM for faster fetching
            print("Loading dataset to RAM...")
            self.tensor_images = [self.get_tensor_image(image_path) for image_path in self.imageList]
            self.tensor_masks = [self.get_tensor_mask(mask_path) for mask_path in self.maskList]
            print("Finish loading dataset to RAM")

    def __getitem__(self, index):
        if self.load_to_RAM:  # if images are loaded to the RAM copy them, otherwise, read them
            x = self.tensor_images[index]
            y_dic = self.tensor_masks[index]
            y = y_dic['seg_target']
            intermediate = y_dic['seg_intermediate']
        else:
            x = self.get_tensor_image(self.imageList[index])
            y_dic = self.get_tensor_mask(self.maskList[index])
            y = y_dic['seg_target']
            intermediate = y_dic['seg_intermediate']
        return x, intermediate, y

    def __len__(self):
        return len(self.imageList)

    def get_tensor_image(self, image_path):
        '''this function get image path and return transformed tensor image'''
        preprocess = transforms.Compose([
            # transforms.Resize((384, 288), 2),
            transforms.Resize(self.targetSize),
            self.augmentation,
            transforms.ToTensor()])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
        X = Image.open(image_path).convert('RGB')
        X = preprocess(X)
        return X

    def get_tensor_mask(self, mask_path):
        trfresize = transforms.Resize(self.targetSize)
        trftensor = transforms.ToTensor()
        yimg = Image.open(mask_path).convert('L')
        y1 = trftensor(trfresize(yimg))
        mask = y1
        y1 = y1.type(torch.BoolTensor)
        y2 = torch.bitwise_not(y1)
        y = torch.cat([y2, y1], dim=0).float()
        mask_dic = {'seg_target': y, 'seg_intermediate': mask}
        # y.squeeze_()
        return mask_dic


# TTR is Train Test Ratio
def trainTestSplit(dataset, TTR,randomSplit):
    '''This function split train test randomely'''
    if not isinstance(dataset, collections.abc.Sequence):
        dataset = (dataset, dataset)
    if randomSplit:
        print("dataset is splitted randomely")
        dataset_size = len(dataset[0])  # dataset is tuble = (megaDataset_augmented, megaDataset_no_augmented)
        dataset_permutation = np.random.permutation(dataset_size)
    else:
        print("dataset is splitted NOT randomely")
        dataset_size = len(dataset[0])  # dataset is tuble = (megaDataset_augmented, megaDataset_no_augmented)
        dataset_permutation = np.arange(start=0, stop=dataset_size, step=1, dtype=np.int)

    # print(dataset_permutation[:10])
    # trainDataset = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    # valDataset = torch.utils.data.Subset(dataset, range(int(TTR*len(dataset)), len(dataset)))
    #
    trainDataset = torch.utils.data.Subset(dataset[0], dataset_permutation[:int(TTR * dataset_size)])
    valDataset = torch.utils.data.Subset(dataset[1], dataset_permutation[int(TTR * dataset_size):])
    print("training indices first samples{}\n val indices first samples{}".format(trainDataset.indices[:5],
                                                                                  valDataset.indices[:5]))
    # print(trainDataset.dataset[0])
    # exit(0)
    return trainDataset, valDataset


def deleteTail(x):
    # print(x)
    # D:/Databases/CVC-ClinicDB/data_C1/images_C1\\1.png --> images_C1\\1.png (windows) 1.png (linux)
    x = x.split('/')[-1]
    x = x.split('\\')[-1]  # images_C1\\1.png --> 1.png for both (linux) and (windows)
    x = x.split('_mask')[0]  # C3_0110_mask.jpg --> C3_0110
    x = x.split('.')[0]  # C3_0110.jpg --> C3_0110
    # x = x.split('.')[0] # 0110.jpg --> C3_0110
    # print(x)
    return x


def pruneFileNames(pair, dataset_name="CVC-ClinicDB"):
    # img_name, mask_name = pair
    # if dataset_name == "CVC-ClinicDB":
    #     img_name = img_name.split('\\')[-1].split('/')[-1]
    #     mask_name = mask_name.split('\\')[-1].split('/')[-1]
    # else:  # I think this is for EndoCV
    #     img_name = img_name.split('_')[-1].split('.')[0]
    #     mask_name = mask_name.split('_')[-2]
    names = []
    for x in pair:
        x = x.split('/')[-1]
        x = x.split('\\')[-1]  # images_C1\\1.png --> 1.png for both (linux) and (windows)
        x = x.split('_mask')[0]  # C3_0110_mask.jpg --> C3_0110
        x_temp = x.split('p')[0]  # this is for Larib dataset p10.tif --> 10.tif. Ofcourse EndoScene has files 10.bmp!
        if x_temp == '':
            x = x.split('p')[-1]
        else:
            x = x.split('p')[0]
        x = x.split('.')[0]
        names.append(x)
    return names


def identifyMismatch(imageList, maskList, dataset_name="CVC-ClinicDB", examples=2):
    mismatch = 0
    for pair in zip(imageList, maskList):
        img_name, mask_name = pruneFileNames(pair)
        if img_name != mask_name:
            mismatch += 1
            print('img={} mask={}'.format(img_name,mask_name))
        if examples > 0:
            pprint(pair)
            examples -= 1
    return mismatch

if __name__=='__main__':
    print('testing this script')
    image_path = r'E:\Databases\CVC-ClinicDB\data_C1\images_C1\1.png'
    mask_path = r'E:\Databases\CVC-ClinicDB\data_C1\mask_C1\1.png'
    yimg = Image.open(mask_path).convert('L')
    trfresize = transforms.Resize((200,300))
    trftensor = transforms.ToTensor()
    mask = trftensor(trfresize(yimg))
    preprocess = transforms.Compose([
        # transforms.Resize((384, 288), 2),
        transforms.Resize((200,300)),
        transforms.ToTensor()])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
    x = Image.open(image_path).convert('RGB')
    x = preprocess(x)
    for i in range(10):
        noise = torch.randn_like(x)
        alpha = 0.96+ 0.04*torch.rand(1)

        x_noised = alpha*x + noise*(1-alpha)

        polyp_scale = 3 * torch.sum(mask) / (mask.nelement())
        # the size of erasing area range between half and full polyp size
        randomErase = transforms.RandomErasing(p=1, scale=(polyp_scale / 16, polyp_scale/8))
        x_noised_erased = randomErase(x_noised)
        toPIL = transforms.ToPILImage()
        import matplotlib.pyplot as plt
        # plt.imshow(toPIL(x))
        plt.imshow(toPIL(x_noised_erased))
        # plt.imshow(toPIL(mask))

        plt.show()
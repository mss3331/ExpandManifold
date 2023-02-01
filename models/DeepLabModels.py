import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class Deeplabv3(nn.Module):
    def __init__(self,num_classes,backbone="resnet50",pretrianed=False):
        super(Deeplabv3, self).__init__()

        if backbone.find("resnet50")>=0:
            self.dl = models.segmentation.deeplabv3_resnet50(pretrained=pretrianed, progress=True)
        elif backbone.find("resnet101")>=0:
            self.dl = models.segmentation.deeplabv3_resnet101(pretrained=pretrianed, progress=True)
        else:
            print("backbone for Deeplap not recognized ...\n")
            exit(-1)
        self.dl.classifier[4] = torch.nn.Conv2d(256, num_classes, 1)
        # self.dl.classifier = DeepLabHead(2048, num_classes)
        # self.dl.classifier[0].project[3]=nn.Dropout(p=0, inplace=False)

    def forward(self, x):
        x = self.dl(x)['out']
        # x_softmax = F.softmax(x, dim=1)
        return x#, x_softmax

class Lraspp(nn.Module):
    def __init__(self,num_classes,backbone=None,pretrianed=False):
        super(Lraspp, self).__init__()
        self.dl = models.segmentation.lraspp_mobilenet_v3_large(pretrained=pretrianed, progress=True)
        self.dl.classifier.low_classifier = torch.nn.Conv2d(40, num_classes, 1)
        self.dl.classifier.high_classifier = torch.nn.Conv2d(128, num_classes, 1)


    def forward(self, x):
        x = self.dl(x)['out']
        # x_softmax = F.softmax(x, dim=1)
        return x#, x_softmax


class Deeplabv3_GRU_ASPP(nn.Module):
    def __init__(self,num_classes,backbone="resnet50"):
        super(Deeplabv3_GRU_ASPP, self).__init__()

        if backbone.find("resnet50")>=0:
            self.dl = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        elif backbone.find("resnet101")>=0:
            self.dl = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True)
        else:
            print("backbone for Deeplap not recognized ...\n")
            exit(-1)
        self.dl.classifier[4] = torch.nn.Conv2d(256, num_classes, 1)
        self.in_channels = self.dl.classifier[0].convs[4][1].in_channels
        self.out_channels = self.dl.classifier[0].convs[4][1].out_channels

        self.dl.classifier[0].convs[4]=ASPPPooling_GRU(self.in_channels, self.out_channels)  #this is where the ASPPPooling happening

    def forward(self, x):
        x = self.dl(x)['out']
        # x_softmax = F.softmax(x, dim=1)
        return x#, x_softmax

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for index,mod in enumerate(self):
            x = mod(x)

        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPPPooling_GRU(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling_GRU, self).__init__(
            # nn.AdaptiveAvgPool2d(1),
            nn.GRU(input_size=1,hidden_size=1,batch_first=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]

        for index,mod in enumerate(self):
            if index==0:#GRU model do the following
                x_gru = []
                for batch in x: # for each image do
                    chennel_size = batch.shape[0]
                    batch = batch.view(chennel_size,-1,1)
                    gru_output, h = mod(batch)
                    # print(h.view(-1).equal(gru_output[:,-1].view(-1)))
                    x_gru.append(h.squeeze(0))
                x = torch.stack(x_gru,dim=0).unsqueeze(-1)
            else:
                x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class Deeplabv3_GRU_CombineChannels(nn.Module):
    def __init__(self,num_classes,backbone="resnet50"):
        super(Deeplabv3_GRU_CombineChannels, self).__init__()

        if backbone.find("resnet50")>=0:
            self.dl = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True)
        elif backbone.find("resnet101")>=0:
            self.dl = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True)
        else:
            print("backbone for Deeplap not recognized ...\n")
            exit(-1)
        self.dl.classifier[4] = torch.nn.Conv2d(256, num_classes, 1)
        out_channels = self.dl.classifier[0].project[0].out_channels
        self.project_gru = nn.Sequential(
            # nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            Project_GRU(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.dl.classifier[0].project=self.project_gru

    def forward(self, x):
        x = self.dl(x)['out']
        # x_softmax = F.softmax(x, dim=1)
        return x#, x_softmax

class Project_GRU(nn.Module):
    def __init__(self,out_channels):
        super(Project_GRU, self).__init__()
        self.out_channels = out_channels
        self.gru = nn.GRU(input_size=out_channels, hidden_size = out_channels) #the input is a pixel in the combined features with C=2048 "Seqlenth"

    def forward(self, x):
        feature_maps= [] #feature map for each image (batch,C,H,W)
        for image in x:#for each image in the batch do
            image_shape = image.shape  # (C,H,W)
            image = image.view(5, self.out_channels, *image_shape[1:])  # (5,256, H, W)
            image = image.view(5, self.out_channels, -1)  # (5,256, HW)
            image = image.permute(0, 2, 1)  # (5,HW,256) (seq_len=C, batch=HW, input_size=256)
            # image = image.view(image_shape[0],-1).unsqueeze(dim=-1) #image =(C,HW,1)=>(seq_len=C, batch=HW, input_size=1) one pixel for each seq

            ouput, h = self.gru(image) # h =(1, batch=HW, hidden_size=out_channels)

            # (1,HW,out_channels)=>(HW,out_channels)=>(out_channels,HW)=>(out_channels,H, W)
            h = h.squeeze().transpose(1,0).view(-1,*image_shape[1:])

            feature_maps.append(h)
        x = torch.stack(feature_maps)

        return x

class Deeplabv3_GRU_ASPP_CombineChannels(nn.Module):
    def __init__(self,num_classes,backbone="resnet50"):
        super(Deeplabv3_GRU_ASPP_CombineChannels, self).__init__()

        if backbone.find("resnet50")>=0:
            self.dl = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True)
        elif backbone.find("resnet101")>=0:
            self.dl = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True)
        else:
            print("backbone for Deeplap not recognized ...\n")
            exit(-1)
        self.dl.classifier[4] = torch.nn.Conv2d(256, num_classes, 1)
        self.in_channels = self.dl.classifier[0].convs[4][1].in_channels
        self.out_channels = self.dl.classifier[0].convs[4][1].out_channels

        self.dl.classifier[0].convs[4]=ASPPPooling_GRU(self.in_channels, self.out_channels)  #this is where the ASPPPooling happening

        out_channels = self.dl.classifier[0].project[0].out_channels
        self.project_gru = nn.Sequential(
            # nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            Project_GRU(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.dl.classifier[0].project = self.project_gru

    def forward(self, x):
        x = self.dl(x)['out']
        # x_softmax = F.softmax(x, dim=1)
        return x#, x_softmax

class Deeplabv3_LSTM(nn.Module):
    def conv_layer(self,in_channels):
        return nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=3,padding=1), nn.BatchNorm2d(64), nn.AdaptiveMaxPool2d(1))

    def __init__(self,num_classes,backbone="resnet50"):
        super(Deeplabv3_LSTM, self).__init__()

        if backbone.find("resnet50")>=0:
            self.dl = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        elif backbone.find("resnet101")>=0:
            self.dl = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True)
        else:
            print("backbone for Deeplap not recognized ...\n")
            exit(-1)
        set_parameter_requires_grad(self.dl,True)
        self.num_classes = num_classes
        # self.dl.classifier[4] = torch.nn.Conv2d(256, num_classes, 1)
        self.segmentation_LSTM = nn.LSTM(input_size=64,hidden_size=3456) #input(64*36*48) given input image size of cvc-clinicDB (288,384)
        self.layers = list(self.dl.children()) # => len(layers) = 2 before ASPP and After ASPP
        self.intermediatelayers_layers = list(self.layers[0].children()) # we want the feature map after index 8 and before index 8
        self.convSeq1 = self.conv_layer(1024)
        self.convSeq2 = self.conv_layer(2048)
        self.afterASPP = self.layers[1][0]
        self.convSeq3 = self.conv_layer(256)

        # print(len(self.intermediatelayers_layers))
        # print(self.layers,len(self.layers))
        # exit(0)



    def forward(self, x):
        image_shape = x.shape[-2:]
        batch_size = x.shape[0]

        input_for_lstm = []
        for i in range(len(self.intermediatelayers_layers)):
            x = self.intermediatelayers_layers[i](x)
            if i >= 6:#if we reached the last intermediate layer
                input_for_lstm.append(x)

        x = self.afterASPP(x)
        intermediate_hight_width = x.shape[-2:]
        # print("self.convSeq1(input_for_lstm[0])=",self.convSeq1(input_for_lstm[0]).shape)
        # image_shape = input_for_lstm[0].shape
        input_for_lstm[0] = self.convSeq1(input_for_lstm[0]).flatten(start_dim=1)
        input_for_lstm[1] = self.convSeq2(input_for_lstm[1]).flatten(start_dim=1)
        input_for_lstm.append(self.convSeq3(x).flatten(start_dim=1))

        # [print(j.shape) for j in input_for_lstm]
        # exit(0)
        # print(x.shape)
        stacked_input = torch.stack(input_for_lstm) # ==> the output is (seq=3,batch, input)
        # print("stacked_input.shape=",stacked_input.shape)
        # exit(0)
        output, (hidden, cell) = self.segmentation_LSTM(stacked_input) # ==> hidden.shape = (1, batch size, hidden_size)
        # print(hidden.squeeze().shape)
        # exit(0)
        x=hidden.squeeze().view(batch_size,self.num_classes,*intermediate_hight_width)
        x = nn.functional.interpolate(x,size=image_shape,mode='bilinear', align_corners=False)

        return x#, x_softmax
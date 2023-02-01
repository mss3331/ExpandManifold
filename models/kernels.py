import torch.nn as nn
import torch

class kernels(nn.Module):
    def __init__(self, kernel_size_list=[27, 55, 81], repeat=2):
        super().__init__()
        conv_list = [nn.Conv2d(in_channels=3,out_channels=1,kernel_size=kernel_size, bias=False, padding=kernel_size//2)
                     for kernel_size in kernel_size_list]
        self.models = nn.ModuleList(conv_list)
        self.sig = nn.Sigmoid()
    def forward(self,x):
        x_list = []
        for i, model in enumerate(self.models):
            x_list.append(model(x))

        #concat the outputs
        output = torch.cat(x_list,dim=1) #--> (N, 6, H, W)
        output, index = output.max(dim=1) #-->(N,H,W)
        output = self.sig(output)

        return output

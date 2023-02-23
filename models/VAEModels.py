##########################
# This code is taken from : https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L17/2_VAE_celeba-sigmoid_mse.ipynb
# He has video demonstration about it as well
### MODEL
##########################
import torch.nn as nn
import torch

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :216, :288]


class VAE(nn.Module):
    def __init__(self, out=5):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(32, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            # Add my be to lessen the dimentionality
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Flatten(),
        )

        self.z_mean = torch.nn.Linear(4032, 200)
        self.z_log_var = torch.nn.Linear(4032, 200)

        self.decoder = nn.Sequential(
            torch.nn.Linear(200, 4032),
            Reshape(-1, 64, 7, 9),
            # Added by me to mirror the encoder
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(32, out, stride=2, kernel_size=3, padding=1),
            #
            Trim(),  # 3x217x289 -> 3x216x288

        )
        self.sigmoid = nn.Sigmoid()

    def encoding_fn(self, x):
        x = self.encoder(x)

        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def forward(self, x, phase, truth_masks,rate= None, z_vectors=None, returnZ= False):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z_vectors = self.reparameterize(z_mean, z_log_var)

        x = self.decoder(z_vectors)

        predicted_masks = x[:, :2, :, :]
        generated_images = self.sigmoid(x[:, 2:, :, :])
        if returnZ:  # we need to retrieve z vectors only for segmentation stage to generate z_prime
            return generated_images, predicted_masks, truth_masks, z_vectors
        else:  # this is for training the VAE, hence,
            return generated_images, predicted_masks, truth_masks, (z_mean, z_log_var)



if __name__=='__main__':
    input = torch.randn((1,3,216,288)).to(torch.device('cuda:0'))
    model = VAE().to(torch.device('cuda:0'))
    output = model(input)
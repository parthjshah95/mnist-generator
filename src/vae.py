import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from tqdm import tqdm



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 28x28x1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        # 28x28x8
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1)
        # 14x14x8
        self.batchnorm1 = nn.BatchNorm2d(8)
        # 14x14x8
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        # 7x7x16
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        # 3x3x16
        self.fc1a = nn.Linear(3*3*16, 100)
        # 100 (mu)
        self.fc1b = nn.Linear(3*3*16, 100)
        # 100 (var)
        
        self.fc2 = nn.Linear(100, 3*3*16)
        # 3x3x16
        self.deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0, output_padding=0)
        # 5x5x16
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0, output_padding=0)
        # 7x7x16
        self.batchnorm2 = nn.BatchNorm2d(16)
        # 7x7x16
        self.deconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=0, output_padding=0)
        # 15x15x8
        self.deconv4 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=5, stride=1, padding=0, output_padding=0)
        # 19x19x4
        self.batchnorm3 = nn.BatchNorm2d(4)
        # 19x19x4
        self.deconv5 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=5, stride=1, padding=0, output_padding=0)
        # 23x23x2
        self.deconv6 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=0, output_padding=0)
        # 27x27x1
        # pad - 28x28x1
        self.lastconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    # ReLU are used instead of sigmoid function for faster computation
    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.batchnorm1(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(-1, 3*3*16)
        mu = self.fc1a(x)
        logvar = self.fc1b(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = F.leaky_relu(self.fc2(z))
        z = z.view(-1, 16, 3, 3)
        z = F.leaky_relu(self.deconv1(z))
        z = F.leaky_relu(self.batchnorm2(self.deconv2(z)))
        z = F.leaky_relu(self.deconv3(z))
        z = F.leaky_relu(self.batchnorm3(self.deconv4(z)))
        z = F.leaky_relu(self.deconv5(z))
        z = F.hardsigmoid(self.deconv6(z))
        z = F.pad(z, [0,1,0,1])
        z = self.lastconv(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    
    
class VAE:
    def __init__(self, device):
        self.model = Model().to(device)
        self.device = device
        # initialize optimizer
        # adam optimizer is used instead of adagrad for better convergence
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.epochs = 0

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    

    # loss function defined as sum of KL divergence and reconstruction loss
    def loss_function(self,recon_x, x, mu, logvar):
#         print(torch.min(recon_x), torch.max(recon_x), torch.min(x), torch.max(x))
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld

    def train(self, train_data, epochs, test_data=None, loss_threshold=100):
        self.model.train()
        train_loss_list = []
        test_loss_list = []
        if type(epochs) == type([]):
            min_epochs = epochs[0]
            max_epochs = epochs[1]
        else:
            min_epochs = epochs
            max_epochs = 10*epochs
        epoch = 0
        mov_av_loss = 300
        pbar = tqdm(total=max_epochs)
        while True:
            pbar.update(1)
            if epoch >= max_epochs:
                break
            if (mov_av_loss < loss_threshold) and (epoch > min_epochs):
                break
            epoch += 1
            batch_loss = []
            for data, _ in (train_data):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.detach().cpu().numpy())
            epoch_loss = np.mean(batch_loss)
            mov_av_loss = 0.8*epoch_loss + 0.2*mov_av_loss
            train_loss_list.append(epoch_loss)
            batch_loss_test = []
            if test_data:
                with torch.no_grad():
                    for data,_ in test_data:
                        data = data.to(self.device)
                        recon_batch, mu, logvar = self.model(data)
                        loss = self.loss_function(recon_batch, data, mu, logvar)                   
                        batch_loss_test.append(loss.cpu().numpy())
                    test_loss_list.append(np.mean(batch_loss_test))
                    
        self.epochs += epoch
        pbar.close()
        return train_loss_list, test_loss_list
        

    def test(self, test_data):
        self.model.eval()
        reconstructions = []
        with torch.no_grad():
            for data, _ in (test_data):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                reconstructions.append(recon_batch.cpu().numpy())
        return reconstructions
    
    def encode(self, enc_data):
        self.model.eval()
        latent_vecs = []
        self.model.eval()
        with torch.no_grad():
            for data, _ in (enc_data):
                data = data.to(self.device)
                mu, logvar = self.model.encode(data)
                latent = torch.stack([mu, logvar]).permute(1,0,2)
                latent_vecs.append(latent)
        return torch.cat(latent_vecs)
    
    def decode(self, latent_vectors):
        self.model.eval()
        reconstructions = []
        with torch.no_grad():
            for vector in latent_vectors:
                mu, logvar = vector[0].to(self.device), vector[1].to(self.device)
                z = self.model.reparameterize(mu, logvar)
                recon = self.model.decode(z)
                reconstructions.append(recon)
        return torch.cat(reconstructions)
    
    
    
    
class VAE_conditional(VAE):
    def temp(self):
        pass
    
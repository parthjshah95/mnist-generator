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



class Model_small(nn.Module):
    def __init__(self, dct_freqs, layers):
        super().__init__()
        vector_size = np.prod(dct_freqs)
        self.fc1 = nn.Linear(vector_size, layers[0])
        self.fc1a = nn.Linear(layers[0], layers[1])
        self.fc21 = nn.Linear(layers[1], layers[2])
        self.fc22 = nn.Linear(layers[1], layers[2])
        self.fc3 = nn.Linear(layers[2], layers[1])
        self.fc3a = nn.Linear(layers[1], layers[0])
        self.fc4 = nn.Linear(layers[0], vector_size)

    # ReLU are used instead of sigmoid function for faster computation
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc1a(h1))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc3a(h3))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    
    
class VAE_small:
    def __init__(self, dct_freqs, device, layers=[640, 320, 160]):
        self.layers = layers
        self.model = Model_small(dct_freqs, layers).to(device)
        self.device = device
        self.dct_freqs = dct_freqs
        # initialize optimizer
        # adam optimizer is used instead of adagrad for better convergence
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.epochs = 0

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    

    # loss function defined as sum of KL divergence and reconstruction loss
    def loss_function(self,recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld

    def train(self, train_data, epochs, test_data=None, loss_threshold=100):
        self.model.train()
        train_loss_list = []
        test_loss_list = []
        vector_size = np.prod(self.dct_freqs)
        if type(epochs) == type([]):
            min_epochs = epochs[0]
            max_epochs = epochs[1]
        else:
            min_epochs = epochs
            max_epochs = 10*epochs
        epoch = 0
        mov_av_loss = 300
        tqdm.write('starting training...')
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
                data = data.reshape([-1, vector_size]).to(self.device)
                assert data.size()[-1] == vector_size, "data size: "+str(data.size())
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
                        data = data.reshape([-1, vector_size]).to(self.device)
                        assert data.size()[-1] == vector_size, "data size: "+str(data.size())
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
        vector_size = np.prod(self.dct_freqs)
        with torch.no_grad():
            for data, _ in (test_data):
                data = data.reshape([-1, vector_size]).to(self.device)
                recon_batch, mu, logvar = self.model(data)
                reconstructions.append(recon_batch.cpu().numpy())
        return reconstructions
    
    def encode(self, enc_data):
        self.model.eval()
        vector_size = np.prod(self.dct_freqs)
        latent_vectors = []
        with torch.no_grad():
            for data, _ in (enc_data):
                data = data.reshape([-1, vector_size]).to(self.device)
                mu, logvar = self.model.encode(data)
                latent_vector = [mu.cpu().numpy(), logvar.cpu().numpy()]
                latent_vectors.append(latent_vector)
        return np.array(latent_vectors)
    
    def decode(self, latent_vectors):
        self.model.eval()
        vector_size = np.prod(self.dct_freqs)
        reconstructions = []
        with torch.no_grad():
            for vector in latent_vectors:
                mu, logvar = vector[0].to(self.device), vector[1].to(self.device)
                z = self.model.reparameterize(mu, logvar)
                recon = self.model.decode(z)
                reconstructions.append(recon.cpu().numpy())
        return np.array(reconstructions)
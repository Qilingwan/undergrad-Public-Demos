# main.py: DCGAN for CIFAR-10 Image Generation Pipeline
# Author: Chen
# Description: This script implements a complete DCGAN model to generate synthetic CIFAR-10 images using PyTorch. It includes data loading from pickle files, model definitions, adversarial training loop, and visualization of generated samples and loss curves.
# Tip: This code was originally run on Kaggle with GPU acceleration. If running in a different environment, please adjust input/output paths (e.g., /kaggle/input/, /kaggle/working/) and ensure CUDA availability.

import torch, torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import pickle, numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

def loadCifarData(root):
    data, labels = [], []
    for batch in Path(root).glob('data_batch*'):
        with open(batch, 'rb') as f:
            entry = pickle.load(f, encoding='bytes')
            data.append(entry[b'data'])
            labels.extend(entry[b'labels'])
    X = np.vstack(data).reshape(-1, 3, 32, 32).astype('float32')/255.
    y = np.array(labels)
    return torch.tensor(X).float(), torch.tensor(y).long()

X, y = loadCifarData('/kaggle/input/cifar-10/cifar-10-batches-py')
dataset = torch.utils.data.TensorDataset(X, y)
loader  = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Dataset loaded: {len(X)} images on {device}")

class Generator(nn.Module):
    def __init__(self, zDim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(zDim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z): return self.net(z.view(z.size(0), -1, 1, 1))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).view(-1)

def trainModel(epochs=50, zDim=100, lr=2e-4):
    G, D = Generator(zDim).to(device), Discriminator().to(device)
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixedZ = torch.randn(64, zDim, device=device)
    hist = {'gLoss': [], 'dLoss': []}

    for epoch in range(1, epochs+1):
        for real,_ in loader:
            real = real.to(device)*2-1
            bsz = real.size(0)
            ones = torch.ones(bsz, device=device)
            zeros = torch.zeros(bsz, device=device)

            z = torch.randn(bsz, zDim, device=device)
            fake = G(z).detach()
            lossD = criterion(D(real), ones) + criterion(D(fake), zeros)
            optD.zero_grad(); lossD.backward(); optD.step()

            z = torch.randn(bsz, zDim, device=device)
            fake = G(z)
            lossG = criterion(D(fake), ones)
            optG.zero_grad(); lossG.backward(); optG.step()

        hist['gLoss'].append(lossG.item())
        hist['dLoss'].append(lossD.item())

        if epoch % 5 == 0:
            with torch.no_grad():
                sample = G(fixedZ).cpu()*0.5 + 0.5
                torchvision.utils.save_image(sample, f'/kaggle/working/samples/epoch_{epoch}.png', nrow=8)
            torch.save(G.state_dict(), f'/kaggle/working/checkpoints/G_epoch_{epoch}.pth')
            print(f'Epoch {epoch}/{epochs}  G_loss:{lossG:.3f}  D_loss:{lossD:.3f}')
    return hist

(Path("/kaggle/working/samples")).mkdir(parents=True, exist_ok=True)
(Path("/kaggle/working/checkpoints")).mkdir(parents=True, exist_ok=True)

hist = trainModel()
plt.plot(hist['gLoss'], label='Generator'); plt.plot(hist['dLoss'], label='Discriminator')
plt.legend(); plt.title('Training Loss'); plt.savefig('/kaggle/working/lossCurve.png')
plt.show()
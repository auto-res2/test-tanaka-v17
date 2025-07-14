import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class NetQ(nn.Module):
    def __init__(self, z_dim):
        super(NetQ, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(32 * 8 * 8, z_dim)

    def forward(self, x, rank=0):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        z = self.fc(x)
        return z

class NetG(nn.Module):
    def __init__(self, z_dim):
        super(NetG, self).__init__()
        self.fc = nn.Linear(z_dim, 32 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        batch_size = z.size(0)
        x = self.fc(z)
        x = x.view(batch_size, 32, 8, 8)
        out = self.deconv(x)
        return out

class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(32 * 8 * 8, 1)

    def forward(self, x, rank=0):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        out = self.fc(x)
        return out

class LWGAN(nn.Module):
    def __init__(self, z_dim, netQ, netG, netD, device=torch.device("cpu")):
        super(LWGAN, self).__init__()
        self.z_dim = z_dim
        self.netQ = netQ
        self.netG = netG
        self.netD = netD
        self.device = device

    def D_loss(self, real_data, fake_data, rank: int, abs: bool = False):
        post_data = self.netG(self.netQ(real_data, rank))
        diff = self.netD(post_data, rank) - self.netD(fake_data, rank)
        losses = -torch.abs(diff) if abs else -diff
        return losses.mean()

    def GQ_loss(self, real_data, fake_data, rank: int, abs: bool = False):
        n = real_data.shape[0]
        post_data = self.netG(self.netQ(real_data, rank))
        l2 = torch.linalg.norm((real_data - post_data).view(n, -1), dim=-1)
        diff = self.netD(post_data, rank) - self.netD(fake_data, rank)
        losses = l2 + torch.abs(diff) if abs else l2 + diff
        return losses.mean()

    def recon_loss(self, real_data, rank: int):
        n = real_data.shape[0]
        post_data = self.netG(self.netQ(real_data, rank))
        l2 = torch.linalg.norm((real_data - post_data).view(n, -1), dim=-1)
        return l2.mean()

    def forward(self, x1, x2, rank: int, lambda_mmd: float, lambda_rank: float):
        n = x1.shape[0]
        noise = torch.randn(n, self.z_dim, device=self.device)
        fake_data = self.netG(noise)
        cost_GQ = self.GQ_loss(x1, fake_data, rank, abs=False)
        primal_cost = cost_GQ + lambda_rank * rank
        return primal_cost

class IALWGAN(LWGAN):
    def __init__(self, z_dim, netQ, netG, netD, device=torch.device("cpu")):
        super(IALWGAN, self).__init__(z_dim, netQ, netG, netD, device)

    def isometric_loss(self, data):
        """
        Compute a simple isometric loss.
        Here we compute pairwise distances in the original data space (flattened)
        and in the latent space (using netQ), and then use MSE loss to encourage
        similarity between these distance matrices.
        NOTE: For efficiency, we use only a subset of pairs.
        """
        batch_size = data.size(0)
        data_flat = data.view(batch_size, -1)
        latent = self.netQ(data, rank=0)
        data_dists = torch.cdist(data_flat, data_flat, p=2)
        latent_dists = torch.cdist(latent, latent, p=2)
        loss = F.mse_loss(latent_dists, data_dists)
        return loss

def train_model(model, dataloader, optimizer, num_epochs=5, lambda_iso=1.0, lambda_mmd=0.1, lambda_rank=0.1):
    """
    Train the given model for a few epochs.
    We assume that model.forward computes the original loss.
    For IALWGAN, we add lambda_iso * isometric_loss.
    """
    model.train()
    loss_history = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(model.device)
            optimizer.zero_grad()
            loss = model.forward(data, data, rank=0, lambda_mmd=lambda_mmd, lambda_rank=lambda_rank)
            if hasattr(model, 'isometric_loss'):
                iso_loss = model.isometric_loss(data)
                loss += lambda_iso * iso_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
    return loss_history

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
from scipy.stats import pearsonr

def extract_embeddings(model, dataloader, num_samples=200):
    model.eval()
    embeddings = []
    images = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(model.device)
            emb = model.netQ(data, rank=0)
            embeddings.append(emb.cpu())
            images.append(data.cpu())
            if sum(item.size(0) for item in embeddings) >= num_samples:
                break
    embeddings = torch.cat(embeddings, dim=0)[:num_samples]
    images = torch.cat(images, dim=0)[:num_samples]
    return embeddings, images

def compute_distance_correlation(embeddings, feature_output=None):
    if feature_output is None:
        feature_output = embeddings
    latent_dists = torch.cdist(embeddings, embeddings, p=2).numpy().flatten()
    feature_dists = torch.cdist(feature_output, feature_output, p=2).numpy().flatten()
    r, _ = pearsonr(latent_dists, feature_dists)
    return r

def latent_interpolation(model, img1, img2, steps=10, device=torch.device("cpu")):
    model.eval()
    with torch.no_grad():
        z1 = model.netQ(img1.unsqueeze(0).to(device), rank=0)
        z2 = model.netQ(img2.unsqueeze(0).to(device), rank=0)
        interpolated = []
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            generated = model.netG(z_interp)
            interpolated.append(generated.squeeze(0).cpu())
    return interpolated

def plot_interpolations(interpolations, title="Interpolation", filename="interpolation.pdf"):
    grid = utils.make_grid(torch.stack(interpolations), nrow=len(interpolations), normalize=True, scale_each=True)
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")
    plt.savefig(f".research/iteration1/images/{filename}", bbox_inches="tight")
    print(f"Saved interpolation plot as {filename}")
    plt.close()

def evaluate_lambda(lambda_iso_value, model_constructor, dataloader, device):
    """ Train one instance of IALWGAN with a given lambda_iso value and return metrics. """
    from train import NetQ, NetG, NetD, train_model
    
    z_dim = 100
    netQ = NetQ(z_dim)
    netG = NetG(z_dim)
    netD = NetD()
    model = model_constructor(z_dim, netQ, netG, netD, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    
    print(f"Training with λ_iso = {lambda_iso_value}")
    loss_history = train_model(model, dataloader, optimizer, num_epochs=3, lambda_iso=lambda_iso_value)
    
    model.eval()
    recon_error = 0.0
    mse_loss = nn.MSELoss(reduction='sum')
    with torch.no_grad():
        total_samples = 0
        for data, _ in dataloader:
            data = data.to(device)
            rec = model.netG(model.netQ(data, rank=0))
            recon_error += mse_loss(data, rec).item()
            total_samples += data.size(0)
            break
    recon_error /= total_samples
    
    batch_data, _ = next(iter(dataloader))
    batch_data = batch_data.to(device)
    with torch.no_grad():
        latent_vectors = model.netQ(batch_data, rank=0)
    batch_flat = batch_data.view(batch_data.size(0), -1)
    latent_dists = torch.cdist(latent_vectors, latent_vectors, p=2).cpu().numpy().flatten()
    data_dists = torch.cdist(batch_flat, batch_flat, p=2).cpu().numpy().flatten()
    r, _ = pearsonr(latent_dists, data_dists)
    
    return loss_history, recon_error, r

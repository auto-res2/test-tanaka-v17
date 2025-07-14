#!/usr/bin/env python3
"""
This script conducts three experiments:
  Experiment 1: Ablation Study on the Isometric Regularizer.
  Experiment 2: Hyperparameter Sensitivity Analysis for the isometric loss weight (λ₂).
  Experiment 3: Visual and Quantitative Evaluation of Latent Space Geometry.
  
All figures are saved as PDF files using the required filename format.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from preprocess import get_cifar10_dataloader
from train import NetQ, NetG, NetD, LWGAN, IALWGAN, train_model
from evaluate import extract_embeddings, compute_distance_correlation, latent_interpolation, plot_interpolations, evaluate_lambda

def experiment_1(device):
    print("\nStarting Experiment 1: Ablation Study on the Isometric Regularizer")

    dataloader = get_cifar10_dataloader(batch_size=64, train=True)
    
    z_dim = 100
    netQ = NetQ(z_dim)
    netG = NetG(z_dim)
    netD = NetD()
    
    full_model = IALWGAN(z_dim, netQ, netG, netD, device=device)
    baseline_model = IALWGAN(z_dim, netQ, netG, netD, device=device)
    
    optimizer_full = optim.Adam(full_model.parameters(), lr=0.0002)
    optimizer_baseline = optim.Adam(baseline_model.parameters(), lr=0.0002)
    
    print("Training Full Model with isometric regularizer (λ_iso = 1.0)")
    loss_history_full = train_model(full_model, dataloader, optimizer_full, num_epochs=5, lambda_iso=1.0)
    
    print("Training Baseline Model without isometric regularizer (λ_iso = 0.0)")
    loss_history_baseline = train_model(baseline_model, dataloader, optimizer_baseline, num_epochs=5, lambda_iso=0.0)
    
    plt.figure()
    plt.plot(loss_history_full, label="Full Model")
    plt.plot(loss_history_baseline, label="Baseline")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.title("Loss Curve Comparison for Ablation Study")
    filename = "training_loss_ablation.pdf"
    plt.savefig(f".research/iteration1/images/{filename}", bbox_inches="tight")
    print(f"Saved loss curve plot as {filename}")
    plt.close()

def experiment_2(device):
    print("\nStarting Experiment 2: Hyperparameter Sensitivity Analysis for Isometric Loss Weight (λ₂)")
    
    dataloader = get_cifar10_dataloader(batch_size=64, train=True)
    
    lambda_values = [0.0, 0.1, 0.5, 1.0, 2.0]
    results = {}
    
    for lam in lambda_values:
        history, mse, corr = evaluate_lambda(lam, IALWGAN, dataloader, device)
        results[lam] = {'loss_history': history, 'recon_error': mse, 'distance_corr': corr}
        print(f"λ_iso: {lam}, Reconstruction Error: {mse:.4f}, Distance Correlation: {corr:.4f}")
    
    lambdas = list(results.keys())
    recon_errors = [results[l]['recon_error'] for l in lambdas]
    correlations = [results[l]['distance_corr'] for l in lambdas]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lambdas, recon_errors, marker='o')
    plt.xlabel("λ₂ (isometric loss weight)")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title("Reconstruction Error vs. λ₂")
    
    plt.subplot(1, 2, 2)
    plt.plot(lambdas, correlations, marker='x', color='red')
    plt.xlabel("λ₂ (isometric loss weight)")
    plt.ylabel("Latent-Data Distance Correlation")
    plt.title("Distance Correlation vs. λ₂")
    plt.tight_layout()
    filename = "loss_vs_lambda.pdf"
    plt.savefig(f".research/iteration1/images/{filename}", bbox_inches="tight")
    print(f"Saved hyperparameter sensitivity plots as {filename}")
    plt.close()

def experiment_3(device):
    print("\nStarting Experiment 3: Visual and Quantitative Evaluation of Latent Space Geometry")
    
    dataloader = get_cifar10_dataloader(batch_size=64, train=True)
    
    z_dim = 100
    netQ_full = NetQ(z_dim)
    netG_full = NetG(z_dim)
    netD_full = NetD()
    full_model = IALWGAN(z_dim, netQ_full, netG_full, netD_full, device=device)
    optimizer_full = optim.Adam(full_model.parameters(), lr=0.0002)
    train_model(full_model, dataloader, optimizer_full, num_epochs=3, lambda_iso=1.0)
    
    netQ_base = NetQ(z_dim)
    netG_base = NetG(z_dim)
    netD_base = NetD()
    baseline_model = IALWGAN(z_dim, netQ_base, netG_base, netD_base, device=device)
    optimizer_base = optim.Adam(baseline_model.parameters(), lr=0.0002)
    train_model(baseline_model, dataloader, optimizer_base, num_epochs=3, lambda_iso=0.0)
    
    embeddings_full, _ = extract_embeddings(full_model, dataloader, num_samples=200)
    embeddings_base, _ = extract_embeddings(baseline_model, dataloader, num_samples=200)
    
    corr_full = compute_distance_correlation(embeddings_full)
    corr_base = compute_distance_correlation(embeddings_base)
    print(f"Latent-Distance Correlation (Full Model): {corr_full:.4f}")
    print(f"Latent-Distance Correlation (Baseline): {corr_base:.4f}")
    
    sample_imgs, _ = next(iter(dataloader))
    img1, img2 = sample_imgs[0], sample_imgs[1]
    
    interpolations_full = latent_interpolation(full_model, img1, img2, steps=10, device=device)
    interpolations_base = latent_interpolation(baseline_model, img1, img2, steps=10, device=device)
    
    plot_interpolations(interpolations_full, title="Full Model Latent Interpolations", filename="interpolation_full.pdf")
    plot_interpolations(interpolations_base, title="Baseline Model Latent Interpolations", filename="interpolation_baseline.pdf")

def test_code():
    print("\n===== Starting Test Run =====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    experiment_1(device)
    experiment_2(device)
    experiment_3(device)
    
    print("Test run finished. If you see PDF files generated and printed outputs, the code is working.")
    
    status_enum = "stopped"
    print(f"Status: {status_enum}")

if __name__ == "__main__":
    test_code()

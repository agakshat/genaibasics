# Implements a basic trainer for the diffusion model.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import os
import time
import math
from unet import UNet


class ModelConfig:
    channels = [512, 1024]
    input_dim = [1, 28, 28]
    kernel_size = 3


class DataConfig:
    train_batch_size = 128
    eval_batch_size = 128


class TrainConfig:
    num_epochs = 10
    T = 200
    beta1 = 1e-4
    beta2 = 2e-2
    learning_rate = 1e-3


def variance_schedule(beta1, beta2, T):
    """linear variance schedule."""
    return torch.linspace(beta1, beta2, T)

def get_positional_embedding(seq_length, c, h, w):
    d_input = c * h * w
    pos = torch.arange(0, seq_length).unsqueeze(1)
    denom = torch.exp(torch.arange(0, d_input, 2) * (-math.log(10000.0 / d_input)))
    pos_encoding = torch.zeros(seq_length, d_input)
    pos_encoding[:, 0::2] = torch.sin(pos*denom)
    pos_encoding[:, 1::2] = torch.cos(pos*denom)
    return torch.reshape(pos_encoding, (-1, c, h, w))

def get_alpha_t(betas, t_sample):
    alphas = 1 - betas
    alpha_t_hat = torch.gather(torch.cumprod(alphas, dim=0), dim=0, index=t_sample)
    return alpha_t_hat


def forward_diffusion(x_0, alpha_t_hat):
    alpha_t_hat = alpha_t_hat.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_t_hat) * x_0 + torch.sqrt(1 - alpha_t_hat) * eps
    return x_t, eps

def denormalize(x):
    return (x + 1) * 255 / 2



def main(unused_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = DataConfig()
    train_dataset = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, data_config.train_batch_size)
    test_dataset = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, data_config.eval_batch_size)

    model_config = ModelConfig()
    model = UNet(model_config)
    model.to(device)
    print(f"Model Summary: \n{model}")
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")

    train_config = TrainConfig()
    betas = variance_schedule(train_config.beta1, train_config.beta2, train_config.T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    for epoch in range(train_config.num_epochs):
        print(f"Starting epoch {epoch}.")
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            features, _ = batch # [batch_size, 1, 28, 28]
            features = 2 * features - 1 # range should be [-1, 1]

            features = features.to(device)

            t_sample = torch.randint(low=1, high=train_config.T + 1, size=(features.shape[0],)).to(device)
            alpha_t_hat = get_alpha_t(betas, t_sample)

            x_t, eps = forward_diffusion(features, alpha_t_hat)
            pe = get_positional_embedding(train_config.T, x_t.shape[-3], x_t.shape[-2], x_t.shape[-1]).to(device)

            predicted_noise = model(x_t + torch.index_select(pe, 0, t_sample).to(device))

            loss = torch.mean((predicted_noise - eps) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print("Loss is: ", total_loss)

        print(f"Epoch {epoch} took {time.time() - epoch_start} seconds.")


if __name__ == "__main__":
    main(None)

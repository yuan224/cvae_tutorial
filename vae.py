import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

#this is a VAE model based on the data generated from gencvaedata.py
#the encoder has two hidden layers
#the decoder has two hidden layers

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        hid1_dim = int(input_dim * 0.5)
        hid2_dim = int(hid1_dim * 0.5)
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hid1_dim)
        self.fc2 = nn.Linear(hid1_dim, hid2_dim)
        self.fc21 = nn.Linear(hid2_dim, latent_dim)  # Mean of the latent space
        self.fc22 = nn.Linear(hid2_dim, latent_dim)  # Log variance of the latent space
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hid2_dim)
        self.fc4 = nn.Linear(hid2_dim, hid1_dim)
        self.fc5 = nn.Linear(hid1_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        h4 = torch.relu(self.fc4(h3))
        return torch.sigmoid(self.fc5(h4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # Clip recon_x to avoid log(0) issues
    recon_x = torch.clamp(recon_x, 1e-6, 1.0 - 1e-6)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if( BCE+KLD < 0 ) : print(f"BCE: {BCE.item()}, KLD: {KLD.item()}")
    return BCE + KLD

class CustomDataset(Dataset):
    def __init__(self, distributions, coefficients):
        self.distributions = distributions
        self.coefficients = coefficients

    def __len__(self):
        return len(self.distributions)

    def __getitem__(self, idx):
        return torch.tensor(self.distributions[idx], dtype=torch.float32), torch.tensor(self.coefficients[idx], dtype=torch.float32)

def normalize_distributions(distributions):
    """Normalize the distributions to be within [0, 1]."""
    min_val = distributions.min()
    max_val = distributions.max()
    normalized_distributions = (distributions - min_val) / (max_val - min_val + 1e-6)
    return normalized_distributions, min_val, max_val

def rescale_distributions(generated_distributions, min_val, max_val):
    """Rescale the generated distributions to their original range."""
    return generated_distributions * (max_val - min_val + 1e-6) + min_val

# Load and split data
def load_and_split_data(csv_path):
    df = pd.read_csv(csv_path)
    distributions = df.iloc[:, :100].values.astype(np.float32)
    distributions, min_val, max_val = normalize_distributions(distributions)  # Normalize the distributions

    coefficients = df.iloc[:, 100:].values.astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(distributions, coefficients, test_size=0.2, random_state=42)
    return CustomDataset(X_train, y_train), CustomDataset(X_test, y_test), min_val, max_val

def save_model(model, path='vae_model.pth'):
    torch.save(model.state_dict(), path)

def load_model(model, path='vae_model.pth'):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print("Model loaded from", path)
    else:
        print("Model file does not exist.")

def count_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if 'bias' in name:
            num_params = param.numel()
            print(f'{name}: {num_params} biases')
        else:
            num_params = param.numel()
            print(f'{name}: {num_params} weights')
        total_params += num_params
    print(f'Total parameters: {total_params}')

# Training function
def train_vae(model, train_data, val_data, epochs, batch_size, model_path='vae_model.pth'):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                recon_batch, mu, logvar = model(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                val_loss += loss.item()
        
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader.dataset):.4f}, Validation Loss: {val_loss / len(val_loader.dataset):.4f}')
        save_model(model, model_path)

# Main function
def main():
    input_dim = 100
    latent_dim = 5
    epochs = 100
    batch_size = 25
    csv_path = 'dataforCVAE3.csv'

    # Initialize model
    model = VAE(input_dim, latent_dim)

    # Print number of weights and biases
    count_parameters(model)

    # Load model if the file exists
    model_path = 'vae_model.pth'
    if os.path.exists(model_path):
        load_model(model, model_path)
#       print("Model loaded from", model_path)

    trainflag = False

    if(trainflag) :
        # Load and split data
        train_data, val_data, min_val, max_val = load_and_split_data(csv_path)

        # Train VAE
        train_vae(model, train_data, val_data, epochs, batch_size)
    else :
        train_data, val_data, min_val, max_val = load_and_split_data(csv_path)

    # Example of generating new distributions using the trained VAE
    model.eval()
    with torch.no_grad():
        z = torch.randn(10, latent_dim)
        generated_distributions = model.decode(z).numpy()
        # Rescale each distribution individually
        generated_distributions = rescale_distributions(generated_distributions, min_val, max_val)

    # Display generated distributions
    for i, dist in enumerate(generated_distributions):
        plt.plot(np.linspace(0, 10, 100), dist, label=f'Generated {i+1}')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Generated Distributions using VAE')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

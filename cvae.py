import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os


# This CVAE uses labels in the 100 to 102 columns of the dataframe as condition

# Define the CVAE model
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, cond_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        
        hid1_dim = int(input_dim * 0.5)
        hid2_dim = int(hid1_dim * 0.5)
        
        # Encoder
        self.fc1 = nn.Linear(input_dim + cond_dim, hid1_dim)
        self.fc2 = nn.Linear(hid1_dim, hid2_dim)
        self.fc21 = nn.Linear(hid2_dim, latent_dim)  # Mean of the latent space
        self.fc22 = nn.Linear(hid2_dim, latent_dim)  # Log variance of the latent space
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim + cond_dim, hid2_dim)
        self.fc4 = nn.Linear(hid2_dim, hid1_dim)
        self.fc5 = nn.Linear(hid1_dim, input_dim)

    def encode(self, x, c):
        h1 = torch.relu(self.fc1(torch.cat([x, c], dim=1)))
        h2 = torch.relu(self.fc2(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h3 = torch.relu(self.fc3(torch.cat([z, c], dim=1)))
        h4 = torch.relu(self.fc4(h3))
        return torch.sigmoid(self.fc5(h4))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    recon_x = torch.clamp(recon_x, 1e-6, 1.0 - 1e-6)  # Clamping to avoid log(0)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class CustomDataset(Dataset):
    def __init__(self, distributions, coefficients):
        self.distributions = distributions
        self.coefficients = coefficients

    def __len__(self):
        return len(self.distributions)

    def __getitem__(self, idx):
        x = torch.tensor(self.distributions[idx], dtype=torch.float32)
        c = torch.tensor(self.coefficients[idx], dtype=torch.float32)
        return x, c

def normalize_distributions(distributions):
    """Normalize the distributions to be within [0, 1]."""
    min_val = distributions.min()
    max_val = distributions.max()
    normalized_distributions = (distributions - min_val) / (max_val - min_val + 1e-6)
    return normalized_distributions, min_val, max_val

def load_and_split_data(csv_path):
    df = pd.read_csv(csv_path)
    distributions = df.iloc[:, :100].values.astype(np.float32)
    distributions, min_val, max_val = normalize_distributions(distributions)  # Normalize the distributions
    
    # Extract only one of the three coefficients (a1 (100:101), b1 (101:102), c1 (102:103))
    #coefficients = df.iloc[:, 100:101].values.astype(np.float32)
    #coefficients = df.iloc[:, 101:102].values.astype(np.float32)
    coefficients = df.iloc[:, 102:103].values.astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(distributions, coefficients, test_size=0.2, random_state=42)
    return CustomDataset(X_train, y_train), CustomDataset(X_test, y_test), min_val, max_val

def save_model(model, path='cvae_model.pth'):
    torch.save(model.state_dict(), path)

def load_model(model, path='cvae_model.pth'):
    if (os.path.exists(path)):
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

# Function to generate a distribution
def generate_distribution(a, b, c, x):
    return a * np.exp(-b * (x - c) ** 2)

# Generate new distributions from sampling the latent space with conditions
def generate_distribution_from_z_c(model, cond, latent_dim, min_val, max_val):    
    model.eval()
    with torch.no_grad():
        z = torch.randn(cond.shape[0], latent_dim)
        cond = torch.tensor(cond, dtype=torch.float32)
        generated_distributions = model.decode(z, cond).numpy()
        # Rescale each distribution individually
        generated_distributions = rescale_distributions(generated_distributions, min_val, max_val)

    # Display generated distributions
    for i, dist in enumerate(generated_distributions):
        plt.plot(np.linspace(0, 10, 100), dist, label=f'Generated {i + 1}')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Generated Distributions using CVAE')
    plt.legend()
    plt.show()

# Function to test CVAE
def test_cvae(model, cond, a, b, c):
    x = np.linspace(0, 10, 100)
    original_distribution = generate_distribution(a, b, c, x)
    original_distribution = original_distribution.reshape(1, -1)
    original_distribution, min_val, max_val = normalize_distributions(original_distribution)
    original_distribution_tensor = torch.tensor(original_distribution, dtype=torch.float32)
    cond_tensor = torch.tensor([[cond]], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(original_distribution_tensor, cond_tensor)
        z = model.reparameterize(mu, logvar)
        reconstructed_distribution = model.decode(z, cond_tensor).numpy().flatten()
        reconstructed_distribution = rescale_distributions(reconstructed_distribution, min_val, max_val)

    plt.plot(x, original_distribution.flatten(), label='Original Distribution')
    plt.plot(x, reconstructed_distribution, label='Reconstructed Distribution', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Original vs Reconstructed Distribution')
    plt.legend()
    plt.show()

def train_cvae(model, train_data, val_data, epochs, batch_size, model_path='cvae_model.pth'):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data, cond in train_loader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, cond)
            loss = loss_function(recon_batch, data, mu, logvar)
            if loss.item() < 0:
                print(f"Negative loss at epoch {epoch + 1}, data idx: {data}, condition idx: {cond}, loss: {loss.item()}")
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, cond in val_loader:
                recon_batch, mu, logvar = model(data, cond)
                loss = loss_function(recon_batch, data, mu, logvar)
                val_loss += loss.item()
        
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader.dataset):.4f}, Validation Loss: {val_loss / len(val_loader.dataset):.4f}')
        save_model(model, model_path)

def rescale_distributions(generated_distributions, min_val, max_val):
    """Rescale the generated distributions to their original range."""
    return generated_distributions * (max_val - min_val + 1e-6) + min_val

def generate_distributions(model, cond, latent_dim, min_val, max_val):
    model.eval()
    with torch.no_grad():
        z = torch.randn(cond.shape[0], latent_dim)
        cond = torch.tensor(cond, dtype=torch.float32)
        generated_distributions = model.decode(z, cond).numpy()
        # Rescale each distribution individually
        generated_distributions = rescale_distributions(generated_distributions, min_val, max_val)
    return generated_distributions

# Main function
def main():
    input_dim = 100
    latent_dim = 5
    cond_dim = 1  # Conditioning on (a1, b1, c1)
    epochs = 100
    batch_size = 25
    csv_path = 'dataforCVAE3.csv'

    # Initialize model
    model = CVAE(input_dim, latent_dim, cond_dim)

    # Print number of weights and biases
    count_parameters(model)

    # Load model if the file exists
    model_path = 'cvae_model.pth'
    if os.path.exists(model_path):
        load_model(model, model_path)

    trainflag = False

    if trainflag:
        # Load and split data
        train_data, val_data, min_val, max_val = load_and_split_data(csv_path)

        # Train CVAE
        train_cvae(model, train_data, val_data, epochs, batch_size, model_path)
    else :
        # need to determine min and max values for the distributions
        train_data, val_data, min_val, max_val = load_and_split_data(csv_path)

    # Example of generating new distributions using the trained CVAE
    cond = np.array([[1], [3], [5], [7], [9]])  #conditions
    generate_distribution_from_z_c(model, cond, latent_dim, min_val, max_val)

    # Test CVAE with a generated distribution
    test_cvae(model, cond=9, a=1.0, b=5.0, c=3.0)

if __name__ == '__main__':
    main()

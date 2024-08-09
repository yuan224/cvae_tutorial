import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        hid1_dim = int(input_dim * 0.5)
        hid2_dim = int(hid1_dim * 0.5)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hid1_dim),
            nn.ReLU(),
            nn.Linear(hid1_dim, hid2_dim),
            nn.ReLU(),
            nn.Linear(hid2_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hid2_dim),
            nn.ReLU(),
            nn.Linear(hid2_dim, hid1_dim),
            nn.ReLU(),
            nn.Linear(hid1_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x

def loss_function(recon_x, x):
    return nn.functional.mse_loss(recon_x, x, reduction='sum')

class CustomDataset(Dataset):
    def __init__(self, distributions):
        self.distributions = distributions

    def __len__(self):
        return len(self.distributions)

    def __getitem__(self, idx):
        return torch.tensor(self.distributions[idx], dtype=torch.float32)

# Load and split data
def load_and_split_data(csv_path):
    df = pd.read_csv(csv_path)
    distributions = df.iloc[:, :100].values.astype(np.float32)
    
    X_train, X_test = train_test_split(distributions, test_size=0.2, random_state=42)
    return CustomDataset(X_train), CustomDataset(X_test)

def save_model(model, path='ae_model.pth'):
    torch.save(model.state_dict(), path)

def load_model(model, path='ae_model.pth'):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print("Model loaded from", path)
    else:
        print("Model file does not exist.")

# Training function
def train_autoencoder(model, train_data, val_data, epochs, batch_size):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            recon_batch = model(data)
            loss = loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                recon_batch = model(data)
                loss = loss_function(recon_batch, data)
                val_loss += loss.item()
        
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader.dataset):.4f}, Validation Loss: {val_loss / len(val_loader.dataset):.4f}')

# Generate distributions via sampling the latent space
def generate_distribution_from_z(model,latent_dim, num_samples):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)
        generated_distributions = model.decoder(z).numpy()

    # Display generated distributions
    for i, dist in enumerate(generated_distributions):
        plt.plot(np.linspace(0, 10, 100), dist, label=f'Generated {i+1}')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Generated Distributions using Autoencoder')
    plt.legend()
    plt.show()


# Function to generate a distribution
def generate_distribution(a, b, c, x):
    return a * np.exp(-b * (x - c) ** 2)

# Function to test the autoencoder
def test_autoencoder(model, a, b, c):
    x = np.linspace(0, 10, 100)
    original_distribution = generate_distribution(a, b, c, x)
    original_distribution_tensor = torch.tensor(original_distribution, dtype=torch.float32).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        reconstructed_distribution = model(original_distribution_tensor).numpy().flatten()
    
    plt.plot(x, original_distribution, label='Original Distribution')
    plt.plot(x, reconstructed_distribution, label='Reconstructed Distribution', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Original vs Reconstructed Distribution')
    plt.legend()
    plt.show()

# Main function
def main():
    input_dim = 100
    latent_dim = 5
    epochs = 100
    batch_size = 25
    csv_path = 'dataforCVAE.csv'

    # Initialize model
    model = Autoencoder(input_dim, latent_dim)

    # Load model if the file exists
    model_path = 'ae_model.pth'
    if os.path.exists(model_path):
        load_model(model, model_path)
        print("Model loaded from", model_path)


    trainflag = False

    if trainflag:
        # Load and split data
        train_data, val_data = load_and_split_data(csv_path)    
        # Train Autoencoder
        train_autoencoder(model, train_data, val_data, epochs, batch_size)
        # Save the trained model
        save_model(model)
        print("Model saved to", model_path)


    # Example of generating new distributions using the trained Autoencoder
    generate_distribution_from_z(model,latent_dim, 5)

    # Test Autoencoder with a generated distribution
    test_autoencoder(model, a=1.0, b=5.0, c=3.0)

if __name__ == '__main__':
    main()

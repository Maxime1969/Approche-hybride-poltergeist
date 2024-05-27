import torch as pt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Définir l'architecture du VAE
class VAE(nn.Module):
    def __init__(self, input_dim, hiden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hiden_dim),
            #nn.ReLU(),
            nn.LeakyReLU(0.3),
            nn.Linear(hiden_dim, hiden_dim // 2),
            #nn.ReLU(),
            nn.LeakyReLU(0.3),
            nn.Linear(hiden_dim // 2, latent_dim * 2)  # Le double de latent_dim pour les moyennes et les log_variances
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hiden_dim // 2),
            #nn.ReLU(),
            nn.LeakyReLU(0.3),
            nn.Linear(hiden_dim//2, hiden_dim),
            #nn.ReLU(),
            nn.LeakyReLU(0.3),
            nn.Linear(hiden_dim, input_dim),
            nn.Sigmoid()  # Pour s'assurer que les valeurs reconstruites sont comprises entre 0 et 1
        )

    def reparameterize(self, mu, log_var):
        std = pt.exp(0.5 * log_var)
        eps = pt.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc_output = self.encoder(x)
        mu, log_var = enc_output[:, :latent_dim], enc_output[:, latent_dim:]
        z = self.reparameterize(mu, log_var)
        dec_output = self.decoder(z)
        return dec_output, mu, log_var, z

# Définir la fonction perte ELBO (Evidence Lower Bound)
def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * pt.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Reconstruction loss
    #MSE = nn.MSELoss()(recon_x, x)
    # KL divergence loss
    #KLD = -0.5 * pt.sum(1 + log_var - mu.pow(2) - log_var.exp())
    #KLD = pt.mean(KLD)

    # Total loss
    return BCE + beta*KLD

# Entraînement du VAE
def train_vae(vae, dataloader, optimizer, num_epochs, test_dataloader):
    vae.train()
    losses_tain = []
    losses_test = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = data.to(device)
            recon_batch, mu, log_var, z = vae(inputs)
            loss = loss_function(recon_batch, inputs, mu, log_var)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        losses_tain.append(epoch_loss)

        # Evaluate on test data
        test_loss = evaluate_vae(vae, test_dataloader)
        losses_test.append(test_loss)
    return losses_tain, losses_test

def evaluate_vae(vae, dataloader):
    vae.eval()
    total_loss = 0.0
    with pt.no_grad():
        for data in dataloader:
            inputs = data.to(device)
            recon_batch, mu, log_var, z = vae(inputs)
            loss = loss_function(recon_batch, inputs, mu, log_var)
            total_loss += loss.item()

    return total_loss / len(dataloader.dataset)

def plot_loss_curves(train_losses, test_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs. Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Initialiser le modèle et l'optimiseur
vae = VAE(n_feature.shape[0], hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=lr)

# Entraînement du VAE
train_losses, test_losses = train_vae(vae, X_padded_train, optimizer, epochs, X_padded_test)

# Tracer les courbes des fonctions perte d'entrainement et de test
plot_loss_curves(train_losses, test_losses)

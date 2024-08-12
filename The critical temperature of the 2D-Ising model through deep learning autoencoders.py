import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


class IsingModel:
    def __init__(self, L, T, p):
        self.L = L
        self.T = T
        self.p = p
        self.grid = np.random.choice([-1, 1], size=(L, L))

    def swendsen_wang_step(self):
        clusters = np.zeros_like(self.grid, dtype=int)
        current_cluster = 1
        for i in range(self.L):
            for j in range(self.L): #For each site 
                if clusters[i, j] == 0: #If cluster does not yet have a label
                    stack = [(i, j)] #We create a stack (Needed for DFS Approach)
                    while stack:
                        x, y = stack.pop() #Needed to remove temporarily site from stack (Takes way longer without this)
                        if clusters[x, y] == 0:
                            clusters[x, y] = current_cluster
                            for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                                if 0 <= nx < self.L and 0 <= ny < self.L:
                                    if self.grid[nx, ny] == self.grid[x, y] and clusters[nx, ny] == 0:
                                        if random.random() < self.p:
                                            stack.append((nx, ny))
                    current_cluster += 1
        for cluster in range(1, current_cluster):
            if random.random() < 0.5:
                self.grid[clusters == cluster] *= -1

    def magnetization(self):
        return np.mean(self.grid)

def generate_ising_data(L, T, iterations): #Generates data for plotting and simulation
    model = IsingModel(L, T, p = 1 - np.exp(-2/T))
    data = []
    for _ in range(iterations):
        model.swendsen_wang_step()
        magnetization = model.magnetization()
        data.append(magnetization)
    return np.array(data)

def plot_magnetization_sw(T_values, L, iterations): #Can only work for 2D
    fig, axs = plt.subplots(len(T_values), 2, figsize=(12, 4 * len(T_values)))
    for idx, T in enumerate(T_values):
        magnetizations = generate_ising_data(L, T, iterations)
        iterations_range = np.arange(iterations)
        unique_magnetizations, counts = np.unique(magnetizations, return_counts=True)
        
        axs[idx, 0].plot(iterations_range, magnetizations)
        axs[idx, 0].set_title(f"Magnetization vs Iterations (T = {T})")
        axs[idx, 0].set_xlabel("Iterations")
        axs[idx, 0].set_ylabel("Magnetization")
        
        axs[idx, 1].bar(unique_magnetizations, counts)
        axs[idx, 1].set_title(f"Magnetization vs Counts (T = {T})")
        axs[idx, 1].set_xlabel("Magnetization")
        axs[idx, 1].set_ylabel("Counts")
    
    plt.tight_layout()
    plt.show()

T_values = [0.269185, 0.769185, 1.269185, 1.769185, 2.269185, 2.769185, 3.269185, 3.769185, 4.269185]
L = 20  # Size of the lattice
iterations = 1000  # Number of iterations

plot_magnetization_sw(T_values, L, iterations)

# Generate example data
def create_dataset(L, T_values, iterations): #Appends data of magnetization per iteration
    data = []
    for T in T_values:
        magnetizations = generate_ising_data(L, T, iterations)
        data.extend(magnetizations)
    return np.array(data)

# Autoencoder definition
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 625),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(625, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 625),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(625, input_size),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, train_loader, test_loader, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=30, min_lr=0.000001)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            inputs = data[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs = data[0]
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        scheduler.step(test_loss)
        
        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# Main function
def main():
    L = 20
    T_values = [0.269185, 0.769185, 1.269185, 1.769185, 2.269185, 2.769185, 3.269185, 3.769185, 4.269185]
    iterations = 100
    input_size = 1  # Since the autoencoder input size is 1 (magnetization value)

    # Generate and preprocess data
    data = create_dataset(L, T_values, iterations)
    data = torch.tensor(data, dtype=torch.float32).view(-1, input_size)
    
    dataset = TensorDataset(data)
    train_size = int(0.6666 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = Autoencoder(input_size)
    train_autoencoder(model, train_loader, test_loader, 2000)

if __name__ == "__main__":
    main()
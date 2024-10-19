import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the data
ally_data = np.load('robot2_trajectories.npy', allow_pickle=True)
enemy_data = np.load('robot3_trajectories.npy', allow_pickle=True)

# Prepare data
def prepare_sequences(ally_data, enemy_data, sequence_length=10):
    X, y = [], []
    for ally_traj, enemy_traj in zip(ally_data, enemy_data):
        for i in range(len(ally_traj) - sequence_length):
            X.append(ally_traj[i:i+sequence_length, :3])  # x, y, theta
            y.append(enemy_traj[i+sequence_length, :2])  # x, y of enemy at next step
    return np.array(X), np.array(y)

X, y = prepare_sequences(ally_data, enemy_data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Instantiate the model
model = LSTMModel(input_size=3, hidden_size=128, num_layers=2, output_size=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), 'lstm_model.pth')

# Test the model
model.eval()
with torch.no_grad():
    test_predictions = model(X_test.to(device)).cpu().numpy()

# Visualize results
plt.figure(figsize=(12, 6))
plt.scatter(y_test[:, 0], y_test[:, 1], c='b', label='Actual')
plt.scatter(test_predictions[:, 0], test_predictions[:, 1], c='r', label='Predicted')
plt.legend()
plt.title('Enemy Positions: Actual vs Predicted')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.show()

# Plot a sample trajectory
sample_idx = np.random.randint(0, len(ally_data))
sample_ally = ally_data[sample_idx]
sample_enemy = enemy_data[sample_idx]

model.eval()
with torch.no_grad():
    sample_predictions = []
    for i in range(len(sample_ally) - 10):
        input_seq = torch.FloatTensor(sample_ally[i:i+10, :3]).unsqueeze(0).to(device)
        pred = model(input_seq).cpu().numpy()[0]
        sample_predictions.append(pred)

sample_predictions = np.array(sample_predictions)

plt.figure(figsize=(12, 6))
plt.plot(sample_ally[:, 0], sample_ally[:, 1], 'b-', label='Ally Trajectory')
plt.plot(sample_enemy[:, 0], sample_enemy[:, 1], 'g-', label='Enemy Trajectory')
plt.plot(sample_predictions[:, 0], sample_predictions[:, 1], 'r--', label='Predicted Enemy Trajectory')
plt.legend()
plt.title('Sample Trajectory: Actual vs Predicted')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.show()
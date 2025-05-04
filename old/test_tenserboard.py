from models.siren import ConditionalSIREN, Sine
import torch
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import datetime
from sklearn.datasets import make_classification

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from sklearn.model_selection import train_test_split

'''
tensorboard --logdir='runs_quickstart'
tensorboard --logdir='runs_siren'
'''

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1")
print(f"Using device: {device}")

X, y = make_classification(n_samples=1000,    # 1000 samples
                           n_features=3,     # 10 features
                           n_informative=2,   # 5 informative features
                           n_redundant=1,     # 2 redundant features
                           n_classes=2,       # Binary classification
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size=16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

log_dir = "runs_siren/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

model = ConditionalSIREN(layers=[4, 256, 256, 256, 256, 256, 1], w0=8.0, w0_initial=1.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

max_epochs = 100
for epoch in (pbar := trange(max_epochs, leave=True, position=0, colour="yellow")):
    running_loss = 0.0
    model.train()
    
    for inputs, labels in train_dataloader:     
        optimizer.zero_grad()     

        inputs, labels = inputs.to(device), labels.to(device)

        zs = torch.zeros((inputs.shape[0], 1), device=device)
        zs.fill_(-1)
        z = zs

        # Capture activations and preactivations
        activations = []
        preactivations = []

        # Forward pass through model and log activations
        xz = torch.cat([inputs, z], dim=-1)  # Concatenate inputs and z
        for i, layer in enumerate(model.network):
            if isinstance(layer, nn.Linear):
                preactivation = layer(xz)
                preactivations.append(preactivation)
            if isinstance(layer, Sine):
                activation = layer(preactivation)
                activations.append(activation)
                xz = activation

        outputs = model(inputs, z)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Evaluate test loss at the end of the epoch
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():  # Disable gradient computation for testing
        for test_inputs, test_labels in test_dataloader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            zs_test = torch.zeros((test_inputs.shape[0], 1), device=device)
            zs_test.fill_(-1)
            z_test = zs_test
            test_outputs = model(test_inputs, z_test)
            test_loss += criterion(test_outputs, test_labels.float().unsqueeze(1)).item()
    test_loss /= len(test_dataloader)

    ### log loss to TensorBoard
    writer.add_scalar('Loss/train', running_loss / len(train_dataloader), epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)

    # log weights and gradients
    for name, param in model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    # log pre-activations and activations to TensorBoard
    for i, (preact, act) in enumerate(zip(preactivations, activations)):
        writer.add_histogram(f'Preactivations/layer_{i}', preact, epoch)
        writer.add_histogram(f'Activations/layer_{i}', act, epoch)

    pbar.set_description(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {running_loss/len(train_dataloader):.4f}, Test Loss: {test_loss:.4f}")

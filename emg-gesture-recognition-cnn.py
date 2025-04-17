import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
import numpy as np
import time

# Load data
data = pd.read_csv('/kaggle/input/emg-signal-for-gesture-recognition/EMG-data.csv')

# Select data from channels (all columns except 'time', 'class', and 'label')
X = data.iloc[:, 1:9]  # Columns from channel1 to channel8
y = data['class']  # Class labels

# Data normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Custom dataset class for EMG data
class EMGDataset(Dataset):
    def __init__(self, features, labels):
        # Reshape data for CNN: [samples, channels, sequence_length=1]
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels.values, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create datasets
train_dataset = EMGDataset(X_train, y_train)
val_dataset = EMGDataset(X_val, y_val)
test_dataset = EMGDataset(X_test, y_test)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN-only model architecture
class CNNModel(nn.Module):
    def __init__(self, input_channels, output_size, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the flattened size after convolutions and pooling
        # For 8 channels input, after 3 pooling layers with stride 2: 8 // (2^3) = 1
        self.flat_features = 256 * 1
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Apply convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten the output
        x = x.view(-1, self.flat_features)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Hyperparameters
dropout_rate = 0.4
learning_rate = 0.0005

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model, define the loss function, and optimizer
num_classes = len(y.unique())
model = CNNModel(input_channels=1, output_size=num_classes, dropout_rate=dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Logging metrics for visualization
train_losses = []
val_accuracies = []
val_f1_scores = []
training_times = []
testing_times = []

# Train the model with validation data
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        start_train_time = time.time()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        end_train_time = time.time()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        training_times.append(end_train_time - start_train_time)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        start_test_time = time.time()
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
        end_test_time = time.time()
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        testing_times.append(end_test_time - start_test_time)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}')
        
        # Save the model with the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_cnn_model.pth')

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')
    test_precision = precision_score(all_labels, all_preds, average='macro')
    
    print(f'Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}, Test Recall: {test_recall:.4f}, Test Precision: {test_precision:.4f}')
    
    return test_accuracy, test_f1, test_recall, test_precision

# Function to plot training metrics
def plot_training_metrics(train_losses, val_accuracies, val_f1_scores):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.plot(val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Train and evaluate the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)
test_accuracy, test_f1, test_recall, test_precision = evaluate_model(model, test_loader)

# Plot the training metrics
plot_training_metrics(train_losses, val_accuracies, val_f1_scores)

# Load the best model for final evaluation
model.load_state_dict(torch.load('best_cnn_model.pth'))
print("\nFinal evaluation with best model:")
evaluate_model(model, test_loader)

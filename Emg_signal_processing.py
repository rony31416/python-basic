# 1. Import Required Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 2. Load and Preprocess EMG Data
DATA_PATH = "emg_data"
WINDOW_SIZE = 700
STEP_SIZE = 350  # for overlapping windows
LABEL_MAP = {'pinch': 2, 'rotate': 1}  # Updated to match your labels

def load_emg_file(filepath):
    """Load EMG data from CSV file with timestamp, signal, and label columns"""
    data = pd.read_csv(filepath, header=None, names=['timestamp', 'signal', 'label'])
    return data['signal'].values, data['label'].iloc[0]  # Return signal and label

def segment_signal(signal, window_size=WINDOW_SIZE, step=STEP_SIZE):
    """Create overlapping windows from the signal"""
    segments = []
    for i in range(0, len(signal) - window_size, step):
        segment = signal[i:i+window_size]
        segments.append(segment)
    return segments

# Load data from the new structure
X, y = [], []

# List all txt files in the emg_data directory
txt_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.txt')]

for file in txt_files:
    filepath = os.path.join(DATA_PATH, file)
    signal, label = load_emg_file(filepath)
    segments = segment_signal(signal)
    X.extend(segments)
    y.extend([label] * len(segments))

X = np.array(X)
y = np.array(y)

# Convert labels to 0-based indexing for neural network
# pinch (2) -> 1, rotate (1) -> 0
y_mapped = np.where(y == 2, 1, 0)  # pinch becomes 1, rotate becomes 0

# Normalize signals
X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

print(f"Loaded {len(X)} segments")
print(f"Pinch samples: {np.sum(y_mapped == 1)}")
print(f"Rotate samples: {np.sum(y_mapped == 0)}")

# 3. PyTorch Dataset
class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [N, 1, L]
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_val, y_train, y_val = train_test_split(X, y_mapped, test_size=0.2, stratify=y_mapped)
train_dataset = EMGDataset(X_train, y_train)
val_dataset = EMGDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 4. CNN-LSTM Model
class CNNLSTM(nn.Module):
    def __init__(self, input_size=700, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # [B, C, L]
        x = x.permute(0, 2, 1)  # [B, L, C]
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# 5. Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_model(model, train_loader, val_loader, epochs=20):
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in tqdm(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
        
        train_acc = correct / total
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_output = model(X_val)
                val_pred = val_output.argmax(1)
                val_correct += (val_pred == y_val).sum().item()
        val_acc = val_correct / len(val_dataset)
        print(f"Validation Accuracy: {val_acc:.4f}")

# Train the model
train_model(model, train_loader, val_loader, epochs=20)

# 6. Confusion Matrix & Report
from sklearn.metrics import ConfusionMatrixDisplay

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_val, y_val in val_loader:
        X_val = X_val.to(device)
        pred = model(X_val).argmax(1).cpu()
        all_preds.extend(pred.numpy())
        all_labels.extend(y_val.numpy())

cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=['Rotate', 'Pinch']).plot()
plt.title('Confusion Matrix: EMG Gesture Classification')
plt.show()

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Rotate', 'Pinch']))

# 7. Real-time Prediction Integration
def predict_gesture(signal_window, model, device):
    """Predict gesture from EMG signal window"""
    # Ensure we have exactly 700 samples
    if len(signal_window) < 700:
        # Pad with zeros if too short
        signal_window = np.pad(signal_window, (0, 700 - len(signal_window)), mode='constant')
    elif len(signal_window) > 700:
        # Take the last 700 samples if too long
        signal_window = signal_window[-700:]
    
    # Normalize the signal
    signal_window = (signal_window - np.mean(signal_window)) / np.std(signal_window)
    
    # Convert to tensor and add batch and channel dimensions
    input_tensor = torch.tensor(signal_window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor).argmax(1).item()
        gesture = "Pinch" if pred == 1 else "Rotate"
        
        # Get prediction confidence
        output = model(input_tensor)
        confidence = torch.softmax(output, dim=1).max().item()
        
        return gesture, confidence

# Example usage for real-time prediction:
# Assuming you have a signal buffer from your real-time EMG acquisition
# signal_window = np.array(self.filtered_buffer)  # Your real-time signal buffer
# gesture, confidence = predict_gesture(signal_window, model, device)
# print(f"Detected Gesture: {gesture} (Confidence: {confidence:.2f})")

# Save the trained model
torch.save(model.state_dict(), 'emg_cnn_lstm_model.pth')
print("Model saved as 'emg_cnn_lstm_model.pth'")

# To load the model later:
# model = CNNLSTM().to(device)
# model.load_state_dict(torch.load('emg_cnn_lstm_model.pth'))
# model.eval()

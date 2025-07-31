# EMG Gesture Recognition System
# Complete pipeline from data collection to real-time classification

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
from collections import deque
import threading
import queue
import pandas as pd
import os
from datetime import datetime
import json

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense, 
                                   Dropout, BatchNormalization, TimeDistributed,
                                   Input, Flatten, GlobalMaxPooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class EMGDataCollector:
    """
    Data Collection System for EMG Gesture Recognition
    
    Recommended Data Collection Protocol:
    - 2 gestures: 'cylindrical_grasp' and 'open_hand'
    - 50-100 trials per gesture (minimum 50 for good performance)
    - Each trial: 3-5 seconds duration
    - Rest periods: 2-3 seconds between trials
    - Total sessions: 3-5 sessions on different days for robustness
    """
    
    def __init__(self, port='COM9', baudrate=1000000, fs=500):
        self.port = port
        self.baudrate = baudrate
        self.fs = fs
        self.ser = None
        self.running = False
        
        # Data collection parameters
        self.trial_duration = 4.0  # seconds
        self.rest_duration = 3.0   # seconds
        self.trials_per_gesture = 60  # Recommended: 50-100
        
        # Gesture labels
        self.gestures = ['cylindrical_grasp', 'open_hand', 'rest']
        self.current_gesture = 'rest'
        self.trial_count = 0
        
        # Data storage
        self.collected_data = []
        self.data_buffer = deque(maxlen=int(self.fs * self.trial_duration))
        
        # Filter setup
        self.setup_filters()
        
    def setup_filters(self):
        """Initialize EMG signal filters"""
        # Notch filter (50Hz power line noise)
        f0_notch = 50
        Q_factor = 30.0
        self.b_notch, self.a_notch = signal.iirnotch(f0_notch, Q_factor, self.fs)
        
        # Bandpass filter (20-200Hz for EMG)
        nyquist = 0.5 * self.fs
        low_cutoff = 20 / nyquist
        high_cutoff = 200 / nyquist
        self.b_band, self.a_band = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        
    def connect_serial(self):
        """Connect to serial port"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"✓ Connected to {self.port}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            return False
            
    def apply_filters(self, data):
        """Apply preprocessing filters to EMG data"""
        # Apply notch filter
        filtered = signal.filtfilt(self.b_notch, self.a_notch, data)
        # Apply bandpass filter
        filtered = signal.filtfilt(self.b_band, self.a_band, filtered)
        return filtered
        
    def collect_trial_data(self, gesture_name):
        """Collect data for one trial"""
        print(f"\n🎯 Collecting {gesture_name} - Trial {self.trial_count + 1}")
        print(f"⏱️  Get ready... Starting in 3 seconds")
        time.sleep(3)
        
        # Data collection
        trial_data = []
        start_time = time.time()
        
        print(f"🔴 RECORDING {gesture_name.upper()}!")
        
        while time.time() - start_time < self.trial_duration:
            if self.ser.in_waiting:
                try:
                    data = self.ser.readline().decode('utf-8').strip()
                    value = int(data)
                    trial_data.append(value)
                except:
                    continue
                    
        print(f"✓ Trial completed! Collected {len(trial_data)} samples")
        
        # Apply filters and store
        if len(trial_data) >= self.fs * 2:  # Minimum 2 seconds of data
            filtered_data = self.apply_filters(trial_data)
            
            self.collected_data.append({
                'gesture': gesture_name,
                'trial': self.trial_count,
                'data': filtered_data,
                'timestamp': datetime.now().isoformat(),
                'fs': self.fs
            })
            
            self.trial_count += 1
            return True
        else:
            print("⚠️  Insufficient data collected. Retrying...")
            return False
            
    def collect_gesture_dataset(self, gesture_name, num_trials):
        """Collect complete dataset for one gesture"""
        print(f"\n📊 Starting data collection for: {gesture_name}")
        print(f"📋 Protocol: {num_trials} trials × {self.trial_duration}s each")
        print(f"⏸️  Rest {self.rest_duration}s between trials")
        
        successful_trials = 0
        attempt = 0
        
        while successful_trials < num_trials:
            attempt += 1
            if self.collect_trial_data(gesture_name):
                successful_trials += 1
                
            # Rest period between trials
            if successful_trials < num_trials:
                print(f"😴 Rest for {self.rest_duration} seconds...")
                time.sleep(self.rest_duration)
                
        print(f"✅ Completed {gesture_name}: {successful_trials} trials")
        
    def collect_full_dataset(self):
        """Collect data for all gestures"""
        if not self.connect_serial():
            return False
            
        print("🚀 EMG Gesture Data Collection Started")
        print("=" * 50)
        
        try:
            for gesture in ['cylindrical_grasp', 'open_hand']:
                self.collect_gesture_dataset(gesture, self.trials_per_gesture)
                
                # Break between gestures
                print(f"\n⏸️  Take a 30-second break before next gesture...")
                time.sleep(30)
                
        except KeyboardInterrupt:
            print("\n⏹️  Collection stopped by user")
        finally:
            if self.ser:
                self.ser.close()
                
        return True
        
    def save_dataset(self, filename=None):
        """Save collected dataset"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emg_gesture_dataset_{timestamp}.json"
            
        # Convert numpy arrays to lists for JSON serialization
        save_data = []
        for trial in self.collected_data:
            trial_copy = trial.copy()
            trial_copy['data'] = trial['data'].tolist()
            save_data.append(trial_copy)
            
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        print(f"💾 Dataset saved: {filename}")
        print(f"📈 Total trials: {len(self.collected_data)}")
        
        # Print summary
        gesture_counts = {}
        for trial in self.collected_data:
            gesture = trial['gesture']
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
            
        print("📊 Dataset Summary:")
        for gesture, count in gesture_counts.items():
            print(f"   {gesture}: {count} trials")
            
        return filename

class EMGFeatureExtractor:
    """Extract features from EMG signals for CNN-LSTM model"""
    
    def __init__(self, window_size=500, overlap=0.5):
        self.window_size = window_size  # samples per window
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        
    def extract_time_features(self, signal_window):
        """Extract time-domain features"""
        features = []
        
        # Statistical features
        features.append(np.mean(signal_window))
        features.append(np.std(signal_window))
        features.append(np.var(signal_window))
        features.append(np.max(signal_window))
        features.append(np.min(signal_window))
        features.append(np.median(signal_window))
        
        # RMS
        features.append(np.sqrt(np.mean(signal_window**2)))
        
        # Zero crossings
        zero_crossings = np.sum(np.diff(np.sign(signal_window)) != 0)
        features.append(zero_crossings)
        
        # Slope sign changes
        diff_signal = np.diff(signal_window)
        slope_changes = np.sum(np.diff(np.sign(diff_signal)) != 0)
        features.append(slope_changes)
        
        return features
        
    def extract_frequency_features(self, signal_window, fs=500):
        """Extract frequency-domain features"""
        features = []
        
        # FFT
        fft_vals = np.fft.fft(signal_window)
        fft_magnitude = np.abs(fft_vals[:len(fft_vals)//2])
        
        # Frequency bins
        freqs = np.fft.fftfreq(len(signal_window), 1/fs)[:len(fft_vals)//2]
        
        # Mean and median frequency
        power_spectrum = fft_magnitude**2
        total_power = np.sum(power_spectrum)
        
        if total_power > 0:
            mean_freq = np.sum(freqs * power_spectrum) / total_power
            median_freq_idx = np.where(np.cumsum(power_spectrum) >= total_power/2)[0][0]
            median_freq = freqs[median_freq_idx]
        else:
            mean_freq = 0
            median_freq = 0
            
        features.append(mean_freq)
        features.append(median_freq)
        
        # Spectral energy in different bands
        bands = [(20, 50), (50, 100), (100, 150), (150, 200)]
        for low, high in bands:
            band_mask = (freqs >= low) & (freqs <= high)
            band_energy = np.sum(power_spectrum[band_mask])
            features.append(band_energy)
            
        return features
        
    def create_sequences(self, data, labels, sequence_length=10):
        """Create sequences for LSTM input"""
        X, y = [], []
        
        for i in range(len(data) - sequence_length + 1):
            X.append(data[i:(i + sequence_length)])
            y.append(labels[i + sequence_length - 1])
            
        return np.array(X), np.array(y)
        
    def process_dataset(self, dataset_file, sequence_length=10):
        """Process raw dataset into features for training"""
        print("🔄 Processing dataset...")
        
        # Load dataset
        with open(dataset_file, 'r') as f:
            raw_data = json.load(f)
            
        all_features = []
        all_labels = []
        
        label_encoder = LabelEncoder()
        
        for trial in raw_data:
            signal_data = np.array(trial['data'])
            gesture = trial['gesture']
            
            # Create overlapping windows
            for i in range(0, len(signal_data) - self.window_size + 1, self.step_size):
                window = signal_data[i:i + self.window_size]
                
                # Extract features
                time_features = self.extract_time_features(window)
                freq_features = self.extract_frequency_features(window)
                
                # Combine features
                combined_features = time_features + freq_features
                all_features.append(combined_features)
                all_labels.append(gesture)
                
        # Convert to arrays
        X = np.array(all_features)
        y = label_encoder.fit_transform(all_labels)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y, sequence_length)
        
        print(f"✓ Processed {len(X_seq)} sequences")
        print(f"✓ Feature shape: {X_seq.shape}")
        print(f"✓ Classes: {label_encoder.classes_}")
        
        return X_seq, y_seq, label_encoder, scaler

class EMGGestureClassifier:
    """CNN-LSTM model for EMG gesture classification"""
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_cnn_lstm_model(self):
        """Build CNN-LSTM architecture"""
        model = Sequential([
            # CNN layers for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # LSTM layers for temporal modeling
            LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            LSTM(50, dropout=0.2, recurrent_dropout=0.2),
            
            # Dense layers for classification
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
        
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        print("🏋️ Training CNN-LSTM model...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
            ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7, monitor='val_loss'),
            ModelCheckpoint('best_emg_model.h5', save_best_only=True, monitor='val_accuracy')
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.history
        
    def plot_training_history(self):
        """Plot training metrics"""
        if not self.history:
            print("No training history available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training & Validation Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Training & Validation Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            
        # Training Summary
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        best_val_acc = max(self.history.history['val_accuracy'])
        
        summary_text = f"""Training Summary:
        
Final Training Accuracy: {final_train_acc:.4f}
Final Validation Accuracy: {final_val_acc:.4f}
Best Validation Accuracy: {best_val_acc:.4f}
Total Epochs: {len(self.history.history['loss'])}

Model Architecture:
- CNN layers for feature extraction
- LSTM layers for temporal modeling
- Dropout for regularization
- Adam optimizer with adaptive learning rate"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def evaluate_model(self, X_test, y_test, label_encoder):
        """Comprehensive model evaluation"""
        print("📊 Evaluating model performance...")
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"📈 Test Accuracy: {test_accuracy:.4f}")
        print(f"📉 Test Loss: {test_loss:.4f}")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=label_encoder.classes_))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(12, 5))
        
        # Confusion Matrix Plot
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Normalized Confusion Matrix
        plt.subplot(1, 2, 2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        return test_accuracy, test_loss, y_pred, y_pred_proba

class RealTimeEMGClassifier:
    """Real-time EMG gesture classification"""
    
    def __init__(self, model_path, scaler_path, label_encoder_path, 
                 port='COM9', baudrate=1000000, fs=500):
        self.port = port
        self.baudrate = baudrate
        self.fs = fs
        
        # Load trained model and preprocessors
        self.model = tf.keras.models.load_model(model_path)
        
        # Load scaler and label encoder (you'll need to save these during training)
        import joblib
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(label_encoder_path)
        
        # Real-time processing parameters
        self.window_size = 500
        self.sequence_length = 10
        self.overlap = 0.5
        self.step_size = int(self.window_size * (1 - self.overlap))
        
        # Buffers
        self.signal_buffer = deque(maxlen=5000)  # Store raw signal
        self.feature_buffer = deque(maxlen=50)   # Store features
        self.prediction_buffer = deque(maxlen=10) # Store recent predictions
        
        # Feature extractor
        self.feature_extractor = EMGFeatureExtractor(self.window_size, self.overlap)
        
        # Serial connection
        self.ser = None
        self.running = False
        
        # Setup filters
        self.setup_filters()
        
        # Visualization
        self.setup_visualization()
        
    def setup_filters(self):
        """Setup real-time filters"""
        # Same as in data collection
        f0_notch = 50
        Q_factor = 30.0
        self.b_notch, self.a_notch = signal.iirnotch(f0_notch, Q_factor, self.fs)
        
        nyquist = 0.5 * self.fs
        low_cutoff = 20 / nyquist
        high_cutoff = 200 / nyquist
        self.b_band, self.a_band = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        
        # Filter states for real-time processing
        self.zi_notch = signal.lfilter_zi(self.b_notch, self.a_notch)
        self.zi_band = signal.lfilter_zi(self.b_band, self.a_band)
        
    def setup_visualization(self):
        """Setup real-time visualization"""
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Signal plot
        self.axes[0].set_title('Real-time EMG Signal')
        self.axes[0].set_ylabel('Amplitude')
        self.axes[0].grid(True)
        self.signal_line, = self.axes[0].plot([], [], 'b-', linewidth=1)
        
        # Prediction confidence
        self.axes[1].set_title('Gesture Prediction Confidence')
        self.axes[1].set_ylabel('Probability')
        self.axes[1].set_ylim(0, 1)
        self.axes[1].grid(True)
        
        self.confidence_bars = self.axes[1].bar(
            range(len(self.label_encoder.classes_)), 
            [0] * len(self.label_encoder.classes_),
            color=['red', 'green', 'blue'][:len(self.label_encoder.classes_)]
        )
        self.axes[1].set_xticks(range(len(self.label_encoder.classes_)))
        self.axes[1].set_xticklabels(self.label_encoder.classes_, rotation=45)
        
        # Prediction history
        self.axes[2].set_title('Prediction History')
        self.axes[2].set_ylabel('Predicted Class')
        self.axes[2].set_ylim(-0.5, len(self.label_encoder.classes_) - 0.5)
        self.axes[2].grid(True)
        self.prediction_line, = self.axes[2].plot([], [], 'ro-', markersize=8)
        
        plt.tight_layout()
        
    def connect_serial(self):
        """Connect to serial port"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"✓ Connected to {self.port}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            return False
            
    def apply_realtime_filters(self, value):
        """Apply filters to single data point"""
        # Apply notch filter
        filtered_val, self.zi_notch = signal.lfilter(
            self.b_notch, self.a_notch, [value], zi=self.zi_notch
        )
        
        # Apply bandpass filter
        filtered_val, self.zi_band = signal.lfilter(
            self.b_band, self.a_band, filtered_val, zi=self.zi_band
        )
        
        return filtered_val[0]
        
    def predict_gesture(self):
        """Make gesture prediction from current features"""
        if len(self.feature_buffer) < self.sequence_length:
            return None, None
            
        # Get latest sequence
        feature_sequence = np.array(list(self.feature_buffer)[-self.sequence_length:])
        feature_sequence = feature_sequence.reshape(1, self.sequence_length, -1)
        
        # Predict
        prediction_proba = self.model.predict(feature_sequence, verbose=0)[0]
        predicted_class = np.argmax(prediction_proba)
        predicted_gesture = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return predicted_gesture, prediction_proba
        
    def update_visualization(self, frame):
        """Update real-time plots"""
        # Read serial data
        if self.ser and self.ser.in_waiting:
            try:
                data = self.ser.readline().decode('utf-8').strip()
                raw_value = int(data)
                
                # Apply filters
                filtered_value = self.apply_realtime_filters(raw_value)
                self.signal_buffer.append(filtered_value)
                
                # Extract features when we have enough data
                if len(self.signal_buffer) >= self.window_size:
                    window_data = list(self.signal_buffer)[-self.window_size:]
                    
                    # Extract features
                    time_features = self.feature_extractor.extract_time_features(window_data)
                    freq_features = self.feature_extractor.extract_frequency_features(window_data, self.fs)
                    combined_features = time_features + freq_features
                    
                    # Scale features
                    scaled_features = self.scaler.transform([combined_features])[0]
                    self.feature_buffer.append(scaled_features)
                    
                    # Make prediction
                    predicted_gesture, prediction_proba = self.predict_gesture()
                    
                    if predicted_gesture:
                        self.prediction_buffer.append(np.argmax(prediction_proba))
                        
                        # Update confidence bars
                        for i, bar in enumerate(self.confidence_bars):
                            bar.set_height(prediction_proba[i])
                            
                        # Update prediction history
                        if len(self.prediction_buffer) > 1:
                            x_pred = list(range(len(self.prediction_buffer)))
                            y_pred = list(self.prediction_buffer)
                            self.prediction_line.set_data(x_pred, y_pred)
                            self.axes[2].set_xlim(0, max(10, len(self.prediction_buffer)))
                            
                        # Print prediction
                        confidence = np.max(prediction_proba)
                        print(f"Predicted: {predicted_gesture} (confidence: {confidence:.3f})")
                        
            except Exception as e:
                print(f"Error: {e}")
                
        # Update signal plot
        if len(self.signal_buffer) > 1:
            signal_data = list(self.signal_buffer)[-1000:]  # Show last 1000 samples
            self.signal_line.set_data(range(len(signal_data)), signal_data)
            self.axes[0].set_xlim(0, len(signal_data))
            self.axes[0].relim()
            self.axes[0].autoscale_view()
            
        return [self.signal_line] + list(self.confidence_bars) + [self.prediction_line]
        
    def start_realtime_classification(self):
        """Start real-time gesture classification"""
        if not self.connect_serial():
            return
            
        print("🚀 Starting real-time EMG gesture classification...")
        print("Gestures:", self.label_encoder.classes_)
        
        self.running = True
        
        # Start animation
        ani = FuncAnimation(self.fig, self.update_visualization, 
                          blit=False, interval=100, cache_frame_data=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n⏹️  Classification stopped")
        finally:
            if self.ser:
                self.ser.close()

# ============================================================================
# MAIN WORKFLOW FUNCTIONS
# ============================================================================

def collect_emg_data():
    """Step 1: Collect EMG data for gesture recognition"""
    print("=" * 60)
    print("📊 STEP 1: EMG DATA COLLECTION")
    print("=" * 60)
    
    collector = EMGDataCollector(port='COM9', baudrate=1000000)
    
    print("""
🎯 DATA COLLECTION PROTOCOL:
   
📋 Gestures to collect:
   1. Cylindrical Grasp: Close hand around imaginary cylinder, rotate wrist
   2. Open Hand: Keep hand open and relaxed
   
📊 Collection requirements:
   - 60 trials per gesture (recommended: 50-100)
   - 4 seconds per trial
   - 3 seconds rest between trials
   - Total time: ~15 minutes
   
💡 Tips for good data:
   - Keep electrode placement consistent
   - Perform gestures consistently
   - Stay relaxed during rest periods
   - Avoid excessive muscle tension
    """)
    
    if collector.collect_full_dataset():
        dataset_file = collector.save_dataset()
        print(f"✅ Dataset ready: {dataset_file}")
        return dataset_file
    else:
        print("❌ Data collection failed")
        return None

def train_gesture_model(dataset_file):
    """Step 2: Train CNN-LSTM model for gesture classification"""
    print("=" * 60)
    print("🏋️ STEP 2: MODEL TRAINING")
    print("=" * 60)
    
    # Process dataset
    feature_extractor = EMGFeatureExtractor(window_size=500, overlap=0.5)
    X, y, label_encoder, scaler = feature_extractor.process_dataset(dataset_file, sequence_length=10)
    
    # Save preprocessors for real-time use
    import joblib
    joblib.dump(scaler, 'emg_scaler.pkl')
    joblib.dump(label_encoder, 'emg_label_encoder.pkl')
    print("💾 Saved preprocessors: emg_scaler.pkl, emg_label_encoder.pkl")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"📊 Data split:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples") 
    print(f"   Test: {X_test.shape[0]} samples")
    
    # Build and train model
    input_shape = (X.shape[1], X.shape[2])  # (sequence_length, features)
    num_classes = len(np.unique(y))
    
    classifier = EMGGestureClassifier(input_shape, num_classes)
    model = classifier.build_cnn_lstm_model()
    
    print("\n🏗️ Model Architecture:")
    model.summary()
    
    # Train model
    history = classifier.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate model
    test_accuracy, test_loss, y_pred, y_pred_proba = classifier.evaluate_model(X_test, y_test, label_encoder)
    
    # Save trained model
    model.save('emg_gesture_model.h5')
    print("💾 Model saved: emg_gesture_model.h5")
    
    return test_accuracy, history

def start_realtime_classification():
    """Step 3: Real-time gesture classification"""
    print("=" * 60)
    print("🚀 STEP 3: REAL-TIME CLASSIFICATION")
    print("=" * 60)
    
    try:
        classifier = RealTimeEMGClassifier(
            model_path='emg_gesture_model.h5',
            scaler_path='emg_scaler.pkl', 
            label_encoder_path='emg_label_encoder.pkl',
            port='COM9',
            baudrate=1000000
        )
        
        classifier.start_realtime_classification()
        
    except FileNotFoundError as e:
        print(f"❌ Missing required files: {e}")
        print("Please run data collection and training first!")

def run_complete_pipeline():
    """Run the complete EMG gesture recognition pipeline"""
    print("🎯 EMG GESTURE RECOGNITION SYSTEM")
    print("=" * 60)
    
    while True:
        print("\n📋 Choose an option:")
        print("1. 📊 Collect EMG Data")
        print("2. 🏋️ Train Model") 
        print("3. 🚀 Real-time Classification")
        print("4. 🔄 Complete Pipeline")
        print("5. ❌ Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            dataset_file = collect_emg_data()
            if dataset_file:
                print(f"✅ Ready for training with: {dataset_file}")
                
        elif choice == '2':
            dataset_file = input("Enter dataset filename: ").strip()
            if os.path.exists(dataset_file):
                test_accuracy, history = train_gesture_model(dataset_file)
                print(f"✅ Model trained! Test accuracy: {test_accuracy:.4f}")
            else:
                print("❌ Dataset file not found!")
                
        elif choice == '3':
            start_realtime_classification()
            
        elif choice == '4':
            # Complete pipeline
            print("🔄 Running complete pipeline...")
            
            # Step 1: Data collection
            dataset_file = collect_emg_data()
            if not dataset_file:
                print("❌ Pipeline stopped: Data collection failed")
                continue
                
            # Step 2: Model training
            test_accuracy, history = train_gesture_model(dataset_file)
            print(f"✅ Model trained! Test accuracy: {test_accuracy:.4f}")
            
            # Step 3: Real-time classification
            input("\n⏸️ Press Enter to start real-time classification...")
            start_realtime_classification()
            
        elif choice == '5':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice!")

# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ============================================================================

def analyze_dataset(dataset_file):
    """Analyze collected dataset"""
    print("📊 DATASET ANALYSIS")
    print("=" * 40)
    
    with open(dataset_file, 'r') as f:
        data = json.load(f)
        
    # Basic statistics
    gesture_stats = {}
    total_duration = 0
    
    for trial in data:
        gesture = trial['gesture']
        duration = len(trial['data']) / trial['fs']
        total_duration += duration
        
        if gesture not in gesture_stats:
            gesture_stats[gesture] = {'count': 0, 'total_duration': 0, 'samples': []}
            
        gesture_stats[gesture]['count'] += 1
        gesture_stats[gesture]['total_duration'] += duration
        gesture_stats[gesture]['samples'].extend(trial['data'])
    
    print(f"📈 Total trials: {len(data)}")
    print(f"⏱️ Total duration: {total_duration:.1f} seconds")
    print("\n📊 Per-gesture statistics:")
    
    for gesture, stats in gesture_stats.items():
        avg_duration = stats['total_duration'] / stats['count']
        signal_std = np.std(stats['samples'])
        signal_mean = np.mean(stats['samples'])
        
        print(f"\n   {gesture}:")
        print(f"      Trials: {stats['count']}")
        print(f"      Avg duration: {avg_duration:.2f}s")
        print(f"      Signal mean: {signal_mean:.2f}")
        print(f"      Signal std: {signal_std:.2f}")
    
    # Plot sample signals
    plt.figure(figsize=(15, 8))
    
    for i, (gesture, stats) in enumerate(gesture_stats.items()):
        plt.subplot(2, len(gesture_stats), i + 1)
        sample_data = stats['samples'][:2000]  # First 2000 samples
        plt.plot(sample_data)
        plt.title(f'{gesture}\n(Sample Signal)')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Histogram
        plt.subplot(2, len(gesture_stats), i + 1 + len(gesture_stats))
        plt.hist(stats['samples'], bins=50, alpha=0.7)
        plt.title(f'{gesture}\n(Amplitude Distribution)')
        plt.xlabel('Amplitude')
        plt.ylabel('Frequency')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def benchmark_model_performance():
    """Benchmark different model architectures"""
    print("🏃 MODEL PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    # This function would compare different architectures
    # CNN-only, LSTM-only, CNN-LSTM, etc.
    # Implementation would be similar to main training but with different architectures
    pass

def export_model_for_deployment():
    """Export model for deployment on embedded systems"""
    print("📦 EXPORTING MODEL FOR DEPLOYMENT")
    print("=" * 50)
    
    # Load trained model
    model = tf.keras.models.load_model('emg_gesture_model.h5')
    
    # Convert to TensorFlow Lite for embedded deployment
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save TensorFlow Lite model
    with open('emg_gesture_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("✅ TensorFlow Lite model saved: emg_gesture_model.tflite")
    
    # Model size comparison
    import os
    keras_size = os.path.getsize('emg_gesture_model.h5') / (1024 * 1024)
    tflite_size = os.path.getsize('emg_gesture_model.tflite') / (1024 * 1024)
    
    print(f"📊 Model sizes:")
    print(f"   Keras model: {keras_size:.2f} MB")
    print(f"   TFLite model: {tflite_size:.2f} MB")
    print(f"   Compression: {(1 - tflite_size/keras_size)*100:.1f}%")

# ============================================================================
# JUPYTER NOTEBOOK STYLE FUNCTIONS
# ============================================================================

def notebook_data_collection():
    """Jupyter notebook style data collection"""
    print("🔬 NOTEBOOK: Data Collection")
    
    # This would be run in separate cells in Jupyter
    collector = EMGDataCollector()
    
    # Cell 1: Setup
    print("Setting up data collector...")
    
    # Cell 2: Collect gesture 1
    print("Collecting cylindrical grasp data...")
    # collector.collect_gesture_dataset('cylindrical_grasp', 60)
    
    # Cell 3: Collect gesture 2  
    print("Collecting open hand data...")
    # collector.collect_gesture_dataset('open_hand', 60)
    
    # Cell 4: Save and analyze
    print("Saving dataset...")
    # dataset_file = collector.save_dataset()
    # analyze_dataset(dataset_file)

def notebook_model_training():
    """Jupyter notebook style model training"""
    print("🔬 NOTEBOOK: Model Training")
    
    # This would be run in separate cells in Jupyter
    
    # Cell 1: Load and preprocess data
    print("Loading and preprocessing data...")
    
    # Cell 2: Build model
    print("Building CNN-LSTM model...")
    
    # Cell 3: Train model
    print("Training model...")
    
    # Cell 4: Evaluate and visualize
    print("Evaluating model performance...")

def notebook_realtime_demo():
    """Jupyter notebook style real-time demo"""
    print("🔬 NOTEBOOK: Real-time Classification Demo")
    
    # This would be run in separate cells in Jupyter
    print("Starting real-time classification...")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    🎯 EMG GESTURE RECOGNITION SYSTEM
    ================================
    
    This system provides complete pipeline for EMG-based gesture recognition:
    
    📊 Data Collection: Collect EMG signals for different gestures
    🏋️ Model Training: Train CNN-LSTM model for classification  
    🚀 Real-time Classification: Classify gestures in real-time
    
    Features:
    ✅ Complete data collection protocol
    ✅ Advanced signal preprocessing
    ✅ CNN-LSTM neural network architecture
    ✅ Comprehensive evaluation metrics
    ✅ Real-time visualization
    ✅ Model deployment ready
    
    """)
    
    # Run the complete pipeline
    run_complete_pipeline()

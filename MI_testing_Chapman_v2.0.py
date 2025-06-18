# ---------------------------------------------------
# Testing AMI Detection Model on Chapman-Shaoxing Dataset
# Following protocol in Chen et al., 2021 (Front. Cardiovasc. Med.)
# ---------------------------------------------------

#%% Cell 1: Setup & Imports
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import wfdb
from scipy.signal import butter, filtfilt, resample
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    roc_curve
)
from tensorflow.keras.models import load_model


# %% Cell 2: Paths
# Set your local dataset and model paths
chapman_base_path = r"D:\srinivas\Syphrosyne\rhythm_detection\MI"  # update as needed
model_path = r"D:\srinivas\Syphrosyne\rhythm_detection\MI\mi_detection_resnet_model_v4.1.h5"

# %% Cell 3: Load labels
label_df = pd.read_csv(os.path.join(chapman_base_path, 'Diagnostics.csv'))
# Define all MI-related codes
mi_labels = {'MI', 'MIBW', 'MIFW', 'MILW', 'MISW'}

# Create the binary AMI label column
label_df['AMI'] = label_df['Beat'].apply(
    lambda x: 1 if any(token in mi_labels for token in str(x).strip().upper().split()) else 0
)
print("Label distribution:")
print(label_df['AMI'].value_counts())

# Select 41 MI + 164 non-MI samples (paper setup)
mi_df = label_df[label_df['AMI'] == 1]
non_mi_df = label_df[label_df['AMI'] == 0]
test_df = pd.concat([
    mi_df.sample(n=41, random_state=42),
    non_mi_df.sample(n=164, random_state=42)
])
record_set = set(test_df['FileName'].astype(str).str.strip().str.upper())

# %% Cell 4: Define preprocessing
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=100, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    padlen = 3 * max(len(a), len(b))
    if signal.shape[1] <= padlen:
        signal = np.pad(signal, ((0, 0), (0, padlen - signal.shape[1] + 1)), mode='edge')
    return filtfilt(b, a, signal, axis=1)

def downsample(signal, original_fs=500, target_fs=100):
    num_samples = int(signal.shape[1] * target_fs / original_fs)
    return resample(signal, num_samples, axis=1)

def load_ecg_csv(filename):
    df = pd.read_csv(filename)
    if df.shape[1] != 12:
        raise ValueError(f"Expected 12 leads, got {df.shape[1]} in {filename}")
    signal = df.to_numpy().T
    signal = downsample(signal, original_fs=500, target_fs=100)
    signal = signal[:, :1000] if signal.shape[1] >= 1000 else np.pad(signal, ((0, 0), (0, 1000 - signal.shape[1])), mode='edge')
    signal = bandpass_filter(signal, fs=100)
    signal = signal/1000
    # Apply per-lead min-max normalization
    signal_min = signal.min(axis=1, keepdims=True)
    signal_max = signal.max(axis=1, keepdims=True)
    signal = (signal - signal_min) / (signal_max - signal_min+ 1e-6)
    
    return signal


# %% Cell 5: Load ECG data
X_chapman = []
y_chapman = []
paths = []

ecg_folder = os.path.join(chapman_base_path, "ECGDataDenoised")  # Update folder name if needed

for file in os.listdir(ecg_folder):
    if not file.endswith(".csv"):
        continue
    base_id = os.path.splitext(file)[0].strip().upper()
    if base_id not in record_set:
        continue
    try:
        full_path = os.path.join(ecg_folder, file)
        signal = load_ecg_csv(full_path)
        X_chapman.append(signal)
        y_chapman.append(test_df.loc[test_df['FileName'].str.upper() == base_id, 'AMI'].values[0])
        paths.append(full_path)
    except Exception as e:
        print(f"Error loading {file}: {e}")

X_chapman = np.array(X_chapman)

X_chapman = np.expand_dims(X_chapman, -1)
y_chapman = np.array(y_chapman).astype(int)
print("Loaded ECGs:", X_chapman.shape)
print("Class distribution:", np.bincount(y_chapman))

plt.plot(X_chapman[0,1,:,:])
# %% Cell 6: Load trained model
model = load_model(model_path)
print("Model loaded.")

# %% Cell 7: Predict and evaluate
y_pred_prob = model.predict(X_chapman)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_chapman, y_pred))
print("AUC Score:", roc_auc_score(y_chapman, y_pred_prob))

# %% Cell 8: Confusion Matrix and ROC
cm = confusion_matrix(y_chapman, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-MI', 'MI'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Chapman Test Set")
plt.show()

fpr, tpr, _ = roc_curve(y_chapman, y_pred_prob)
plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Chapman")
plt.grid(True)
plt.legend()
plt.show()
# %%

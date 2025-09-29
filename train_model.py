# train_model.py
import pandas as pd
import numpy as np
import pywt
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ===============================
# CONFIG
# ===============================
FILE_PATH = "final.csv"        # Path to dataset
MODEL_PATH = "bearing_fault_model.h5"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
SEQ_LENGTH = 10
EPOCHS = 20
BATCH_SIZE = 32

# ===============================
# HELPER FUNCTIONS
# ===============================
def wavelet_transform(signal, wavelet="db4", level=3):
    """Extract wavelet features (mean, std) for each decomposition level."""
    signal = np.array(signal).reshape(-1)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    for coeff in coeffs:
        features.append(np.mean(coeff))
        features.append(np.std(coeff))
    return np.array(features)

def create_sequences(data, labels, seq_length=10):
    """Create sliding window sequences for LSTM input."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(labels[i + seq_length])
    return np.array(X), np.array(y)

# ===============================
# LOAD & PREPROCESS DATA
# ===============================
print("📥 Loading dataset...")
df = pd.read_csv(FILE_PATH)

# Ensure Time_Col is numeric
df["Time_Col"] = pd.to_numeric(df["Time_Col"], errors="coerce")
df.dropna(inplace=True)

# Encode output labels
label_encoder = LabelEncoder()
df["Output"] = label_encoder.fit_transform(df["Output"])

# Save label encoder
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)
print("✅ Label encoder saved.")

# Wavelet feature extraction
print("🔎 Extracting wavelet features...")
wavelet_features = []
for _, row in df.iterrows():
    x_feat = wavelet_transform(row["X"])
    y_feat = wavelet_transform(row["Y"])
    z_feat = wavelet_transform(row["Z"])
    wavelet_features.append(np.concatenate([x_feat, y_feat, z_feat]))

wavelet_df = pd.DataFrame(wavelet_features)
wavelet_df["Output"] = df["Output"].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(wavelet_df.drop(columns=["Output"]))

# Save scaler
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved.")

# ===============================
# CREATE SEQUENCES
# ===============================
X, y = create_sequences(X_scaled, wavelet_df["Output"].values, SEQ_LENGTH)
print(f"📊 Sequence data shape: {X.shape}, Labels shape: {y.shape}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# DEFINE LSTM MODEL
# ===============================
print("🧠 Building LSTM model...")
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(len(np.unique(y)), activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# ===============================
# TRAIN MODEL
# ===============================
print("🚀 Training model...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)

# Save trained model
model.save(MODEL_PATH)
print(f"✅ Model trained and saved at {MODEL_PATH}")
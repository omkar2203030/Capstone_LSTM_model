from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import pywt
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

# Load the model and preprocessors
model = load_model("bearing_fault_model (1).h5")
with open("scaler (1).pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder (1).pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Wavelet feature extraction
def wavelet_transform(signal, wavelet='db4', level=3):
    signal = np.array(signal).reshape(-1)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    for coeff in coeffs:
        features.append(np.mean(coeff))
        features.append(np.std(coeff))
    return np.array(features)

# Sequence builder for LSTM
def create_sequences(data, seq_length=10):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
    return np.array(X)

# FFT plot as base64 image
def compute_fft_plot(signal, label="X", sampling_rate=20):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/sampling_rate)
    fft_vals = np.abs(np.fft.rfft(signal))

    plt.figure(figsize=(6, 4))
    plt.plot(freqs, fft_vals, label=f"{label}-Axis")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(f"FFT of {label} Axis")
    plt.grid(True)
    plt.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_fault():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    df = pd.read_csv(file)

    if not all(col in df.columns for col in ["X", "Y", "Z", "Time_Col"]):
        return jsonify({"error": "CSV must contain X, Y, Z, Time_Col columns"}), 400

    df["Time_Col"] = pd.to_numeric(df["Time_Col"], errors="coerce")
    df.dropna(inplace=True)

    # FFT Plots
    x_fft = compute_fft_plot(df["X"].values[:256], "X")
    y_fft = compute_fft_plot(df["Y"].values[:256], "Y")
    z_fft = compute_fft_plot(df["Z"].values[:256], "Z")

    # Wavelet Feature Extraction
    wavelet_features = []
    for _, row in df.iterrows():
        x_feat = wavelet_transform(row["X"])
        y_feat = wavelet_transform(row["Y"])
        z_feat = wavelet_transform(row["Z"])
        wavelet_features.append(np.concatenate([x_feat, y_feat, z_feat]))

    wavelet_df = pd.DataFrame(wavelet_features)
    X_scaled = scaler.transform(wavelet_df)

    SEQ_LENGTH = 10
    X_seq = create_sequences(X_scaled, SEQ_LENGTH)

    if len(X_seq) == 0:
        return jsonify({"error": "Insufficient data for sequence creation"}), 400

    predictions = model.predict(X_seq)
    predicted_labels = np.argmax(predictions, axis=1)
    decoded_predictions = label_encoder.inverse_transform(predicted_labels)

    most_common = pd.Series(decoded_predictions).mode()[0]

    return jsonify({
        "predictions": decoded_predictions.tolist(),
        "most_common": most_common,
        "fft_images": {
            "x": x_fft,
            "y": y_fft,
            "z": z_fft
        }
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 4001))
    app.run(host="0.0.0.0", port=port, debug=True)




























# from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS
# import os
# import pandas as pd
# import numpy as np
# import pywt
# import pickle
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import io
# import base64
# from tensorflow.keras.models import load_model

# app = Flask(__name__)
# CORS(app)

# # Load model and preprocessors
# model = load_model("bearing_fault_model (1).h5")
# with open("scaler (1).pkl", "rb") as f:
#     scaler = pickle.load(f)
# with open("label_encoder (1).pkl", "rb") as f:
#     label_encoder = pickle.load(f)

# # === Wavelet feature extraction for 1D signal ===
# def wavelet_transform(signal, wavelet='db4', level=3):
#     signal = np.array(signal).astype(float).reshape(-1)
#     coeffs = pywt.wavedec(signal, wavelet, level=level)
#     features = []
#     for coeff in coeffs:
#         features.append(np.mean(coeff))
#         features.append(np.std(coeff))
#     return np.array(features)

# # === Create sequences for LSTM input ===
# def create_sequences(data, seq_length=10):
#     X = []
#     for i in range(len(data) - seq_length + 1):
#         X.append(data[i:i + seq_length])
#     return np.array(X)

# # === FFT Plot (returns base64) ===
# def compute_fft_plot(signal, label="X", sampling_rate=20):
#     n = len(signal)
#     freqs = np.fft.rfftfreq(n, d=1/sampling_rate)
#     fft_vals = np.abs(np.fft.rfft(signal))

#     plt.figure(figsize=(6, 4))
#     plt.plot(freqs, fft_vals, label=f"{label}-Axis")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Amplitude")
#     plt.title(f"FFT of {label} Axis")
#     plt.grid(True)
#     plt.legend()
#     buf = io.BytesIO()
#     plt.tight_layout()
#     plt.savefig(buf, format="png")
#     plt.close()
#     buf.seek(0)
#     return base64.b64encode(buf.getvalue()).decode("utf-8")

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict_fault():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     df = pd.read_csv(file)

#     if not all(col in df.columns for col in ["X", "Y", "Z", "Time_Col"]):
#         return jsonify({"error": "CSV must contain X, Y, Z, Time_Col columns"}), 400

#     df["Time_Col"] = pd.to_numeric(df["Time_Col"], errors="coerce")
#     df.dropna(inplace=True)

#     # FFT Plots from the first window
#     x_fft = compute_fft_plot(df["X"].values[:256], "X")
#     y_fft = compute_fft_plot(df["Y"].values[:256], "Y")
#     z_fft = compute_fft_plot(df["Z"].values[:256], "Z")

#     # Sliding window feature extraction
#     window_size = 256
#     stride = 128
#     features = []

#     for i in range(0, len(df) - window_size + 1, stride):
#         x_window = df["X"].values[i:i+window_size]
#         y_window = df["Y"].values[i:i+window_size]
#         z_window = df["Z"].values[i:i+window_size]

#         x_feat = wavelet_transform(x_window)
#         y_feat = wavelet_transform(y_window)
#         z_feat = wavelet_transform(z_window)

#         combined = np.concatenate([x_feat, y_feat, z_feat])
#         features.append(combined)

#     if len(features) < 10:
#         return jsonify({"error": "Insufficient data for sequence creation"}), 400

#     # Scale and sequence
#     X_scaled = scaler.transform(features)
#     SEQ_LENGTH = 10
#     X_seq = create_sequences(X_scaled, SEQ_LENGTH)

#     # Prediction
#     predictions = model.predict(X_seq)
#     predicted_labels = np.argmax(predictions, axis=1)
#     decoded_predictions = label_encoder.inverse_transform(predicted_labels)
#     most_common = pd.Series(decoded_predictions).mode()[0]

#     return jsonify({
#         "predictions": decoded_predictions.tolist(),
#         "most_common": most_common,
#         "fft_images": {
#             "x": x_fft,
#             "y": y_fft,
#             "z": z_fft
#         }
#     })

# if __name__ == "__main__":
#     port = int(os.getenv("PORT", 5000))
#     app.run(host="0.0.0.0", port=port, debug=True)

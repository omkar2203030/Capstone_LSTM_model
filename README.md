# Capstone_LSTM_model


# 🛠️ Bearing Fault Detection using LSTM & Wavelet Transform

A **Bearing Fault Detection System** that uses **Wavelet Transform** for feature extraction and an **LSTM (Long Short-Term Memory)** model to classify bearing health conditions. The system also provides **FFT (Fast Fourier Transform) visualizations** for vibration signals.

## 📂 Dataset

* **Source:** HUST Bearing Dataset (Huazhong University of Science and Technology)
* **Sample file:** `final.csv`
* **Features:**

  * `Time_Col` → Timestamp (numeric)
  * `X, Y, Z` → Vibration signals along three axes
  * `Output` → Bearing condition label (`Healthy`, `Inner Fault`, `Outer Fault`, `Ball Fault`, etc.)


## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/omkar2203030/Capstone_LSTM_model.git
cd bearing-fault-detection

# Install dependencies
pip install flask flask-cors pywavelets tensorflow scikit-learn pandas numpy matplotlib python-dotenv
```

---

## 🏋️ Model Training

Run the training script:

```bash
python train_model.py
```

This will:

* Preprocess dataset
* Extract wavelet features
* Create sequences for LSTM
* Train the LSTM model

Generated files:

* `bearing_fault_model.h5` → Trained LSTM model
* `scaler.pkl` → StandardScaler for normalization
* `label_encoder.pkl` → Encoder for output labels


## 🚀 Running the Flask App

Start the server:

```bash
python app.py
```

* App runs at → `http://127.0.0.1:4001/`
* Upload `.csv` file via frontend (`index.html`)


## 🌐 Web Interface

* Upload `.csv` with columns: `Time_Col, X, Y, Z`
* Get **predictions + most common fault type**
* View **FFT plots** for X, Y, Z axes


## 📊 Example API Response

```json
{
  "predictions": ["Healthy", "Healthy", "Outer Fault", "Outer Fault"],
  "most_common": "Outer Fault",
  "fft_images": {
    "x": "base64string...",
    "y": "base64string...",
    "z": "base64string..."
  }
}
```


## 📁 Project Structure

```
├── app.py                  # Flask backend
├── train_model.py          # Model training script
├── templates/
│   └── index.html          # Frontend UI
├── bearing_fault_model.h5  # Trained LSTM model
├── scaler.pkl              # Scaler for features
├── label_encoder.pkl       # Output label encoder
├── final.csv               # Preprocessed HUST dataset sample
└── README.md               # Project documentation
```


## 🧑‍💻 Technologies Used

* **Python** → Data processing + ML
* **TensorFlow / Keras** → LSTM model
* **PyWavelets** → Wavelet Transform feature extraction
* **Matplotlib** → FFT plots
* **Flask + flask-cors** → Backend API
* **HTML/JS** → Frontend UI

## 📌 Future Improvements

* ✅ Integrate real-time sensor data streaming
* ✅ Docker / Cloud deployment
* ✅ More advanced wavelet families & features
* ✅ Better frontend (React.js, TailwindCSS)




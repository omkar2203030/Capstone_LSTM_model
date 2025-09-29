# Capstone_LSTM_model


🛠️ Bearing Fault Detection using LSTM & Wavelet Transform

This project is a Bearing Fault Detection System that uses Wavelet Transform for feature extraction and an LSTM (Long Short-Term Memory) deep learning model to classify bearing health conditions. The system also provides FFT (Fast Fourier Transform) visualizations for vibration signals.

It includes:

🧾 Preprocessing & Model Training (Python + TensorFlow)

🌐 Flask Backend for inference & serving predictions

💻 Frontend (HTML/JS) to upload CSV files and visualize results

📊 Visualization of FFT plots for X, Y, Z vibration signals

📂 Dataset

We use the HUST Bearing Dataset (Huazhong University of Science and Technology), which contains vibration signals for bearings under different working conditions and fault types.

Features in dataset (final.csv format):

Time_Col → Time stamp (numeric)

X, Y, Z → Vibration signal values along three axes

Output → Bearing condition label (e.g., Healthy, Inner Fault, Outer Fault, Ball Fault, etc.)

⚙️ Installation

Clone the repository and install dependencies:

git clone https://github.com/yourusername/bearing-fault-detection.git
cd bearing-fault-detection

Install required Python packages:

pip install flask flask-cors pywavelets tensorflow scikit-learn pandas numpy matplotlib python-dotenv

🏋️ Model Training

Run the training script to:

Preprocess the dataset

Extract wavelet features

Create sequences for LSTM

Train the LSTM model

Save the trained model + scaler + label encoder


python train_model.py

This will generate:

bearing_fault_model.h5 → Trained LSTM model

scaler.pkl → StandardScaler for normalization

label_encoder.pkl → Encoder for output labels

🚀 Running the Flask App

Start the Flask server:

python app.py

The app will run on http://127.0.0.1:5000/

🌐 Web Interface

Open the frontend (index.html) in a browser

Upload a .csv file with vibration data (X, Y, Z, Time_Col)

Get predictions + most common fault type

FFT plots of X, Y, Z axes will be displayed

📊 Example Prediction Output

{
  "predictions": ["Healthy", "Healthy", "Outer Fault", "Outer Fault"],
  "most_common": "Outer Fault",
  "fft_images": {
    "x": "base64string...",
    "y": "base64string...",
    "z": "base64string..."
  }
}

📁 Project Structure

├── app.py                  # Flask backend
├── train_model.py          # Model training script
├── templates/
│   └── index.html          # Frontend
├── bearing_fault_model.h5  # Trained LSTM model
├── scaler.pkl              # Scaler for features
├── label_encoder.pkl       # Output label encoder
├── final.csv               # Preprocessed HUST dataset sample
└── README.md               # Project documentation

🧑‍💻 Technologies Used

Python (Data Processing + ML)

TensorFlow/Keras (LSTM model)

PyWavelets (Wavelet Transform feature extraction)

Matplotlib (FFT plots)

Flask (Backend API)

HTML/JS (Frontend UI)

📌 Future Improvements

Integrate with real-time sensor data streaming

Deploy on Docker / Cloud for scalability

Add more advanced wavelet families and feature extraction methods

Improve frontend UI with React.js or TailwindCSS


---

Do you want me to also write the train_model.py file separately (clean version of the first part of your script) so that your project folder has a clear separation between training and deployment?

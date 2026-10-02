# ⚡ AI-Based Smart Energy Consumption Prediction System

A laptop-only AI + IoT simulation project that predicts electricity consumption and detects unusual energy usage using simulated sensor data and Machine Learning.

## 🚀 Project Overview
This project simulates a smart energy monitoring system without physical IoT hardware. Virtual sensors generate voltage, current, power factor, temperature, load, and time features. Machine Learning predicts next-hour energy consumption and detects unusual usage patterns.

### System Flow
Virtual Sensors → Synthetic Energy Data → Data Preprocessing → ML Models → Prediction + Anomaly Detection → Streamlit Dashboard

## ✨ Features
- Virtual IoT energy sensor simulation
- Synthetic energy-consumption dataset
- Random Forest regression for energy prediction
- Isolation Forest for anomaly detection
- Real-time simulated readings
- Interactive Streamlit dashboard
- Energy trend charts and alerts
- Model evaluation metrics

## 🛠️ Tech Stack
Python | NumPy | Pandas | Scikit-learn | Joblib | Streamlit | Plotly | Matplotlib

## 📁 Project Structure
```text
ai-smart-energy-prediction/
├── app.py
├── generate_dataset.py
├── train_model.py
├── sensor_simulator.py
├── predict.py
├── requirements.txt
├── .gitignore
├── LICENSE
├── README.md
├── data/
│   ├── .gitkeep
│   └── sample_energy_data.csv
├── models/
│   └── .gitkeep
└── reports/
    └── .gitkeep
```

## ▶️ Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/ai-smart-energy-prediction.git
cd ai-smart-energy-prediction
python -m venv .venv
```
Windows: `.venv\Scripts\activate`
macOS/Linux: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
python predict.py
python sensor_simulator.py
streamlit run app.py
```

## 📊 Machine Learning
**RandomForestRegressor** predicts energy consumption in kWh for the next hour. **IsolationForest** identifies unusual energy-consumption patterns without requiring anomaly labels.

## ⚠️ Important Limitation
This project uses synthetic/simulated sensor data for educational and portfolio purposes. Predictions are not validated for real electrical systems and should not be used for operational energy-management decisions.

## 🔮 Future Scope
ESP32/Raspberry Pi integration, real voltage/current sensors, MQTT, smart-meter integration, cloud storage, appliance-level monitoring, time-series forecasting, LSTM/Transformer models, and mobile notifications.

## 👨‍💻 Author
**Nazmath Pasha**

## 📄 License
MIT License

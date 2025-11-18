# Eco-Driving Analysis & Driving Style Classification

This project analyzes real driving datasets to detect **eco-driving patterns**, **sudden driving events**, and **driver behavior types** using machine learning and time-series feature extraction.

It processes OBD-II–based vehicle signals, computes adaptive thresholds, extracts sliding-window features, performs clustering, and generates a full dashboard visualization.

---

## 🚗 Features

### ✅ Data Processing
- Loads 3 datasets: **traffic jam**, **normal traffic**, **free-flow traffic**.
- Cleans timestamps and converts them to seconds.
- Forward-fills missing sensor values.

### ⚙️ Adaptive Event Detection
Uses dataset-derived percentiles to compute thresholds:
- **Sudden acceleration** based on accelerator pedal rate (% / sec)
- **Sudden braking** based on speed deceleration (m/s²)

### 📊 Sliding Window Feature Extraction
For each 15-second window:
- RPM mean & variance  
- Speed mean & variance  
- Throttle & accelerator pedal stats  
- Mass Air Flow (MAF)  
- Engine load  
- Sudden braking / acceleration events  
- Time-aware acceleration & deceleration  
- Eco-driving proxy indicator

### 🤖 Driving Style Classification
Uses KMeans (k=3):
- Computes cluster statistics
- Maps clusters to **calm**, **normal**, **aggressive** using a weighted scoring system

### 🌱 Eco-Driving Detection
- Computes a fuel efficiency indicator  
- Marks top 33% as eco-friendly windows

### 📈 Dashboard Visualization
Generates 3 subplots:
1. **Driving style distribution per traffic state**
2. **Sudden events by traffic state & style**
3. **Eco-driving rate by traffic state & style**

---

## 📂 Project Structure

├── Dataset/
│ ├── 2018-03-29_Seat_Leon_KA_RT_Stau.csv
│ ├── 2018-04-23_Seat_Leon_RT_KA_Normal.csv
│ └── 2018-04-23_Seat_Leon_KA_KA_Frei.csv
├── eco_driving_analysis.py # (your script)
├── requirements.txt
└── README.md

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

2. Run the script
python main.py

3. View the dashboard
A Matplotlib window will open showing the 3 analysis charts.

🧪 Requirements
See requirements.txt for all Python dependencies.

📝 Notes
Works with any car OBD-II dataset if the same column names exist.
Thresholds automatically adapt to dataset behavior.
Window size and overlap are configurable.


# 🚗 Driving Style Analysis System

A comprehensive machine learning system that analyzes OBD-II vehicle data to classify driving styles into **calm**, **normal**, and **aggressive** categories using deep learning.

---

## 📋 Table of Contents

* Overview
* Features
* Installation
* Quick Start
* Dataset
* Model Architecture
* Usage
* Results
* Testing New Data
* Troubleshooting
* Project Structure
* Contributing
* License

---

## 🎯 Overview

This system uses **PyTorch** to build a neural network that analyzes real-time vehicle data from OBD-II sensors to classify driving behavior. The model processes time-series data through sliding windows and extracts comprehensive features to make accurate predictions.

### 🔑 Key Applications

* Insurance telematics
* Fleet management
* Driver safety monitoring
* Eco-driving feedback
* Driving behavior research

---

## ✨ Features

* 🤖 **Deep Learning Model**: Enhanced neural network with batch normalization and dropout
* 📊 **Comprehensive Analysis**: Multi-feature engineering from OBD-II data
* 🕒 **Time-Series Processing**: Sliding window approach for temporal patterns
* 📈 **Multiple Outputs**: Driving style classification + confidence scores
* 🔍 **Detailed Reporting**: Comprehensive analysis with visualizations
* 💾 **Model Persistence**: Multiple save formats for compatibility
* 🎯 **High Accuracy**: ~99% accuracy on test data

---

## 🚀 Installation

### ✅ Prerequisites

* Python 3.8+
* 8GB RAM (16GB recommended)
* 2GB free storage

### Step 1: Clone / Download the Project

```bash
# If using git
git clone <repository-url>
cd driving-style-analysis
```

Or download and extract the project files manually.

### Step 2: Set Up Environment

#### Using pip (Recommended)

```bash
pip install -r requirements.txt
```

#### Using conda

```bash
conda create -n driving_analysis python=3.9
conda activate driving_analysis
pip install -r requirements.txt
```

#### Using virtual environment

```bash
python -m venv driving_env
source driving_env/bin/activate  # Linux/Mac
# OR
driving_env\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python verify_installation.py
```

---

## 🏃 Quick Start

### 1. Prepare Your Data

Place your OBD-II dataset in the following structure:

```
Dataset/
└── OBD-II-Dataset.zip
    ├── 2018-03-29_Seat_Leon_KA_RT_Stau.csv
    ├── 2018-04-23_Seat_Leon_RT_KA_Normal.csv
    └── 2018-04-23_Seat_Leon_KA_KA_Frei.csv
```

### 2. Run the Training

```bash
python IA_Forum.ipynb
```

Or if running as script:

```bash
python -c "from IA_Forum import main; main()"
```

### 3. Test with New Data

```bash
python test_new_data.py
```

---

## 📊 Dataset

### Required CSV Format

Your CSV files must contain the following columns:

| Column                              | Description        | Unit     |
| ----------------------------------- | ------------------ | -------- |
| Time                                | Timestamp          | HH:MM:SS |
| Vehicle Speed Sensor                | Vehicle speed      | km/h     |
| Engine RPM                          | Engine revolutions | RPM      |
| Accelerator Pedal Position D        | Pedal position     | %        |
| Absolute Throttle Position          | Throttle position  | %        |
| Air Flow Rate from Mass Flow Sensor | Air flow rate      | g/s      |

### Sample Structure

```csv
Time,Vehicle Speed Sensor [km/h],Engine RPM [RPM],Accelerator Pedal Position D [%],Absolute Throttle Position [%],Air Flow Rate from Mass Flow Sensor [g/s]
08:30:15,45.2,2345,32.1,28.5,24.3
08:30:16,46.1,2389,33.2,29.1,25.0
```

---

## 🧠 Model Architecture

### Neural Network Structure

```
Input Layer (27 features)
    ↓
Hidden Layer 1 (256) + BatchNorm + ReLU + Dropout
    ↓
Hidden Layer 2 (128) + BatchNorm + ReLU + Dropout
    ↓
Hidden Layer 3 (64) + BatchNorm + ReLU + Dropout
    ↓
Hidden Layer 4 (32) + BatchNorm + ReLU + Dropout
    ↓
Output Layer (3) → [calm, normal, aggressive]
```

### Feature Engineering (27 Features)

* Statistical metrics: mean, std, min, max
* Temporal patterns: rolling statistics
* Event detection: sudden acceleration & braking
* Engine load indicators
* Fuel efficiency metrics
* Entropy-based features

---

## 📖 Usage

### Train the Model

```python
from IA_Forum import main
main()
```

### Test New Driving Data

```python
from test_new_data import DrivingStyleTester

tester = DrivingStyleTester()
results, stats = tester.test_single_driving_session(
    csv_file_path="path/to/your/data.csv",
    traffic_state="normal_traffic"  # traffic_jam | normal_traffic | traffic_free
)
```

### Quick Test

```python
from test_new_data import quick_test
results = quick_test("path/to/your/data.csv")
```

---

## 📈 Results

### Performance

* ✅ Test Accuracy: ~99%
* Precision: 98–100%
* Recall: 83–100%
* F1-Score: 91–99%

### Output Includes

* Driving style distribution
* Confidence scores
* Safety assessment
* Behavioral metrics
* Visual analytics

---

## 🔍 Testing New Data

The analysis provides:

* Basic statistics (confidence, duration, windows)
* Speed & RPM analysis
* Aggressive event ratio
* Style breakdown per session
* Safety level + recommendations
* CSV & JSON reports

### Example Output

```
COMPREHENSIVE DRIVING STYLE ANALYSIS REPORT
============================================

📊 BASIC STATISTICS:
   • Total time windows analyzed: 417
   • Average prediction confidence: 0.954
   • Data duration: 3120.5 seconds

🚗 DRIVING BEHAVIOR METRICS:
   • Average speed: 45.2 km/h
   • Average RPM: 2345
   • Total sudden events: 127
   • Aggressive driving ratio: 8.9%

🎯 DRIVING STYLE DISTRIBUTION:
   • calm: 156 windows (37.4%)
   • normal: 224 windows (53.7%)
   • aggressive: 37 windows (8.9%)

🏆 DOMINANT DRIVING STYLE: NORMAL
🛡️ SAFETY ASSESSMENT: ✅ GOOD - Mostly calm and normal driving
```

---

## 🛠️ Troubleshooting

### Common Issues

1. Missing Dependencies

```bash
pip install --force-reinstall -r requirements.txt
```

2. PyTorch Installation Issues

```bash
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

3. File Not Found Errors

* Verify dataset location
* Check CSV structure
* Validate paths

4. Memory Issues

* Reduce batch size
* Use smaller window size
* Close other programs

### Verification

```bash
python verify_installation.py
```

---

## 📁 Project Structure

```
driving-style-analysis/
├── IA_Forum.ipynb
├── test_new_data.py
├── quick_test.py
├── requirements.txt
├── verify_installation.py
├── Dataset/
│   └── OBD-II-Dataset.zip
├── driving_model_state_dict.pth
├── safe_complete_model.pth
├── complete_driving_model.pth
├── feature_scaler.json
├── label_encoder.json
└── driving_analysis_report.csv
```

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if needed
5. Submit a pull request

### Development Setup

```bash
pip install -r requirements_dev.txt
python -m pytest tests/
black .
```

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🆘 Support

If you encounter issues:

* Check Troubleshooting section
* Run `verify_installation.py`
* Confirm your data format
* Review console error logs

---

## 🎊 Acknowledgments

* OBD-II dataset providers
* PyTorch team
* Scikit-learn contributors

---

✅ **This README is ready to copy & paste directly into your GitHub repository.**

# 🛡️ CyberKavach-AI  
**AI-Powered Cyber Fraud & Phishing Detection System**

CyberKavach-AI is an intelligent cybersecurity application that detects **fraudulent URLs and scam messages** using a hybrid approach of **Machine Learning, rule-based heuristics, and external threat-intelligence APIs**.  
The system is designed for **real-time analysis** with a professional Streamlit web interface.

---

## 🚀 Key Features

### 🔗 URL Fraud Detection
- Machine-learning based URL classification  
- URL structural feature extraction  
- Probability-calibrated fraud scoring  
- Rule-based risk analysis (IP URLs, suspicious keywords, deep subdomains, etc.)

### 💬 Message / SMS Scam Detection
- NLP-based text preprocessing  
- TF-IDF vectorization  
- SVM-based spam & scam detection  

### 🌐 External Security Checks (Optional)
- VirusTotal integration  
- Google Safe Browsing integration  
- Trusted domain whitelisting  

### 📊 Analytics & Logging
- Scan history stored in CSV  
- Fraud vs Safe statistics  
- Adjustable ML confidence threshold  
- Lightweight model option for faster predictions  

---

## 🧠 Tech Stack

- **Python 3.10+**
- **Streamlit** – Web Interface  
- **Scikit-learn** – Machine Learning  
- **Pandas & NumPy** – Data processing  
- **Pickle** – Model storage  
- **VirusTotal & Google Safe Browsing APIs**

---

## 📁 Project Structure

cyberkavach-AI/
│
├── app.py # Main Streamlit application
├── train_model.py # Model training pipeline
│
├── model/
│ ├── url_model.pkl
│ ├── url_model_light.pkl
│ ├── text_model.pkl
│ ├── vectorizer.pkl
│
├── dataset/
│ ├── messages.csv
│ ├── urls_train.csv
│ ├── verified_online.csv
│ ├── negatives_seed.csv
│ ├── forced_negatives.txt
│ └── scan_log.csv
│
├── dataset/utils/
│ ├── url_features.py
│ ├── url_normalize.py
│ ├── url_rules.py
│ └── text_clean.py
│
├── requirements.txt
└── README.md


---

## ⚙️ Installation & Setup

 Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows

 Install Dependencies
pip install -r requirements.txt
🧪 Train the Models (Optional)
python train_model.py
This will:

Train message and URL detection models

Calibrate probability scores

Save trained models to the model/ directory

▶️ Run the Application
streamlit run app.py
Open in browser:

http://localhost:8501
🔑 API Keys (Optional)
You can enable external security checks using API keys.

VirusTotal
export VT_API_KEY="your_api_key_here"
Google Safe Browsing
export GSB_API_KEY="your_api_key_here"
You may also enter API keys directly from the Streamlit sidebar.

⚠️ Never commit API keys to GitHub

📈 Machine Learning Overview
Text Model: TF-IDF + Support Vector Machine (SVM)

URL Model: Logistic Regression / Gradient Boosting

Calibration: CalibratedClassifierCV

Evaluation Metrics: Brier Score, ROC-AUC

Hybrid Decision: ML score + rules + reputation signals

🛡️ Disclaimer
This project is intended for educational and research purposes only.
It should not be used as the sole system for financial, legal, or security-critical decisions.

🤝 Contributing
Contributions are welcome!

Fork the repository

Create a new branch

Commit your changes

Open a Pull Request

📄 License
This project is licensed under the MIT License.

👤 Author
Nilesh choudhary 
Cybersecurity & AI Enthusiast
Focus: AI Security & Cloud Security

GitHub: https://github.com/Nile5h

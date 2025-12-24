# 🩺 Diabetes Prediction using PIMA Indians Dataset

🔗 **Live Demo:** https://diabetes-prediction-kozs.onrender.com/  
📦 **Repository:** https://github.com/krishnagupta2107/diabetes_prediction

---

## 📌 Project Overview

This project presents an **end-to-end Diabetes Prediction System** developed using **Machine Learning** and deployed as a **Flask-based web application**.  
It predicts whether a person is **Diabetic or Non-Diabetic** based on clinical and physiological health parameters.

The system is built using the **PIMA Indians Diabetes Dataset** and demonstrates the complete ML lifecycle:  
data preprocessing → exploratory data analysis → model training & evaluation → deployment.

---

## 🎯 Problem Statement

Diabetes is a chronic disease that often remains undiagnosed until severe complications occur.  
Traditional diagnostic methods can be time-consuming and inaccessible for early screening.

This project aims to build an **automated, reliable, and easy-to-use system** that:
- Predicts diabetes risk using basic health indicators
- Handles missing or noisy medical data
- Provides real-time predictions via a web interface
- Supports early screening and awareness

---

## 🎯 Objectives

- Analyze and preprocess the PIMA Indians Diabetes Dataset
- Handle missing and zero-valued medical attributes
- Train and evaluate multiple ML classification models
- Compare models using standard evaluation metrics
- Deploy the best-performing model using Flask
- Provide a simple and user-friendly web interface

---

## 🧠 Machine Learning Methodology

### Dataset
- **PIMA Indians Diabetes Dataset**
- Features include:
  - Pregnancies
  - Glucose Level
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age

### Data Preprocessing
- Zero values treated as missing (Insulin, BMI, Skin Thickness, etc.)
- Feature scaling using **Standard Scaler**
- Train-test split (80/20)

### Models Trained
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

The best-performing model was selected based on evaluation metrics.

---

## 📊 Model Performance

| Metric | Score |
|------|------|
| Accuracy | 70.78% |
| Precision | 60.00% |
| Recall | 50.00% |
| F1-Score | 54.55% |

- Recall was prioritized to reduce missed diabetic cases.
- Confusion matrix was used for deeper analysis.

---

## 🌐 Web Application & Deployment

- **Backend:** Flask
- **Frontend:** HTML, CSS
- **Model Integration:** Pickle/Joblib
- **Deployment Platform:** Render

🔗 **Live Application:**  
👉 https://diabetes-prediction-kozs.onrender.com/

⏱️ **Prediction Time:** ~40–100 ms  
⚡ Smooth real-time performance

---

## 🛠️ Tech Stack

- Python
- NumPy, Pandas
- scikit-learn
- Flask
- HTML, CSS
- Render (Deployment)

---

## ⚙️ Installation & Setup (Local)

```bash
git clone https://github.com/krishnagupta2107/diabetes_prediction.git
cd diabetes_prediction
pip install -r requirements.txt
python app.py
```

Open in browser:  
http://127.0.0.1:5000/

---

## 🚀 Future Enhancements

- Improve accuracy using advanced models (XGBoost, ANN)
- Add larger and more diverse datasets
- Store user prediction history
- Enhance UI with visual analytics
- Deploy at scale for real-world screening

---

## 👨‍💻 Author

**Krishna Gupta**  
B.Tech CSE (AIML)  
GLA University, Mathura  
GitHub: https://github.com/krishnagupta2107

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only** and should not be used as a substitute for professional medical diagnosis.

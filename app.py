from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open('model/diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        input_data = pd.DataFrame({
            'Pregnancies': [float(data['pregnancies'])],
            'Glucose': [float(data['glucose'])],
            'BloodPressure': [float(data['bloodPressure'])],
            'SkinThickness': [float(data['skinThickness'])],
            'Insulin': [float(data['insulin'])],
            'BMI': [float(data['bmi'])],
            'DiabetesPedigreeFunction': [float(data['dpf'])],
            'Age': [float(data['age'])]
        })
        
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)
        
        result = {
            'prediction': int(prediction[0]),
            'probability_non_diabetic': float(probability[0][0] * 100),
            'probability_diabetic': float(probability[0][1] * 100)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Cargar modelos
with open('../models/regression_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)

with open('../models/classification_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = reg_model.predict(features)
    return jsonify({'prediction': float(prediction[0])})

@app.route('/predict_classification', methods=['POST'])
def predict_classification():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    proba = clf_model.predict_proba(features)[0][1]
    return jsonify({'probability': float(proba)})

if __name__ == '__main__':
    app.run(debug=True)
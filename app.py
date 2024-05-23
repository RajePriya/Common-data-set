import warnings
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
with open('model/best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the label encoders
with open('model/label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Load the scaler
with open('model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Crop Production Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Encode categorical features
    for feature in ['State_Name', 'District_Name', 'Season', 'Crop']:
        if feature in data:
            data[feature] = label_encoders[feature].transform([data[feature]])[0]
    
    # Prepare input data
    input_data = np.array([[data['State_Name'], data['District_Name'], data['Crop_Year'], data['Season'], data['Crop'], data['Area']]])
    input_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return the prediction as JSON response
    return jsonify({'predicted_production': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

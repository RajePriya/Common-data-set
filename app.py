from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model, label encoders, and scaler
with open('model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

import pickle

# Open the file in read-binary mode
with open('model/scaler.pkl', 'rb') as f:
    # Load the scaler object
    scaler = pickle.load(f)

# Now you can use the scaler object

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = [data.get(col) for col in ['State_Name', 'District_Name', 'Season', 'Crop', 'Area', 'Crop_Year']]
        
        # Encode categorical features
        for i, col in enumerate(['State_Name', 'District_Name', 'Season', 'Crop']):
            features[i] = label_encoders[col].transform([features[i]])[0]

        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)  # Scale the features
        prediction = model.predict(features)

        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
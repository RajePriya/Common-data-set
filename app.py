from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask("__name__")

# Load the pre-trained model and label encoders
model = joblib.load('random_forest_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define the /predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json()

        # Extract features from the request data
        area = data['Area']
        season = label_encoders['Season'].transform([data['Season']])[0]
        crop = label_encoders['Crop'].transform([data['Crop']])[0]
        yield_per_area = data['Yield_Per_Area']

        # Create a feature array
        features = np.array([[area, season, crop, yield_per_area]])

        # Make prediction
        prediction = model.predict(features)

        # Return the prediction as JSON
        return jsonify({'Production': prediction[0]})
    except KeyError as e:
        return jsonify({'error': f'Missing key in request data: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '_main_':
    app.run(debug=True)
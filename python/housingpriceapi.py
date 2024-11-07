# Import necessary libraries
import pandas as pd
from flask import Flask, request, jsonify
import pickle
import locale

# Create a Flask app
app = Flask(__name__)

# Load the trained model
with open('housing_price_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Define the correct order of features
feature_order = [
    'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
    'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
]

# Set a fixed exchange rate from USD to NOK (example rate)
usd_to_nok_rate = 10.95  # Oppdater dette med den nyeste valutakursen

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()
    print("Received data:", data)  # Logg dataene for feilsøking

    # Convert square meters to square feet for LotArea and GrLivArea
    data['LotArea'] = data['LotArea'] * 10.764
    data['GrLivArea'] = data['GrLivArea'] * 10.764
    data['GarageArea'] = data['GarageArea'] * 10.764

    # Ensure the features are in the correct order
    try:
        input_data = pd.DataFrame([data], columns=feature_order)
        print("Input DataFrame:", input_data)  # Logg DataFrame for å bekrefte korrekt formatering
    except Exception as e:
        print("Error creating DataFrame:", str(e))
        return jsonify({'error': 'Error creating DataFrame'})

    # Predict house price using the loaded model
    try:
        prediction_usd = loaded_model.predict(input_data)[0]
        print("Prediction (USD):", prediction_usd)  # Logg prediksjonen
        # Predict house price using the loaded model with adjustments for price increase and inflation
        prediction_nok = prediction_usd * usd_to_nok_rate * 1.446

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({'error': 'Prediction error'})

    # Format the prediction with thousand separators and decimal points
    formatted_prediction_nok = locale.format_string("%.2f", prediction_nok, grouping=True)

    # Return the prediction as JSON
    return jsonify({'estimated_price_nok': formatted_prediction_nok})


# Define a route for predicting house prices using GET request
@app.route('/predict_get', methods=['GET'])
def predict_get():
    # Get parameters from the URL
    data = {
        'LotArea': float(request.args.get('LotArea')) * 10.764,
        'OverallQual': int(request.args.get('OverallQual')),
        'OverallCond': int(request.args.get('OverallCond')),
        'YearBuilt': int(request.args.get('YearBuilt')),
        'YearRemodAdd': int(request.args.get('YearRemodAdd')),
        'GrLivArea': float(request.args.get('GrLivArea')) * 10.764,
        'FullBath': int(request.args.get('FullBath')),
        'HalfBath': int(request.args.get('HalfBath')),
        'BedroomAbvGr': int(request.args.get('BedroomAbvGr')),
        'KitchenAbvGr': int(request.args.get('KitchenAbvGr')),
        'TotRmsAbvGrd': int(request.args.get('TotRmsAbvGrd')),
        'Fireplaces': int(request.args.get('Fireplaces')),
        'GarageCars': int(request.args.get('GarageCars')),
        'GarageArea': float(request.args.get('GarageArea')) * 10.764
    }

    # Convert data to DataFrame
    try:
        input_data = pd.DataFrame([data], columns=feature_order)
        print("Input DataFrame (GET):", input_data)  # Logg DataFrame for å bekrefte korrekt formatering
    except Exception as e:
        print("Error creating DataFrame (GET):", str(e))
        return jsonify({'error': 'Error creating DataFrame'})

    # Predict house price using the loaded model
    try:
        prediction_usd = loaded_model.predict(input_data)[0]
        print("Prediction (USD) (GET):", prediction_usd)  # Logg prediksjonen
        # Predict house price using the loaded model with adjustments for price increase and inflation
       
        prediction_nok = prediction_usd * usd_to_nok_rate * 1.446

    except Exception as e:
        print("Prediction error (GET):", str(e))
        return jsonify({'error': 'Prediction error'})

   # Format the prediction with thousand separators and decimal points
    formatted_prediction_nok = locale.format_string("%.2f", prediction_nok, grouping=True)

    # Return the prediction as JSON
    return jsonify({'estimated_price_nok': formatted_prediction_nok})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

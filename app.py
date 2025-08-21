import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load preprocessor and model
try:
    preprocessor = joblib.load('crop_yield_preprocessor.pkl')
    # Use a dummy DataFrame to get the input dimension after preprocessing
    dummy_df = pd.DataFrame([{
        'District_Name': 'Bagerhat',
        'Year': 2020,
        'Crop_Name': 'Banana',
        'Rainfall_mm': 150,
        'Sunshine_Hours': 6,
        'Avg_Temperature_Min (°C)': 22,
        'Avg_Temperature_Max (°C)': 31,
        'Humadity_%': 80
    }])
    input_dim = preprocessor.transform(dummy_df).shape[1]
except FileNotFoundError:
    print("Preprocessor file not found. Please ensure 'crop_yield_preprocessor.pkl' exists.")
    preprocessor = None
    input_dim = 1 # A default value to allow the app to start, but prediction will fail.
except Exception as e:
    print(f"Error loading preprocessor: {e}")
    preprocessor = None
    input_dim = 1

# Define the Transformer model (same as training)
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.dropout(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Initialize and load model
if preprocessor:
    model = TransformerRegressor(input_dim)
    try:
        model.load_state_dict(torch.load('crop_yield_transformer.pth', map_location=torch.device('cpu')))
        model.eval()
    except FileNotFoundError:
        print("Model file not found. Please ensure 'crop_yield_transformer.pth' exists.")
        model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    model = None

# Available options for dropdowns
districts = ['Bagerhat', 'Bandarban', 'Barguna', 'Barishal', 'Bhola', 'Bogura', 'Brahmanbaria', 'Chandpur', 'Chapai Nawabganj', 'Chattogram', 
'Chuadanga', 'Cox’s Bazar', 'Cumilla', 'Dhaka', 'Dinajpur', 'Faridpur', 'Feni', 'Gaibandha', 'Gazipur', 'Gopalganj', 
'Habiganj', 'Jamalpur', 'Jashore', 'Jhallokati', 'Jhenaidah', 'Joypurhat', 'Khagrachari', 'Khulna', 'Kishoreganj', 'Kurigram', 
'Kushtia', 'Lakshmipur', 'Lalmonirhat', 'Madaripur', 'Magura', 'Manikganj', 'Meherpur', 'Moulvibazar', 'Munshiganj', 'Mymensingh', 
'Naogaon', 'Narail', 'Narayanganj', 'Narsingdi', 'Natore', 'Netrokona', 'Nilphamari', 'Noakhali', 'Pabna', 'Panchagar', 
'Patuakhali', 'Pirojpur', 'Rajbari', 'Rajshahi', 'Rangamati', 'Rangpur', 'Satkhira', 'Shariatpur', 'Sherpur', 'Sirajganj', 
'Sunamganj', 'Sylhet', 'Tangail', 'Thakurgaon']

crops = ['Turmeric', 'Tomato', 'Ginger', 'Sugarcane', 'Soyabean', 'Garlic', 'Motor Dal', 
'Khesari Dal', "Lady's Finger", 'Mug Dal', 'Brinjal', 'Masur Dal', 'Chilli', 
'Onion', 'Groundnut', 'Sweet Potato', 'Gram', 'Maize', 'Tea', 'Cucumber', 
'Pumpkin', 'Banana']

@app.route('/')
def index():
    return render_template('index.html', districts=districts, crops=crops, years=list(range(2018, 2024)))

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not preprocessor:
        return jsonify({
            'success': False,
            'error': 'Model or preprocessor not loaded. Please check the server logs.'
        })
        
    try:
        # Get form data
        data = request.form
        
        # Create DataFrame with the same structure as training data
        input_df = pd.DataFrame([{
            'District_Name': data['district'].strip(),
            'Year': int(data['year']),
            'Crop_Name': data['crop'].strip(),
            'Rainfall_mm': float(data['rainfall']),
            'Sunshine_Hours': float(data['sunshine']),
            'Avg_Temperature_Min (°C)': float(data['temp_min']),
            'Avg_Temperature_Max (°C)': float(data['temp_max']),
            'Humadity_%': float(data['humidity'])
        }])
        
        # Preprocess the input
        processed_input = preprocessor.transform(input_df)
        
        # Convert to tensor
        input_tensor = torch.tensor(
            processed_input.toarray() if hasattr(processed_input, 'toarray') else processed_input, 
            dtype=torch.float32
        )
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).item()
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'crop': data['crop'],
            'district': data['district'],
            'year': data['year']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
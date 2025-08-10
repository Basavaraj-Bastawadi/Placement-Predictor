import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load the scaler used in training
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import pandas as pd

# Fit scaler on the original data (required for correct scaling)
df = pd.read_csv("placement.csv")
df = df.iloc[:, 1:]  # Remove first column if needed
X = df.iloc[:, 0:2]
scaler.fit(X)

@app.route('/')
def home():
    return render_template('index.html')  # You need to create an index.html file

@app.route('/predict', methods=['POST'])
def predict():
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])
    features = np.array([[cgpa, iq]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    result = 'Placed' if prediction[0] == 1 else 'Not Placed'
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def home():
    return "Breast Cancer Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['features']
        data = np.array(data).reshape(1, -1)

        prediction = model.predict(data)[0]

        result = "Malignant" if prediction == 0 else "Benign"

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)

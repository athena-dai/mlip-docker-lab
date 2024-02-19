from flask import Flask
import numpy as np
import joblib

app = Flask(__name__)

from flask import request

@app.route('/predict', methods=['GET'])
def predict():
    # Get the input array from the request
    get_json = request.get_json()
    iris_input = get_json['input']
    
    # TODO: Import trained model
    # model = ...
    with open('iris_model.pkl', 'rb') as f:
        model = joblib.load(f)
    
    # TODO: Make prediction using the model 
    # HINT: use np.array().reshape(1, -1) to convert input to 2D array
    # prediction = ...
    prediction = model.predict(np.array(iris_input).reshape(1, -1))
    
    # TODO: Return the prediction as a response
    # return ...
    return str(prediction[0])

@app.route('/')
def hello():
    return 'Welcome to Docker Lab'

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')

from flask import Flask, jsonify, request
import joblib

model = joblib.load('model.joblib')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # get the image from request
    image = request.json['image']

    # make prediction
    prediction = model.predict([image])

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run()
import pickle
from flask import Flask
from flask import request
from flask import jsonify

#code which helps initialize our server
app = Flask(__name__)

# load the models and scaler
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

#defining a /predict route for only post requests
@app.route('/predict', methods=['POST'])
def wine():
    req_body = request.form.to_dict()
    features = [[float(feature) for feature in req_body['features'].split(',')]]
    # print(features)
    scaled_features = scaler.transform(features)
    # print(scaled_features)
    prediction = model.predict(scaled_features).tolist()
    # print(prediction)

    return jsonify({'predictions': prediction})

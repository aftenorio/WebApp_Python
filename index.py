# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 21:18:28 2020

@author: Yohana Delgado Ramos
"""
import pickle
import flask
import os

app = flask.Flask(__name__)
port = int(os.getenv("PORT", 8095))

model = pickle.load(open("linearmodel.pkl","rb"))


@app.route('/predict', methods=['POST'])
def predict():

    features = flask.request.get_json(force=True)['features']
    prediction = model.predict([features])[0,0]
    response = {'prediction': prediction}

    return flask.jsonify(response)

if __name__ == '__main__':
    app.run(host='localhost', port=port)
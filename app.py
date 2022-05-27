# -*- coding: utf-8 -*-
"""
Created on Thu May 26 18:36:41 2022

@author: Raghu
"""

# flask app
# importing libraries
import numpy as np
from collections.abc import Mapping
from flask import Flask, request, jsonify, render_template
import pickle

from markupsafe import escape
# flask app
app = Flask(__name__)
# loading model
model = pickle.load(open('modelbest.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction',methods = ['POST'])
def prediction():
    final_features = [float(x) for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)
    
    if prediction == 'Y':
       output ='Sample delivered on time'
    else:
       output ='Sample NOT delivered on time'       
    
    return render_template('index.html', output = output)

if __name__ == '__main__':
    app.run(debug=True)
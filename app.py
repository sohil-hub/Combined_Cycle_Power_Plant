from os import name
from flask import Flask, render_template, request
import sklearn
import joblib
import requests
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

app = Flask(__name__)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        AT         = float(request.form['AT'])
        V          = float(request.form['V'])
        AP         = float(request.form['AP'])
        RH         = float(request.form['RH'])
       
 
        input = np.array([AT, V, AP, RH])
        print(input)

        scaler = joblib.load('scaler')
        model = tf.keras.models.load_model('regressor.h5')
        prediction = float(model.predict(scaler.transform([input]))[0][0])
        
        return render_template('index.html',prediction_text="Energy generated is {}".format(prediction))
    
    else:
        return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)


#France 100, spain 001, germany 010
#male 1 female 0
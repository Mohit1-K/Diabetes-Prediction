from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler
app = Flask(__name__, template_folder='template')
model = joblib.load("lr5_model.pkl")
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        insulin = float(request.form['insulin'])    
        bmi = int(request.form['BMI'])
        Pedigree_function = int(request.form['Diabetes Pedigree Function'])
        Age = int(request.form['Age'])
        
        prediction=model.predict([[Pregnancies,Glucose,insulin,bmi,Pedigree_function,Age]])
        output=(prediction[0])
        if output<0:
            return render_template('index.html',prediction_texts="You have lower chances of having Diabetes")
        else:
            return render_template('index.html',prediction_text="You have higher chances of having Diabetes")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
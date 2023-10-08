from flask import Flask, request,render_template

from src.pipeline.predict_pipeline import CustomData,PredictPipline

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET","POST"])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    
    else:
        data=CustomData(
        age=request.form.get('age'),
        gender=request.form.get('gender'),
        hypertension=request.form.get('hypertension'),
        heart_disease=request.form.get('heart_disease'),
        smoking_history=request.form.get('smoking_history'),
        bmi=request.form.get('bmi'),
        HbA1c_level=request.form.get('HbA1c_level'),
        blood_glucose_level=request.form.get('blood_glucose_level'),
        
        )
        
        pred_df=data.get_data_as_data_frame()
        

        predict_pipline=PredictPipline()
    
        result=predict_pipline.predict(pred_df)

        return render_template('home.html',results=result[0])
    
if __name__=='__main__':
    app.run(host="0.0.0.0",debug=True,port=5000)
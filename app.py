from flask import Flask , render_template, request
import pickle

import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictionPipeline

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/home",methods=['GET','POST'])
def home():
    if request.method == 'GET':
        return render_template("home.html")
    else:
        data = CustomData(
                    gender=request.form.get('gender',""),
                    race_ethnicity=request.form.get('race_ethnicity',""),  # ✅ FIXED
                    parental_level_of_education=request.form.get('parental_level_of_education',""),
                    lunch=request.form.get('lunch',""),
                    test_preparation_course=request.form.get('test_preparation_course',""),
                    reading_score=float(request.form.get('reading_score', 0)),   # ✅ swapped back
                    writing_score=float(request.form.get('writing_score',0))    # ✅ swapped back
                )
        
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predicted_pipeline = PredictionPipeline()
        results = predicted_pipeline.predict(pred_df)
        return render_template("home.html",result=results[0])


if __name__ == "__main__":

    app.run(port=5000, debug=True)

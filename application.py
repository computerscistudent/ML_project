from flask import Flask, render_template, request, url_for, redirect
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline
from sklearn.preprocessing import StandardScaler
import traceback  # ✅ Added for logging errors

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/home", methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template("home.html")
    else:
        try:
            # Collect form data
            data = CustomData(
                gender=request.form.get('gender', ""),
                race_ethnicity=request.form.get('race_ethnicity', ""),  
                parental_level_of_education=request.form.get('parental_level_of_education', ""),
                lunch=request.form.get('lunch', ""),
                test_preparation_course=request.form.get('test_preparation_course', ""),
                reading_score=float(request.form.get('reading_score', 0)),  
                writing_score=float(request.form.get('writing_score', 0))    
            )
            
            # Convert to DataFrame
            pred_df = data.get_data_as_dataframe()
            print("✅ Input DataFrame:\n", pred_df)

            # Predict
            predicted_pipeline = PredictionPipeline()
            results = predicted_pipeline.predict(pred_df)
            results = np.clip(results, 0, 100)
            
            return redirect(url_for("result", result=round(results[0], 2)))

        except Exception as e:
            # Print full Python traceback to logs
            print("❌ ERROR in /home POST route:")
            traceback.print_exc()
            return "Something went wrong! Check logs for details."

@app.route('/result/<float:result>', methods=['GET', 'POST'])
def result(result):
    res = ''
    if result >= 90:
        res = f'You are pass with awesome {result} Marks'
    elif result < 90 and result >= 70:
        res = f'Pass with {result} Marks, keep going and improving!!'
    elif result < 70 and result >= 45:
        res = f'You are pass with ordinary {result} Marks, practice hard'
    else:
        res = f'You are fail with unremarkable {result} Marks'    
    return render_template('result.html', result=res)   

if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")

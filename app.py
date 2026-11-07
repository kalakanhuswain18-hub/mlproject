
from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)

application = Flask(__name__)
app = application


# Home page
@app.route('/')
def index():
    return render_template('index.html')


# Student prediction page
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')

    try:

        # Collect form data
        data = CustomData(
            gender=request.form.get('gender'),

            race_ethnicity=request.form.get('ethnicity'),

            parental_level_of_education=request.form.get(
                'parental_level_of_education'
            ),

            lunch=request.form.get('lunch'),

            test_preparation_course=request.form.get(
                'test_preparation_course'
            ),

            reading_score=float(
                request.form.get('reading_score')
            ),

            writing_score=float(
                request.form.get('writing_score')
            )
        )

        # Convert data into DataFrame
        pred_df = data.get_data_as_data_frame()

        print("Input DataFrame:")
        print(pred_df)

        print("Input Columns:")
        print(pred_df.columns.tolist())

        # Prediction
        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)

        prediction = round(float(results[0]), 2)

        print("Predicted Maths Score:", prediction)

        return render_template(
            'home.html',
            results=prediction
        )

    except Exception as e:

        print("Prediction Error:", e)

        return render_template(
            'home.html',
            error=str(e)
        )


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

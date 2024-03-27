from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained RandomForestClassifier model
with open('model/RandomForestClassifier_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    age = int(request.form['age'])
    income = float(request.form['income'])
    loan_amount = float(request.form['loanAmount'])
    credit_score = int(request.form['creditScore'])
    months_employed = int(request.form['monthsEmployed'])
    num_credit_lines = int(request.form['numCreditLines'])
    interest_rate = float(request.form['interestRate'])
    loan_term = int(request.form['loanTerm'])
    dti_ratio = float(request.form['dtiRatio'])

    # Create a numpy array with the input features in the same order as your model expects
    features = np.array([[age, income, loan_amount, credit_score, months_employed, num_credit_lines,
                          interest_rate, loan_term, dti_ratio]])

    # Make prediction using the loaded model
    prediction = model.predict(features)[0]

    # Map prediction result to human-readable output
    prediction_text = 'Will default' if prediction == 1 else 'Will not default'

    return render_template('prediction.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)


import pickle

from flask import Flask
from flask import request
from flask import jsonify


# Loading the model
model_file = "model_C=0.1.bin"
with open(model_file, "rb") as f_in:
    (dv, model) = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods = ['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


# customer
# {    
#      "gender": "female",
#      "seniorcitizen": 1,
#      "partner": "no",
#      "dependents": "yes",
#      "phoneservice": "yes",
#      "multiplelines": "yes",
#      "internetservice": "fiber_optic",
#      "onlinesecurity": "yes",
#      "onlinebackup": "no",
#      "deviceprotection": "yes",
#      "techsupport": "yes",
#      "streamingtv": "yes",
#      "streamingmovies": "yes",
#      "contract": "month-to-month",
#      "paperlessbilling": "no",
#      "paymentmethod": "electronic_check",
#      "tenure": 6,
#      "monthlycharges": 124.2,
#      "totalcharges": 1003.5
# }
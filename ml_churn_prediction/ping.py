from flask import Flask

app = Flask('churn')

@app.route('/predict', methods = ['POST'])
def predict(customer):    
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9696)
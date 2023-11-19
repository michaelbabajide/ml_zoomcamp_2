# Churn Prediction Project

This project is a churn prediction model for a telecom company. It uses customer data to predict whether a customer is likely to stop using the company's service, a process known as churn. The model is trained on a dataset containing customer information such as gender, senior citizen status, partner status, dependents, phone service, multiple lines, internet service, online security, online backup, device protection, tech support, streaming TV, streaming movies, contract, paperless billing, payment method, tenure, monthly charges, and total charges.

## Project Structure

The project consists of three main Python scripts:

1. `train.py`: This script is responsible for training the churn prediction model. It first configures logging, then loads and preprocesses the data. The data is split into training, validation, and test sets. The script then trains a logistic regression model using K-Fold cross-validation, and finally saves the trained model and its associated data vectorizer.

2. `predict.py`: This script is a Flask application that loads the trained model and uses it to make churn predictions. It exposes a single endpoint, `/predict`, which accepts POST requests with customer data in JSON format and returns the predicted churn probability and a boolean indicating whether the customer is predicted to churn.

3. `test.py`: This script is used to test the prediction service. It sends a POST request to the `/predict` endpoint with sample customer data and prints the response.

The project also includes a Dockerfile for building a Docker image of the prediction service. The Docker image is based on Python 3.8.12 and uses Pip for dependency management. The prediction service runs on Gunicorn and listens on port 9696.

## How to Run

To run the prediction service, build the Docker image and run a container:

```bash
docker build -t churn-prediction .
docker run -p 9696:9696 churn-prediction
```

Then, send a POST request to `http://localhost:9696/predict` with customer data in JSON format. For example:

```bash
curl -X POST -H "Content-Type: application/json" -d @customer.json http://localhost:9696/predict
```

Where `customer.json` is a file containing customer data in the following format: (you can make changes to the data to see how it affects the prediction)

```json
{
    "gender": "female",
    "seniorcitizen": 1,
    "partner": "no",
    "dependents": "yes",
    "phoneservice": "yes",
    "multiplelines": "yes",
    "internetservice": "fiber_optic",
    "onlinesecurity": "yes",
    "onlinebackup": "no",
    "deviceprotection": "yes",
    "techsupport": "yes",
    "streamingtv": "yes",
    "streamingmovies": "yes",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 6,
    "monthlycharges": 124.2,
    "totalcharges": 1003.5
}
```

The service will return the predicted churn probability and a boolean indicating whether the customer is predicted to churn:

```json
{
    "churn_probability": 0.32940789808151005,
    "churn": false
}
```
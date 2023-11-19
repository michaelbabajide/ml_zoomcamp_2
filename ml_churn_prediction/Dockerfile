FROM python:3.8.12

# RUN pip install pipenv

WORKDIR /app
# COPY ["Pipfile", "Pipfile.lock", "./"]
COPY ["requirements.txt", "./"]

# RUN pipenv install --system --deploy
RUN pip install --trusted-host pypi.org -r requirements.txt

COPY ["predict.py", "model_C=0.1.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
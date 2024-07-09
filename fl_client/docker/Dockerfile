FROM python:3.11
COPY ../requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install torchtuples pycox statsmodels
WORKDIR /app
COPY ../src/client.py src/client.py
CMD ["python", "src/client.py"]
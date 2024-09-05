FROM python:3.11-buster

COPY e2e/docker_full/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY common/ /common
COPY e2e/docker_full/test.py ./
ENV PYTHONPATH="/common"

CMD ["python", "test.py"]
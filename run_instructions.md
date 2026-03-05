## Custom run instructions for local execution (no docker)
Note that all commands that follow are asumed to be executed while initially on the **root directory** of the project.
1. Clean previously created virtual environments (if any)
```
cd scripts
bash clean.sh
```

2. Create virtual environments
```
cd scripts
bash generate_envs.sh
```

3. Start MLFlow
```
docker run -d --rm -p 10000:5000 ghcr.io/mlflow/mlflow:v2.16.2 mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```
After this step MLFLow web UI should be accessible from http://localhost:10000

4. Start RabbitMQ
```
docker run -d --rm -p 31871:5672 -p 31872:15672 rabbitmq:3.13.3-management
```

5. Start the flower superlink
```
cd apps/fl_client
source .venv/bin/activate
flower-superlink --insecure --driver-api-address 127.0.0.1:9091 --fleet-api-address 127.0.0.1:9092
```

6. Start FL clients
For each client we want to start, excute the following in a **new terminal**.
```
export PYTHONPATH=$(pwd)/common:$PYTHONPATH from fl_components
export $(grep -v '^#' .env | xargs)
cd apps/fl_client
export PYTHONPATH=$(pwd)/src:$PYTHONPATH from fl_client
source .venv/bin/activate
flower-supernode --insecure src.fl_client.main:app --superlink 127.0.0.1:9092
```

7. Create an exepriment in MLFlow
Visit MLFlow's web UI at http://localhost:10000 and create a new experiment under the "Experiments" tab.

8. Upload the model to MLFlow
```
export MYPYPATH=$(pwd)
export PYTHONPATH=$(pwd)
export $(grep -v '^#' .env | xargs) # This manually sources the file in the terminal
cd common
source .venv/bin/activate
python -m fl_models.iris.fl_model
```
At this step some dependencies may missing. Install them manually using pip.
More specifcially `cloudpickle`, `mlflow` and `flwr` were missing.
After a while you should see the model registered under the "Models" tab in MLFlow's web UI.

9. Start the FL server
In a new terminal execute the following to start the FL server.
```
export PYTHONPATH=$(pwd)/common:$PYTHONPATH  from fl_components
export $(grep -v '^#' .env | xargs)
cd apps/fl_server
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
source .venv/bin/activate
python -m src.fl_server.main --superlink 127.0.0.1:9091
```
A this step `structlog` and `flwr`

10. Start the REST API
Important: **comment out any code related to RabbitMQ and RabbitMQ setup** as the current dockerfile setup is unable to copy relevant code from the `common` directory. 
First **copy `common/schemas` directory under `src/restapi`** in order to make import work in the restapi code.

Excute the following commands:
```
cd restapi
docker build -t restapi .
docker run --name fl-restapi-container -d -p 8000:8000 restapi
```
Then you should be able to see the swagger in the address http://localhost:8000

11. Submit a training task
Via the REST API, submit a tasl using the POST `/tasks` endpoint. Make sure to use an experiment name existing in the MLFlow.

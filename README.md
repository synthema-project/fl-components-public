# Federated Learning Components

## Description
This project hosts the code for the three components dealing with the FL workflow in Synthema:
- The FL restapi is the main entrypoint for the system. It will allow users to manage tasks.
- The FL server is the coordinator of the FL cluster.
- The FL client is the entity deployed at the edge where data chunks live.

### Structure
The componentes are structured as a monorepo to harmonise common utilities and to pair the versions for all FL restapi, FL server and FL client.

- The folder common provides general utilities, datasets and models.
- The folder apps includes the various applications mentioned.
    - The folder fl_client is dedicated to the component FL client of the architecture.
    - The folder fl_server is dedicated to the component FL server of the architecture.
    - The folder restapi is dedicated to the REST API to allow for human interaction with the system.

## Contributing
Pull requests are welcome. Please make sure to update tests as appropriate.

### Prerequisites
Extending the code requires a set of tools to be present in your machine:

1. Poetry
1. A running mlflow server -> check the repo for [mlflow](https://github.com/synthema-project/model-experiment_registries) for a local mlflow deployment on K8s or use docker container directly through the command
    ```bash
    docker run -p 5000:5000 ghcr.io/mlflow/mlflow
    ```
1. A running instance of rabbitmq -> see the [docker-compose.yml](docker-compose.yml) file.

### Setting up the environment

There are several configuration options that need to be established in order to start contributing.

The process is as follows:

1. Clone the repository
1. Copy the .env from the templates folder with `cp templates/dot_env .env` and sync it with your IDE. (In vscode this is done automatically if the file is named ".env"). Note that the path appended to both PYTHONPATH and MYPYPATH must be the root dir of this project.
1. Run the script [generate_envs.sh](scripts/generate_envs.sh) to create all dedicated venvs. **Note that the script generate_envs.sh install runs `poetry install` and creates a virtual environment by default in the home directory, if you want to set the project dir as the parent folder for the venv, then make sure to [config poetry](https://python-poetry.org/docs/configuration/#virtualenvsin-project).**


### Running tests and code checks
The main entrypoint for testing purposes is the file [tests.sh](scripts/tests.sh), which perform local tests. You can also use the script [images_test.sh](scripts/images_test.sh) to perform tests on top of docker.


### Bulding the images
To build the images, the [script](scripts/images_build.sh) can be used as follows.

```bash
bash images_build.sh build <image_tag> <docker_registry>
```

### Deploying all apps
There are 2 main options for deploying the apps. Running them in docker, and running them in kubernetes.

#### Docker deployment
You can deploy the apps in docker as follows.

1. Open the [docker-compose.yml](docker-compose.yml) file and set the appropriate env variables for each component.
1. Run `docker compose up`to start the FL components and their dependencies.
1. Go to [localhost:8000/docs](localhost:8000/docs) in the browser, which is the [OpenAPI](https://www.openapis.org/) spec for the REST API.

#### Kubernetes deployment
You can deploy the apps in kubernetes seamlessly with [helm](https://helm.sh/). To do that, you can create a custom values.yml file following what is done in this [file](k8s/dev-values.yaml). Then you can use the following command.

```bash
helm install fl-components ./fl-chart -f <your_values.yaml>
```

### Running a federated learning job
To run an end to end example, the easiest option is to run the preloaded model targeting a classification task on the iris dataset.

1. Make sure that all apps and dependencies are deployed.
1. Go to common/fl_models/iris/fl_model.py and run it to upload a federated model into mlflow and copy the model details (name and version) returned. Bear in mind that you should change the url in there.
1. Open the [docs](localhost:8000/docs) while we finalise the UI.
1. Run the FL task by using the model attributes (name and version) returned when the model was uploaded. Set the use case to iris as it is the demo use case.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

This project extends and uses the following Open Softwares, which are compliant with MIT License:
- FastAPI (MIT License)
- fastapi-cli (MIT License)
- Jinja (MIT License)
- email_validator  (MIT License)
- pydantic  (MIT License)
- python-multipart (MIT License)
- pytest (MIT License)
- Uvicorn (MIT License)
- starlette  (MIT License)
- httpx (MIT License)
- setuptools (MIT License)
- sqlmodel (MIT License)
- pika (MIT License)
- Flower (Apache 2.0 License)
- MLFlow (Apache 2.0 License)
- boto3 (Apache 2.0 License)
- Numpy (BSD License)
- Pandas (BSD License)
- torch (BSD License)
- psycopg2-binary (PostgreSQL License)
- typing-extensions (PSF License)
- Psutil (PSL License)
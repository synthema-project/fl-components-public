# Federated Learning Componentes

## Description

This project hosts the code for the two components dealing with the FL workflow in Synthema.

### Structure

The componentes are structured as a monorepo to harmonise common utilities and to pair the versions for both the FL-server and the FL-client.

- The folder common provides general utilities, datasets and models.
- The folder fl_client is dedicated to the component FL client of the architecture.
- The folder fl_server is dedicated to the component FL server of the architecture.

### Notice

Current code status is likely to change in the future, this is not intended to be used as a prototype but rather to start any implementation like SMPC and DP.

## Contributing

Pull requests are welcome. Please make sure to update tests as appropriate.

### Prerequisites

Extending the code requires a set of tools to be present in your machine:

1. Poetry
1. Make
1. A running mlflow server -> check the docs within [synthema](https://github.com/synthema-project/model-experiment_registries) for a local mlflow deployment on K8s or use docker container directly through the command

```bash
docker run -p 5000:5000 mlflow/mlflow
```

### Setting up the environment

There are several configuration options that need to be established in order to start contributing.

The process is as follows:

1. Clone the repository
1. Copy the .env from the templates folder with `cp templates/dot_env .env` and sync it with your IDE. (In vscode this is done automatically if the file is named ".env")
1. Open the file "pyproject.toml" and look for the [tool.pytest_env] section -> adapt this section at will or use environmental variables for your tests
1. Run the command `make install` to install all dependencies using poetry


### Running tests and code checks

There are several make commands to run different checks. Note that all must pass before any code is commited.

```bash
make lint-fix
make format
make type-check
make test
```

### Running a preloaded federated learning job

To run an end to end example, the easiest option is to run the preloaded model targeting a classification task on the iris dataset.

1. Go to common/fl_models/iris/fl_model.py and run it to upload a federated model into mlflow. Bear in mind that you should change the url in there.
1. Go to fl_server/src/main.py and modify the commands load_model and require_load_model to target the version returned in the console with the previous step.
1. Open a console to run the command `make run-superlink`
1. Open a console and run the following commands
    ```bash
    export $(grep -v '^#' .env | xargs)
    make run-fl-client
    ```
1. Open a console and run the following commands
    ```bash
    export $(grep -v '^#' .env | xargs)
    make run-fl-sever
    ```

Note that not all environmental variables are necessary in the case of the fl-server.

## License

<!-- [MIT](https://choosealicense.com/licenses/mit/) -->
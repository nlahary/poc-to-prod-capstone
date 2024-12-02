<!-- Load images from images/ -->
<p align="center">
  <img src="images/airflow.png" alt="Airflow Logo" width="300">
  <img src="images/tensorflow.png" alt="TensorFlow Logo" width="100" style="margin-left: 10px; margin-right: 10px;">
  <img src="images/docker.png" alt="Docker Logo" width="450">
</p>


The project is a simple example of a ML model deployment using FastAPI, Airflow and Docker. The model predicts the tag of a StackOverflow question, using BERT embeddings and a simple neural network with a single hidden layer. 

The project primarily focuses on the deployment aspect, and the model is a simple example to demonstrate the deployment process.

The project has the following Python packages:
- `train`: Contains the model, the configuration file, the training data, a directory to save artefacts.
- `predict`: Contains the API to serve the model.
- `preprocessing`: Contains the classes to embed the text data.
- `config`: Contains a Settings class from Pydantic to handle the configurations using the `.dev.env` file.

The three first packages contains a `tests` directory with unit tests, which can be run using `pytest`.

A single DAG in the `airflow/dags` directory is used to train a new model and save it.

To start all services (Airflow + API):

```bash
docker-compose up
```

The building might take a while depending on your internet connection (Tensorflow + BERT model). 
Once the services are up, you can access the Airflow UI at `http://localhost:8080` and the API at `http://localhost:3000`.


Volumes are used in the docker composer for a development purpose, you can remove them to better fit your needs.

The `.dev.env` file contains the configuration for the API and the pathes for the inputs/outputs of the model.

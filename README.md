The project is a simple example of a ML model deployment using FastAPI, Airflow and Docker. The model predicts the tag of a StackOverflow question, using BERT embeddings and a simple Deep Learning model. 

The project primarily focuses on the deployment aspect, and the model is a simple example to demonstrate the deployment process.

The project has the following Python packages:
- `train`: Contains the model, the configuration file, the training data, a directory to save artefacts.
- `predict`: Contains the API to serve the model.
- `preprocessing`: Contains the code to embed the text data.
- `config`: Contains a Settings class from Pydantic to load the configuration file from the `.dev.env` file.

A single DAG in the `airflow/dags` directory is used to train a new model and save it.

To start all services (Airflow + API):

```bash
docker-compose up
```

The building might take a while depending on your internet connection (Tensorflow + BERT model). 
Once the services are up, you can access the Airflow UI at `http://localhost:8080` and the API at `http://localhost:5000`.


Volumes are used for a development purpose, you can remove them to better fit your needs.

The `.dev.env` file contains the configuration for the API, Airflow (not mandatory) and the model.

Additionnal, some Airflow variables are set in the same file. The keys have been generated through websites just for the sake of the example.

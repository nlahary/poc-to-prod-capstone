from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime
import os
import yaml
import logging
from tensorflow.keras.models import load_model
from train.train.run import train
from config import settings


logger = logging.getLogger(__name__)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

######## FUNCTIONS ########


def load_training_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_load_model(artefacts_path):
    return load_model(os.path.join(artefacts_path, 'model.h5'))


def check_model_saved(artefacts_path):
    for file in ['labels_index.json', 'model.h5', 'params.json', 'scores.json', 'train_output.json']:
        if not os.path.exists(os.path.join(artefacts_path, file)):
            raise FileNotFoundError(file)
    logger.info(f'Artefacts successfuly saved in {artefacts_path}')


def train_and_save_artefacts(**kwargs):
    train_config = load_training_config(settings.MODEL_CONFIG_PATH)
    artefacts_path = kwargs['artefacts_path']
    dataset_path = kwargs['dataset_path']
    _, artefacts_path = train(
        dataset_path, train_config, artefacts_path, add_timestamp=True)
    check_model_saved(artefacts_path)
    test_load_model(artefacts_path)


######## DAG ########


dag = DAG(
    'train_and_create_model',
    default_args=default_args,
    description='Train and create a model',
    schedule_interval=None,


)

start = DummyOperator(task_id='start', dag=dag)

train_and_save_artefacts = PythonOperator(
    task_id='train_and_save_artefacts',
    python_callable=train_and_save_artefacts,
    op_kwargs={'artefacts_path': settings.ARTEFACTS_PATH,
               'dataset_path': settings.DATASET_PATH},
    provide_context=True,
    dag=dag,
)

start >> train_and_save_artefacts

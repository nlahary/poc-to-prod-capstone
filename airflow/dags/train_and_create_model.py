from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime
import os
import yaml
import logging
from tensorflow.keras.models import load_model
from train.train.run import train
logger = logging.getLogger(__name__)

ARTEFACTS_PATH = '/opt/airflow/data/artefacts'
DATASET_PATH = '/opt/airflow/train/data/training-data/'
CONFIG_PATH = '/opt/airflow/train/conf/train-conf.yml'

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


def create_output_dir(base_path, add_timestamp=False):
    if add_timestamp:
        output_dir = os.path.join(
            base_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    else:
        output_dir = base_path
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def train_and_save_artefacts(**kwargs):
    train_config = load_training_config(CONFIG_PATH)
    artefacts_path = kwargs['artefacts_path']
    dataset_path = kwargs['dataset_path']
    train(dataset_path, train_config, artefacts_path, add_timestamp=True)


def evaluate_model(artefacts_path, **kwargs):
    model_path = os.path.join(artefacts_path, "model.h5")
    model = load_model(model_path)
    logger.info("Model evaluation completed.")
    return model.evaluate()

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
    op_kwargs={'artefacts_path': ARTEFACTS_PATH, 'dataset_path': DATASET_PATH},
    dag=dag,
)

evaluate_model = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    op_kwargs={'artefacts_path': ARTEFACTS_PATH},
    dag=dag,
)

start >> train_and_save_artefacts >> evaluate_model

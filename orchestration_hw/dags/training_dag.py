from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

from ml_scripts import train_model, evaluate_on_train


with DAG(
    'Training_pipeline',
    description='Train ML model',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2022, 1, 1),
    catchup=False,
    tags=['hw4'],
) as dag:

    t1 = PythonOperator(
        task_id='training',
        python_callable=train_model
    )

    t2 = PythonOperator(
        task_id='train_evaluation',
        python_callable=evaluate_on_train
    )

    t1 >> t2
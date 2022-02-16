from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
import requests

def request_etl():
    response = requests.post('http://pyspark:5000/', json={'task': 'etl'})
    print(response.json())
    if response.status_code != 200 or response.json()['status'] != 'success':
        raise ValueError('ETL job failed')

with DAG(
    'ETL',
    description='ETL process organized with PySpark',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2022, 1, 1),
    catchup=False,
    tags=['hw4'],
) as dag:

    t1 = PythonOperator(
        task_id='etl_request',
        python_callable=request_etl
    )

    t1
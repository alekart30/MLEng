from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

from ml_scripts import infer_predictions, evaluate_on_test


with DAG(
    "Inference_pipeline",
    description="Perform inference for ML model",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2022, 1, 1),
    catchup=False,
    tags=["hw4"],
) as dag:

    t1 = PythonOperator(task_id="inference", python_callable=infer_predictions)

    t2 = PythonOperator(task_id="train_evaluation", python_callable=evaluate_on_test)

    t1 >> t2

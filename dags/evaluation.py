import os
import sys
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


sys.path.append("/opt/airflow")


from src import evaluation as eval


with DAG(
    dag_id="evaluation_DAG",
    start_date=datetime(2026,11,12),
    schedule=None,
    catchup=False
    ) as dag:

    get_predections = PythonOperator(
        task_id="get_model_preds",
        python_callable = eval.get_predections
    )

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable = eval.evaluate_model,
    )

    get_predections >> evaluate_model

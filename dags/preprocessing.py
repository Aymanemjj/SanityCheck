import os
import sys
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


sys.path.append("/opt/airflow")


from src import preprocessing as pre


with DAG(
    dag_id="preprocessing_DAG",
    start_date=datetime(2026,11,12),
    schedule=None,
    catchup=False
    ) as dag:

    check_dataset = PythonOperator(
        task_id = "get_info_about_dataset",
        python_callable = pre.check_data,
    )

    prepare_data = PythonOperator(
        task_id = "prepare_data",
        python_callable = pre.prepare_data
    )


    trigger_train = TriggerDagRunOperator(
        task_id = "trigger_training_dag",
        trigger_dag_id="training_DAG"
    )

    check_dataset >> prepare_data >> trigger_train

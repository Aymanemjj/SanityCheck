import os
import sys
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


sys.path.append("/opt/airflow")

from src import training as train


with DAG(
    dag_id = "training_DAG", 
    start_date=datetime(2026,11,12),
    schedule = None,
    catchup = False
    ) as dag:

    assemble_pipeline = PythonOperator(
        task_id = "assemble_pipeline",
        python_callable = train.assembler
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable = train.train_model,
    )
    
    trigger_evaluation = TriggerDagRunOperator(
        task_id="trigger_evaluation_dag",
        trigger_dag_id="evaluation_DAG"
    )

    assemble_pipeline >> train_model >> trigger_evaluation

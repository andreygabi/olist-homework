"""Weekly CPU training: PostgreSQL -> validation -> preparation -> MLflow."""

from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, get_current_context, task


@dag(
    dag_id='olist_training', schedule='@weekly',
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False, max_active_runs=1, tags=['olist', 'mlflow'],
)
def olist_training():
    @task(task_id='extract', retries=2, retry_delay=timedelta(seconds=30),
          execution_timeout=timedelta(minutes=3))
    def extract_task():
        from olist_pipeline import extract
        return extract(get_current_context()['run_id'])

    @task(task_id='validate', retries=0, trigger_rule='all_success',
          execution_timeout=timedelta(minutes=2))
    def validate_task(snapshot):
        from olist_pipeline import validate
        return validate(snapshot)

    @task(task_id='prepare', retries=0, trigger_rule='all_success', multiple_outputs=False,
          execution_timeout=timedelta(minutes=3))
    def prepare_task(snapshot):
        from olist_pipeline import prepare
        return prepare(snapshot)

    @task(task_id='train', retries=0, trigger_rule='all_success',
          execution_timeout=timedelta(minutes=10))
    def train_task(prepared):
        from olist_pipeline import train
        return train(prepared, dag_run_id=get_current_context()['run_id'])

    train_task(prepare_task(validate_task(extract_task())))


olist_training()

"""PostgreSQL snapshot, validation, time split and MLflow training."""

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from olist_model import (
    FEATURE_COLUMNS, NUMERIC_COLUMNS, fit_model, load_predictor,
    save_predictor, split_by_time,
)


REQUIRED_COLUMNS = ['order_id', 'order_purchase_timestamp', *FEATURE_COLUMNS, 'late']


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path):
    return pd.read_csv(path, encoding='utf-8', dtype={'order_id': 'string'}, float_precision='round_trip')


def _write_csv(frame, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_csv(temporary, index=False, encoding='utf-8', lineterminator='\n',
                     float_format='%.17g', compression=None)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def extract(run_key):
    """Read orders into an atomic, run-specific UTF-8 CSV snapshot."""
    from sqlalchemy import create_engine, text

    if not isinstance(run_key, str) or not run_key.strip():
        raise ValueError('run_key must be a nonempty Airflow run ID')
    workdir = Path(os.environ.get('OLIST_WORKDIR', '/opt/olist/work')).expanduser().resolve()
    run_directory = workdir / 'runs' / hashlib.sha256(run_key.encode('utf-8')).hexdigest()
    snapshot = run_directory / 'snapshot.csv'
    engine = create_engine(os.environ['DATABASE_URL'], pool_pre_ping=True,
                           connect_args={'connect_timeout': 10})
    query = text('SELECT ' + ', '.join(REQUIRED_COLUMNS)
                 + ' FROM public.orders ORDER BY order_purchase_timestamp, order_id')
    try:
        with engine.connect() as connection:
            frame = pd.read_sql_query(query, connection)
    finally:
        engine.dispose()
    _write_csv(frame, snapshot)
    return str(snapshot)


def validate(snapshot):
    """Read-only validation: no filesystem writes, network or database access."""
    path = Path(snapshot).expanduser().resolve(strict=True)
    with path.open('r', encoding='utf-8', newline='') as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError('CSV must have a nonempty header with unique columns')
    missing = sorted(set(REQUIRED_COLUMNS) - set(header))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    frame = _read_csv(path)
    if frame.empty:
        raise ValueError('Snapshot must contain at least one order')
    ids = frame['order_id']
    if ids.isna().any() or ids.str.strip().eq('').any() or not ids.is_unique:
        raise ValueError('order_id must be nonempty and unique')
    try:
        numbers = frame[NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
        labels = frame['late'].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('Numeric features and late must be numbers') from exc
    if not np.isfinite(numbers).all():
        raise ValueError('Numeric features must be finite, without NaN/Inf')
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('late must contain finite 0/1 labels')
    try:
        dates = pd.to_datetime(frame['order_purchase_timestamp'], format='ISO8601', errors='raise', utc=True)
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid order_purchase_timestamp') from exc
    if dates.isna().any():
        raise ValueError('Purchase timestamps must not be missing')
    return str(path)


def prepare(snapshot):
    """Use the supplied temporal split and preserve the raw snapshot hash."""
    snapshot = Path(validate(snapshot))
    digest = _sha256(snapshot)
    cutoff = os.environ.get('OLIST_CUTOFF', '2017-07-31')
    train_frame, validation_frame = split_by_time(_read_csv(snapshot), cutoff=cutoff)
    directory = snapshot.parent / 'prepared'
    train_path, validation_path = directory / 'train.csv', directory / 'validation.csv'
    _write_csv(train_frame, train_path)
    _write_csv(validation_frame, validation_path)
    return {
        'snapshot': str(snapshot), 'snapshot_sha256': digest, 'cutoff': cutoff,
        'train': str(train_path), 'validation': str(validation_path),
        'train_rows': len(train_frame), 'validation_rows': len(validation_frame),
    }


def train(prepared, *, dag_run_id, epochs=None, seed=None):
    """Fit the supplied model and log a new, traceable MLflow run."""
    import mlflow

    if not isinstance(dag_run_id, str) or not dag_run_id:
        raise ValueError('dag_run_id must be the actual Airflow run_id')
    epochs = int(os.environ.get('OLIST_EPOCHS', '5')) if epochs is None else epochs
    seed = int(os.environ.get('OLIST_SEED', '42')) if seed is None else seed
    if type(epochs) is not int or epochs < 1:
        raise ValueError('epochs must be a positive integer')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    lr = float(os.environ.get('OLIST_LR', '0.003'))
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError('lr must be positive and finite')
    if _sha256(prepared['snapshot']) != prepared['snapshot_sha256']:
        raise ValueError('Source snapshot changed after preparation')
    train_frame = _read_csv(prepared['train'])
    validation_frame = _read_csv(prepared['validation'])
    if len(train_frame) != prepared['train_rows'] or len(validation_frame) != prepared['validation_rows']:
        raise ValueError('Prepared row counts changed')
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(os.environ.get('MLFLOW_EXPERIMENT_NAME', 'olist_delivery'))
    with mlflow.start_run(run_name=dag_run_id, tags={
        'dag_run_id': dag_run_id, 'snapshot_path': prepared['snapshot'],
    }) as run:
        run_id = run.info.run_id
        mlflow.log_params({
            'epochs': epochs, 'seed': seed, 'lr': lr, 'cutoff': prepared['cutoff'],
            'train_rows': len(train_frame), 'validation_rows': len(validation_frame),
            'snapshot_sha256': prepared['snapshot_sha256'],
        })
        predictor, history = fit_model(train_frame, validation_frame, epochs=epochs, seed=seed, lr=lr)
        for epoch in history:
            mlflow.log_metrics({key: epoch[key] for key in ('train_loss', 'validation_pr_auc')},
                               step=epoch['epoch'])
        model_directory = Path(prepared['snapshot']).parent / 'models' / run_id
        save_predictor(predictor, model_directory)
        sample = validation_frame.head(5)
        np.testing.assert_array_equal(predictor.predict(sample), load_predictor(model_directory).predict(sample))
        mlflow.log_artifacts(str(model_directory), artifact_path='model')
        mlflow.log_dict(prepared, 'data/snapshot.json')
        print(json.dumps({'mlflow_run_id': run_id, 'dag_run_id': dag_run_id,
                          'snapshot_sha256': prepared['snapshot_sha256']}, allow_nan=False))
    return run_id

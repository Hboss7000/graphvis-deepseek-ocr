"""Mandatory flushed CSV and optional network-free MLflow file-store logging."""
import csv
import hashlib
import json
import os
from pathlib import Path
import warnings

FIELDS = ['event', 'step', 'loss', 'lr', 'projector_lr', 'grad_norm', 'tokens_per_second',
          'examples_per_second', 'peak_vram_bytes', 'step_seconds', 'examples', 'tokens',
          'validation_loss', 'validation_seconds'] + [
              f'validation_{task}_loss' for task in ('node_description', 'node_degree',
              'highest_node_degree', 'node_number', 'edge_number', 'triple_listing')]


class TrainingLogger:
    def __init__(self, output_dir, params, tracking_uri=None, resume_step=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / 'metrics.csv'
        self.client = None
        self.run_id = None
        self.errors = []
        if self.csv_path.exists():
            if resume_step is None:
                raise FileExistsError(self.csv_path)
            with self.csv_path.open() as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != FIELDS:
                    raise ValueError('CSV schema mismatch on resume')
                previous = list(reader)
            # Preserve metrics of rolled-back work instead of hiding them.
            discarded = [r for r in previous if int(r['step']) > resume_step]
            if discarded:
                archive = self.output_dir / f'metrics_rolled_back_after_{resume_step}.csv'
                with archive.open('x', newline='') as handle:
                    writer = csv.DictWriter(handle, FIELDS)
                    writer.writeheader()
                    writer.writerows(discarded)
            with self.csv_path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, FIELDS)
                writer.writeheader()
                writer.writerows(r for r in previous if int(r['step']) <= resume_step)
        self.handle = self.csv_path.open('a', newline='')
        self.writer = csv.DictWriter(self.handle, FIELDS)
        if self.csv_path.stat().st_size == 0:
            self.writer.writeheader()
            self.flush()
        if tracking_uri:
            self.safe(self.start, params, tracking_uri)

    def safe(self, operation, *args):
        try:
            return operation(*args)
        except Exception as error:
            message = f'{type(error).__name__}: {error}'
            self.errors.append(message)
            with (self.output_dir / 'mlflow_errors.jsonl').open('a') as handle:
                handle.write(json.dumps({'operation': operation.__name__, 'error': message}) + '\n')
            warnings.warn('MLflow unavailable; CSV continues: ' + message)
            return None

    def start(self, params, tracking_uri):
        if not tracking_uri.startswith('file:'):
            raise ValueError('Only a file: MLflow store is allowed; no network URI')
        os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
        os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'false'
        os.environ['MLFLOW_ENABLE_ASYNC_LOGGING'] = 'false'
        os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'
        os.environ['MLFLOW_DISABLE_AGENT_HINT'] = 'true'
        os.environ['DO_NOT_TRACK'] = '1'
        from mlflow.tracking import MlflowClient
        self.client = MlflowClient(tracking_uri=tracking_uri)
        state_path = self.output_dir / 'mlflow_run.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['tracking_uri'] != tracking_uri:
                raise ValueError('MLflow store changed on resume')
            self.run_id = state['run_id']
            self.client.get_run(self.run_id)
        else:
            experiment = self.client.get_experiment_by_name('llava-stage1')
            experiment_id = experiment.experiment_id if experiment else self.client.create_experiment('llava-stage1')
            self.run_id = self.client.create_run(experiment_id,
                tags={'mlflow.runName': self.output_dir.name}).info.run_id
            state_path.write_text(json.dumps({'tracking_uri': tracking_uri, 'run_id': self.run_id}, indent=2) + '\n')
        for name, value in params.items():
            encoded = json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else str(value)
            if len(encoded) > 6000:
                encoded = 'sha256:' + hashlib.sha256(encoded.encode()).hexdigest() + '; full value in run_config.json'
            self.client.log_param(self.run_id, name, encoded)

    def flush(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def log(self, event, step, metrics):
        row = {'event': event, 'step': step, **metrics}
        self.writer.writerow({name: row.get(name, '') for name in FIELDS})
        self.flush()
        if self.client and self.run_id:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.safe(self.client.log_metric, self.run_id, name, float(value), None, step)

    def checkpoint(self, path, digest):
        if self.client and self.run_id:
            self.safe(self.client.set_tag, self.run_id, 'latest_checkpoint_path', str(path))
            self.safe(self.client.set_tag, self.run_id, 'latest_checkpoint_sha256', digest)

    def has_event(self, event, step):
        self.flush()
        with self.csv_path.open() as handle:
            return any(row['event'] == event and int(row['step']) == step
                       for row in csv.DictReader(handle))

    def readable(self):
        if not self.client or not self.run_id:
            return False
        run = self.safe(self.client.get_run, self.run_id)
        history = self.safe(self.client.get_metric_history, self.run_id, 'loss')
        return bool(run and history and not self.errors)

    def close(self, status='FINISHED'):
        self.flush()
        self.handle.close()
        if self.client and self.run_id:
            self.safe(self.client.set_terminated, self.run_id, status)

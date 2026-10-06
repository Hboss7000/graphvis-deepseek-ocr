#!/usr/bin/env python3
"""Export thesis loss/lr PNGs from mandatory CSV or a fetched MLflow file store."""
import argparse
import csv
import os
from pathlib import Path


def read_csv(path):
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    series = {}
    for key in ('loss', 'validation_loss', 'lr', 'projector_lr'):
        series[key] = [(int(row['step']), float(row[key])) for row in rows if row.get(key)]
    return series


def read_mlflow(store, run_id):
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=Path(store).resolve().as_uri())
    return {key: sorted([(m.step, m.value) for m in client.get_metric_history(run_id, key)])
            for key in ('loss', 'validation_loss', 'lr', 'projector_lr')}


def plot(series, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not series['loss'] or not series['lr']:
        raise ValueError('Need measured loss and learning-rate data')
    for filename, keys, ylabel in (
        ('loss.png', ('loss', 'validation_loss'), 'Answer + EOS loss'),
        ('lr.png', ('lr', 'projector_lr'), 'Learning rate')):
        fig, ax = plt.subplots(figsize=(6.4, 3.6), layout='constrained')
        for key in keys:
            points = series[key]
            if points:
                ax.plot(*zip(*points), label=key.replace('_', ' '), linewidth=1.25,
                        marker='o' if key == 'validation_loss' else None, markersize=3)
        ax.set(xlabel='Optimizer step', ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.ticklabel_format(axis='y', useOffset=False)
        ax.grid(alpha=.25)
        ax.legend()
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv', type=Path)
    group.add_argument('--mlflow-store', type=Path)
    p.add_argument('--run-id')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.mlflow_store and not args.run_id:
        p.error('--mlflow-store requires --run-id')
    series = read_csv(args.csv) if args.csv else read_mlflow(args.mlflow_store, args.run_id)
    plot(series, args.output_dir)
    print(args.output_dir / 'loss.png')
    print(args.output_dir / 'lr.png')


if __name__ == '__main__':
    main()

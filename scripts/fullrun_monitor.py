#!/usr/bin/env python3
"""One finite snapshot of the active job/log/progress; suitable for watch."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=Path('/workspace'));p.add_argument('--gpu',action='store_true')
    args=p.parse_args();path=args.workspace/'logs/fullrun_current_job.json'
    if path.exists():
        row=json.loads(path.read_text());print(f"{row['model']}/{row['job']} {row['state']} {row['completed']}/{row['expected']} cap={row['token_cap']}")
        print('Current job log: '+row['log_path'])
        log=Path(row['log_path'])
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-8192));lines=f.read().decode(errors='replace').splitlines()
            print('\n'.join(lines[-6:]))
    else:print('No active job marker yet: '+str(path))
    if args.gpu:subprocess.run(['nvidia-smi'],check=False)


if __name__=='__main__':main()

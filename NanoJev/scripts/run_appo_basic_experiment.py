#!/usr/bin/env python3
"""Run all predeclared APPO Basic inference controls using downloaded models."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('configs/appo_basic_v1.json'))
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.output_dir.exists():
        parser.error('workers must be positive and output-dir must be new')
    protocol = json.loads(args.protocol.read_text())
    jobs = []
    repositories = [protocol['primary']['checkpoint_repository'], *protocol['secondary_repositories']]
    for repo in repositories:
        directory = args.model_dir / repo.split('/')[-1]
        downloaded = json.loads((directory / 'download_manifest.json').read_text())
        if downloaded['repo'] != repo:
            raise ValueError('Downloaded repository disagrees with protocol')
        weights = [r for r in downloaded['files'] if r['file'].endswith('.pth')]
        if len(weights) != 1:
            raise ValueError('Exactly one pinned checkpoint per repository is required')
        weight = weights[0]
        for control in protocol['controllers_for_every_repository']:
            name = f"{repo.split('_')[-1]}_{control['controller']}_eps{control['epsilon']:g}"
            command = [sys.executable, str(Path(__file__).with_name('evaluate_appo_doom.py')),
                       '--checkpoint', str(directory / weight['file']), '--cfg', str(directory / 'cfg.json'),
                       '--checkpoint-sha256', weight['sha256'], '--model-id', repo,
                       '--revision', downloaded['revision'], '--sf-source-commit', protocol['sample_factory_source_commit'],
                       '--cases', protocol['case_file'], '--output', str(args.output_dir / (name + '.jsonl')),
                       '--controller', control['controller'], '--epsilon', str(control['epsilon']),
                       '--seed', '17', '--device', 'cpu', '--torch-threads', '1']
            jobs.append((name, command))
    args.output_dir.mkdir(parents=True)
    def run(job):
        name, command = job
        with (args.output_dir / (name + '.log')).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        return {'name': name, 'returncode': result.returncode, 'command': command}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({'name': result['name'], 'returncode': result['returncode']}), flush=True)
    (args.output_dir / 'launch_results.json').write_text(json.dumps(results, indent=2) + '\n')
    if any(r['returncode'] for r in results):
        raise SystemExit('One or more evaluations failed; inspect per-run logs')


if __name__ == '__main__':
    main()

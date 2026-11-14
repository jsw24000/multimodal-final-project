"""Collect only completed full-validation metrics from existing evaluator logs."""
import csv
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('PEARL_DATA_ROOT', '/data/Multimodal-final-project'))
OUT = ROOT / 'outputs/performance'
EXPECTED = {'voc21': 1449, 'voc20': 1449, 'ade20k': 2000}
PAPER = {
    'full_pearl': (64.1, 86.9, 19.4),
    'pa_only': (59.2, 85.0, 17.9), 'tlp_only': (35.4, 79.3, 11.7),
    'sclip': (59.1, 80.4, 16.1), 'naclip': (58.9, 79.7, 17.4),
    'clearclip': (51.8, 80.9, 16.7),
}
REPO_REF = {'sclip': (59.63, 81.54, 16.45),
            'naclip': (64.07, 83.03, 19.05)}
DATASETS = ('voc21', 'voc20', 'ade20k')
METHODS = ('vanilla', 'reduced', 'pa_only', 'tlp_only', 'full_pearl',
           'sclip', 'naclip', 'clearclip')


def git_sha(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def sources(method, dataset):
    if method == 'full_pearl':
        work = ROOT / 'outputs/work_dirs' / dataset
        log = OUT / f'{dataset}_full_pearl.log'
        if dataset == 'voc21':
            logs = sorted(work.glob('*/*.log'))
            log = logs[-1] if logs else log
    else:
        name = f'{dataset}_{method}' if method in METHODS[:4] else f'{method}_{dataset}'
        work, log = OUT / name, OUT / f'{name}.log'
    files = sorted(work.glob('*/*.json'))
    return (files[-1] if files else None), log


def run_command(method, dataset):
    config = f'configs/cfg_{dataset}.py'
    if method in ('sclip', 'naclip', 'clearclip'):
        return f'python experiments/performance/run_external.py {method} {dataset}'
    if method == 'full_pearl':
        return f'python eval.py --config {config} --attn pearl --prop on'
    settings = {'vanilla': ('vanilla', 'vanilla', 'off'),
                'reduced': ('reduced', 'vanilla', 'off'),
                'pa_only': ('reduced', 'pearl', 'off'),
                'tlp_only': ('reduced', 'vanilla', 'on')}
    arch, attn, prop = settings[method]
    return (f'python eval.py --config {config} --arch {arch} '
            f'--attn {attn} --prop {prop} '
            f'--work-dir $PEARL_DATA_ROOT/outputs/performance/{dataset}_{method}')


def main():
    rows, manifest = [], {}
    for method in METHODS:
        for j, dataset in enumerate(DATASETS):
            metric_file, log_file = sources(method, dataset)
            row = dict(method=method, dataset=dataset, protocol='official-code',
                       status='Not Run', samples='', expected_samples=EXPECTED[dataset],
                       paper_mIoU=PAPER.get(method, ('N/A',)*3)[j],
                       official_repo_mIoU=REPO_REF.get(method, ('N/A',)*3)[j],
                       local_mIoU='', delta='', delta_vs_official_repo='',
                       comparability=('different published protocol' if method in ('sclip','naclip','clearclip') else
                                      'no directly matched paper value' if method in ('vanilla','reduced') else
                                      'matched official-code setting'),
                       mAcc='', pAcc='',
                       peak_logged_memory_MiB='', elapsed_s='', log=str(log_file))
            if log_file.is_file():
                log = log_file.read_text(errors='replace')
                counts = re.findall(r'Iter\(test\)\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]', log)
                if metric_file and counts and tuple(map(int, counts[-1])) == (EXPECTED[dataset], EXPECTED[dataset]):
                    metrics = json.loads(metric_file.read_text())
                    row.update(status='Completed', samples=EXPECTED[dataset],
                               local_mIoU=metrics['mIoU'], mAcc=metrics['mAcc'],
                               pAcc=metrics['aAcc'])
                    if isinstance(row['paper_mIoU'], float) and method not in REPO_REF:
                        row['delta'] = round(metrics['mIoU'] - row['paper_mIoU'], 2)
                    if isinstance(row['official_repo_mIoU'], float):
                        row['delta_vs_official_repo'] = round(metrics['mIoU'] - row['official_repo_mIoU'], 2)
                    memories = [int(x) for x in re.findall(r'\bmemory:\s*(\d+)', log)]
                    if memories: row['peak_logged_memory_MiB'] = max(memories)
                    times = re.findall(r'\b(\d\d/\d\d \d\d:\d\d:\d\d)\b', log)
                    if len(times) > 1:
                        start, end = (datetime.strptime('2026/' + x, '%Y/%m/%d %H:%M:%S')
                                      for x in (times[0], times[-1]))
                        row['elapsed_s'] = (end-start).total_seconds()
                    classes = {}
                    for name, iou, acc in re.findall(r'^\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|$', log, re.M):
                        classes[name.strip()] = dict(iou=float(iou), acc=float(acc))
                    manifest[f'{method}_{dataset}'] = dict(metrics=metrics, per_class=classes,
                        metrics_file=str(metric_file), log=str(log_file), samples=EXPECTED[dataset],
                        command=run_command(method, dataset),
                        peak_logged_memory_MiB=row['peak_logged_memory_MiB'], elapsed_s=row['elapsed_s'])
                else:
                    row['status'] = 'Failed or Incomplete'
            rows.append(row)
    (PROJECT / 'results').mkdir(exist_ok=True)
    with (PROJECT / 'results/performance_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    checkpoint = ROOT / 'checkpoints/clip/ViT-B-16.pt'
    digest = hashlib.sha256()
    with checkpoint.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    metadata = dict(project_git_sha=git_sha(PROJECT), checkpoint_sha256=digest.hexdigest(),
                    commands={'pearl_variants':'bash experiments/performance/run_pearl_matrix.sh',
                              'external':'bash experiments/performance/run_external_matrix.sh',
                              'full_pearl':'python eval.py --config configs/cfg_<dataset>.py --attn pearl --prop on'},
                    versions={n:version(n) for n in ('torch','torchvision','mmcv','mmengine','mmsegmentation')},
                    external_repos={n:git_sha(ROOT/'cache/external_repos'/n) for n in ('SCLIP','NACLIP','ClearCLIP')},
                    runs=manifest)
    (PROJECT/'results/experiment_manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f"Completed {len(manifest)}/{len(rows)} full-validation runs")


if __name__ == '__main__': main()

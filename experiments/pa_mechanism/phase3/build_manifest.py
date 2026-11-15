"""Record actual Phase 3 completion and provenance from sidecars and raw JSONL."""
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(os.environ.get('PEARL_DATA_ROOT', '/data/Multimodal-final-project'))
RUNS = ROOT/'outputs/pa_mechanism/phase3/runs'
OUT = Path('results/pa_mechanism_phase3_manifest.json')


def main():
    checkpoint = ROOT/'checkpoints/clip/ViT-B-16.pt'
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    records = []
    for config_path in sorted(RUNS.glob('*.config.json')):
        config = json.loads(config_path.read_text())
        path = config_path.with_name(config_path.name.replace('.config.json', '.jsonl'))
        observed = {variant: set() for variant in config['variants']}
        duplicate = 0
        if path.exists():
            for line in path.open():
                row = json.loads(line)
                variant = row['variant']
                index = row['index']
                if variant not in observed or index in observed[variant]:
                    duplicate += 1
                else:
                    observed[variant].add(index)
        expected = set(config['indices'])
        completion = {v: len(seen) for v, seen in observed.items()}
        records.append(dict(config=str(config_path), records=str(path), dataset=config['dataset'],
                            tlp=config['tlp'], variants=config['variants'],
                            expected_images=len(expected), completed_images=completion,
                            complete=duplicate == 0 and all(seen == expected for seen in observed.values()),
                            duplicate_or_unknown_records=duplicate, seed=config['seed'],
                            solver=config['solver'], original_alpha_kk=config['original_alpha_kk'],
                            slide_crop=config['slide_crop'], slide_stride=config['slide_stride'],
                            checkpoint=config['checkpoint'], git_base_commit=config['git_base_commit']))
    manifest = dict(created_from='actual Phase 3 raw records',
                    base_commit_at_generation=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    checkpoint=str(checkpoint), checkpoint_sha256=digest.hexdigest(),
                    data_root=str(ROOT),
                    environment=str(ROOT/'cache/conda_envs/pearl'),
                    selection_protocol='results/pa_mechanism_phase3_selection_protocol.json',
                    runs=records, runs_complete=sum(r['complete'] for r in records),
                    runs_total=len(records),
                    raw_root=str(ROOT/'outputs/pa_mechanism/phase3'))
    OUT.write_text(json.dumps(manifest, indent=2))
    print(OUT, manifest['runs_complete'], '/', manifest['runs_total'])


if __name__ == '__main__':
    main()

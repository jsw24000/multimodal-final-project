"""Build a traceable manifest from actual Phase 2 run records."""
import json,subprocess,hashlib
from pathlib import Path

ROOT=Path('/data/Multimodal-final-project')
RUN=ROOT/'outputs/pa_mechanism/phase2/runs'
OUT=Path('results/pa_mechanism_phase2_manifest.json')

def main():
    ckpt=ROOT/'checkpoints/clip/ViT-B-16.pt'
    digest=hashlib.sha256()
    with ckpt.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            digest.update(chunk)
    sha=digest.hexdigest()
    runs=[]
    for config in sorted(RUN.glob('*.config.json')):
        cfg=json.loads(config.read_text());path=config.with_name(config.name.replace('.config.json','.jsonl'))
        done={v:set() for v in cfg['variants']}
        if path.exists():
            for line in path.read_text().splitlines():
                r=json.loads(line);done[r['variant']].add(r['index'])
        counts={v:len(x) for v,x in done.items()}
        expected=len(cfg['indices'])
        runs.append(dict(config=str(config),records=str(path),dataset=cfg['dataset'],tlp=cfg['tlp'],variants=cfg['variants'],
                         expected_images=expected,completed_images=counts,complete=all(x==expected for x in counts.values()),
                         seed=cfg['seed'],base_commit=cfg['git_commit']))
    diagnostics=ROOT/'outputs/pa_mechanism/phase2/metrics/voc21_attention_1449_24000.jsonl'
    diag_count=sum(1 for _ in diagnostics.open()) if diagnostics.exists() else 0
    receivers=ROOT/'outputs/pa_mechanism/phase2/metrics/voc21_key_receivers_1449.jsonl'
    receiver_count=sum(1 for _ in receivers.open()) if receivers.exists() else 0
    fg_bg=ROOT/'outputs/pa_mechanism/phase2/metrics/voc21_fg_bg_1449.jsonl'
    fg_bg_count=sum(1 for _ in fg_bg.open()) if fg_bg.exists() else 0
    ade_crop=ROOT/'outputs/pa_mechanism/phase2/metrics/ade20k_crop_attention_2000.jsonl'
    ade_crop_count=sum(1 for _ in ade_crop.open()) if ade_crop.exists() else 0
    manifest=dict(created_from='actual records',base_commit_at_generation=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  checkpoint=str(ckpt),checkpoint_sha256=sha,data_root=str(ROOT),python_environment=str(ROOT/'cache/conda_envs/pearl'),
                  gpu='NVIDIA RTX PRO 6000 Blackwell Workstation Edition',runs=runs,
                  voc21_attention_diagnostic_images=diag_count,
                  voc21_key_receiver_diagnostic_images=receiver_count,
                  voc21_foreground_background_audit_images=fg_bg_count,
                  ade20k_attention_sampled_crop_images=ade_crop_count,
                  benchmark_files=[str(p) for p in sorted((ROOT/'outputs/pa_mechanism/phase2/metrics').glob('benchmark_voc21_tlp_*.json'))],
                  artifacts_root=str(ROOT/'outputs/pa_mechanism/phase2'))
    OUT.write_text(json.dumps(manifest,indent=2))
    print(OUT,'runs',len(runs),'diagnostics',diag_count)

if __name__=='__main__':main()

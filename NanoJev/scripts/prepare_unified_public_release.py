#!/usr/bin/env python3
"""Prepare public cards and manifests from the verified unified private snapshot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n' if isinstance(data, dict) else data)


def copy_snapshot(source, target):
    # Only immutable large weights share storage. Cards and manifests get new files.
    if str(source).endswith('.safetensors'):
        os.link(source, target)
    else:
        shutil.copy2(source, target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    assert not args.output.exists()
    summary = {}
    for kind in ('model', 'dataset'):
        source, target = args.source / kind, args.output / kind
        manifest = json.loads((source / 'SHA256_MANIFEST.json').read_text())
        assert manifest['kind'] == kind and manifest['private'] is True
        expected = {row['path'] for row in manifest['files']} | {'SHA256_MANIFEST.json'}
        assert len(expected) == len(manifest['files']) + 1
        entries = list(source.rglob('*'))
        assert all(not entry.is_symlink() for entry in entries)
        assert {str(entry.relative_to(source)) for entry in entries if entry.is_file()} == expected
        for row in manifest['files']:
            relative = Path(row['path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            assert sha(source / row['path']) == row['sha256']
        shutil.copytree(source, target, copy_function=copy_snapshot)
        for name in ('README.md', 'TRAINING_RECIPE.md'):
            text = (target / name).read_text()
            text = text.replace('NanoJev-Data-dev', 'NanoJev-Data').replace('NanoJev-dev', 'NanoJev')
            text = text.replace('This private development release', 'This public release')
            text = text.replace('Authenticate with a Hugging Face account that can read this private repository.\n', 'The release is available without authentication.\n')
            text = text.replace('Model and data repositories stay private during this review.', 'Model and data snapshots are published as `unified-games-v1`.')
            text = text.replace('Download this private review package\nwith an authorized Hugging Face login:', 'Download the public data package:')
            text = text.replace('Stored training rows total 18,760 per target variant.',
                                'Stored questions across all five splits total 18,760 per target variant.')
            text = text.replace('After authenticated `snapshot_download` of both private development repositories,', 'After downloading the public release snapshots with the allowlist below,')
            if name == 'README.md' and kind == 'model':
                text = text.replace('path = snapshot_download("C-Tianyu/NanoJev", revision="unified-games-v1")',
                    'path = snapshot_download("C-Tianyu/NanoJev", revision="unified-games-v1",\n'
                    '                         token=False, allow_patterns=["best.safetensors", "config.json",\n'
                    '                         "backbone_config/*", "tokenizer/*", "source/*"])')
                text += '\n## Release history\n\nThe inference download above includes the current model and source. The full training\npackage also includes `training_initialization/`; use the manifest allowlist in\n[the recipe](TRAINING_RECIPE.md) to download all current files.\n\nThe current release is verified by `SHA256_MANIFEST.json`. Retained\n`MODEL_MANIFEST.json`, `GAMES_MODEL_MANIFEST.json`, `stage1/` and `variants/`\nbelong to earlier releases. The prior root snapshot remains available at\n[`legacy-before-unified-games-v1`](https://huggingface.co/C-Tianyu/NanoJev/tree/legacy-before-unified-games-v1).\n'
            if name == 'README.md' and kind == 'dataset':
                text = text.replace('from huggingface_hub import snapshot_download\npath = snapshot_download("C-Tianyu/NanoJev-Data", repo_type="dataset",\n                         revision="unified-games-v1")',
                    'import json\nfrom huggingface_hub import hf_hub_download, snapshot_download\n'
                    'repo = "C-Tianyu/NanoJev-Data"\nversion = "unified-games-v1"\n'
                    'manifest_path = hf_hub_download(repo, "SHA256_MANIFEST.json", repo_type="dataset",\n'
                    '                                revision=version, token=False)\n'
                    'with open(manifest_path) as handle:\n    names = [row["path"] for row in json.load(handle)["files"]]\n'
                    'path = snapshot_download(repo, repo_type="dataset", revision=version, token=False,\n'
                    '                         allow_patterns=names + ["SHA256_MANIFEST.json"])')
                text += '\n## Release history\n\n`SHA256_MANIFEST.json` describes this complete unified release. The retained root\n`manifest.json` and `verify_dataset.py` describe the earlier dataset layout.\nIts original snapshot remains available at\n[`legacy-before-unified-games-v1`](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data/tree/legacy-before-unified-games-v1).\n'
            if name == 'TRAINING_RECIPE.md':
                marker='## Environment and selected arm\n'
                download='''## Download the complete current package

Download the exact manifest-listed files so retained historical checkpoints are
not included in the training package download. No authentication is required.

```python
import json
from huggingface_hub import hf_hub_download, snapshot_download

for repo, kind, name in [("C-Tianyu/NanoJev", "model", "MODEL_DIR"),
                         ("C-Tianyu/NanoJev-Data", "dataset", "DATA_DIR")]:
    manifest = hf_hub_download(repo, "SHA256_MANIFEST.json", repo_type=kind,
                               revision="unified-games-v1", token=False)
    with open(manifest) as handle:
        names = [row["path"] for row in json.load(handle)["files"]]
    path = snapshot_download(repo, repo_type=kind, revision="unified-games-v1",
                             token=False, allow_patterns=names + ["SHA256_MANIFEST.json"])
    print(name + "=" + path)
```

'''
                assert marker in text
                text=text.replace(marker,download+marker)
            assert 'NanoJev-dev' not in text and 'NanoJev-Data-dev' not in text
            assert 'private' not in text.lower(), (kind, name)
            save(target / name, text)
        if kind == 'dataset':
            results = target / 'docs/SONIC_PREDICT_POSITION_RESULTS.md'
            text = results.read_text().replace('The private development checkout contains:',
                'Local experiment artifacts are organized as follows:')
            text = text.replace('The current model and dataset\nremain private development artifacts.',
                'The current model and complete mixed dataset are published as `unified-games-v1` in\n'
                '[NanoJev](https://huggingface.co/C-Tianyu/NanoJev) and\n'
                '[NanoJev-Data](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data).')
            save(results, text)
            recipe = target / 'docs/SONIC_PREDICT_POSITION.md'
            save(recipe, recipe.read_text().replace('private checkpoint bundles', 'checkpoint bundles'))
        manifest.update({'private': False, 'model_repo': 'C-Tianyu/NanoJev',
            'dataset_repo': 'C-Tianyu/NanoJev-Data', 'created_at': datetime.now(timezone.utc).isoformat(),
            'source_snapshot': {**inventory['private_sources'][kind],
                'package_manifest_sha256': sha(source / 'SHA256_MANIFEST.json')},
            'previous_public_revision': inventory['public'][kind]['revision'],
            'legacy_tag': 'legacy-before-unified-games-v1', 'release_tag': 'unified-games-v1',
            'public_package_builder_sha256': sha(__file__)})
        manifest['files'] = [{'path': str(p.relative_to(target)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                             for p in sorted(target.rglob('*')) if p.is_file() and p != target / 'SHA256_MANIFEST.json']
        save(target / 'SHA256_MANIFEST.json', manifest)
        summary[kind]={'files': len(manifest['files']) + 1,
                       'bytes': sum(r['bytes'] for r in manifest['files']) + (target / 'SHA256_MANIFEST.json').stat().st_size,
                       'manifest_sha256': sha(target / 'SHA256_MANIFEST.json')}
    save(args.output.parent / 'package_inventory.json', summary)
    print(json.dumps({'passed': True, 'packages': summary}))


if __name__ == '__main__':
    main()

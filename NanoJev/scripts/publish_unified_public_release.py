#!/usr/bin/env python3
"""Publish the fixed unified game release, retaining earlier public snapshots."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_VERBOSITY'] = 'error'
os.environ['RUST_LOG'] = 'error'
from dotenv import dotenv_values
import httpx
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from upload_unified_private_release import sha, blob_sha, save, log, validate_lfs_attribute_additions

REPOSITORIES = {'model': 'C-Tianyu/NanoJev', 'dataset': 'C-Tianyu/NanoJev-Data'}
SOURCES = {'model': 'C-Tianyu/NanoJev-dev', 'dataset': 'C-Tianyu/NanoJev-Data-dev'}
TAG = 'unified-games-v1'
LEGACY = 'legacy-before-unified-games-v1'
WEIGHT_SHA = 'f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b'


def fixed_tag(api, repo, kind, name, revision):
    refs = api.list_repo_refs(repo, repo_type=kind)
    if any(item.name == name for item in refs.tags):
        assert api.repo_info(repo, repo_type=kind, revision=name).sha == revision, 'Existing tag points elsewhere'
    else:
        api.create_tag(repo, repo_type=kind, tag=name, revision=revision,
                       tag_message='Preserved previous release' if name == LEGACY else 'Verified unified game release')
    assert api.repo_info(repo, repo_type=kind, revision=name).sha == revision


def source_guards(api, inventory):
    for kind, repo in SOURCES.items():
        expected = inventory['private_sources'][kind]
        info = api.repo_info(repo, repo_type=kind)
        tagged = api.repo_info(repo, repo_type=kind, revision=TAG)
        assert expected['repo_id'] == repo and info.private is True
        assert info.sha == tagged.sha == expected['revision']


def range_check(repo, revision, name, local):
    url = hf_hub_url(repo, name, revision=revision)
    with local.open('rb') as handle:
        expected = handle.read(1024)
    with httpx.stream('GET', url, headers={'Range': 'bytes=0-1023', 'Accept-Encoding': 'identity'},
                      follow_redirects=True, timeout=60, trust_env=False) as response:
        assert response.status_code == 206
        assert response.headers.get('content-range', '').startswith('bytes 0-1023/')
        payload = b''
        for chunk in response.iter_bytes():
            payload += chunk
            if len(payload) > 1024:
                break
        assert payload == expected
    return {'path': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
            'http_status': 206, 'authenticated': False}


def publish_one(api, token, kind, folder, inventory, receipts, resume):
    repo = REPOSITORIES[kind]
    baseline = inventory['public'][kind]
    assert baseline['repo_id'] == repo and baseline['private'] is False
    manifest = json.loads((folder / 'SHA256_MANIFEST.json').read_text())
    assert manifest['kind'] == kind and manifest['private'] is False
    assert manifest['selected_checkpoint_sha256'] == WEIGHT_SHA
    assert manifest['source_snapshot']['revision'] == inventory['private_sources'][kind]['revision']
    assert manifest['model_repo'] == REPOSITORIES['model'] and manifest['dataset_repo'] == REPOSITORIES['dataset']
    rows = manifest['files'] + [{'path': 'SHA256_MANIFEST.json', 'bytes': (folder / 'SHA256_MANIFEST.json').stat().st_size,
                                'sha256': sha(folder / 'SHA256_MANIFEST.json')}]
    names = {row['path'] for row in rows}
    assert len(names) == len(rows) and '.gitattributes' not in names
    for row in rows:
        relative = Path(row['path']); path = folder / relative
        assert not relative.is_absolute() and '..' not in relative.parts and path.is_file() and not path.is_symlink()
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
    before = api.repo_info(repo, repo_type=kind, revision=baseline['revision'], files_metadata=True)
    head = api.repo_info(repo, repo_type=kind)
    assert before.private is False and head.private is False
    assert {item.rfilename: item.blob_id for item in before.siblings} == {row['path']: row['blob_id'] for row in baseline['files']}
    if head.sha != baseline['revision']:
        assert resume, 'Public head changed; inspect it before using --verify-existing'
        revision = head.sha
    else:
        # Fail before any upload if a release tag already belongs to another commit.
        refs = api.list_repo_refs(repo, repo_type=kind)
        assert not any(item.name == TAG for item in refs.tags), 'Release tag already exists'
        fixed_tag(api, repo, kind, LEGACY, baseline['revision'])
        log(phase='public_upload_started', repo=repo, files=len(rows), bytes=sum(row['bytes'] for row in rows))
        revision = api.upload_folder(repo_id=repo, repo_type=kind, folder_path=folder,
            allow_patterns=sorted(names), ignore_patterns=['.cache/**'], parent_commit=baseline['revision'],
            commit_message='Publish unified game model and complete reproducibility package' if kind == 'model' else
                           'Publish complete unified game data and recorded evaluation').oid
    after = api.repo_info(repo, repo_type=kind, revision=revision, files_metadata=True)
    assert after.private is False
    remote = {item.rfilename: item for item in after.siblings}
    verified = []
    for row in rows:
        item = remote[row['path']]
        assert item.size == row['bytes']
        if item.lfs is not None:
            assert item.lfs.sha256 == row['sha256']; method = 'remote_lfs_sha256'
        else:
            assert item.blob_id == blob_sha(folder / row['path']); method = 'remote_git_blob_identity_of_local_sha256_verified_bytes'
        verified.append({**row, 'remote_verification': method})
    preserved = []; hub_added = []
    for item in before.siblings:
        if item.rfilename in names:
            continue
        current = remote[item.rfilename]
        if item.rfilename == '.gitattributes' and current.blob_id != item.blob_id:
            paths = {row['path'] for row in rows if remote[row['path']].lfs is not None}
            old = Path(hf_hub_download(repo, '.gitattributes', repo_type=kind, revision=before.sha,
                        token=token, cache_dir=receipts / 'attributes')).read_text()
            new = Path(hf_hub_download(repo, '.gitattributes', repo_type=kind, revision=revision,
                        token=token, cache_dir=receipts / 'attributes')).read_text()
            hub_added = validate_lfs_attribute_additions(old, new, paths)
        else:
            assert current.blob_id == item.blob_id and current.size == item.size
            if item.lfs is not None:
                assert current.lfs is not None and current.lfs.sha256 == item.lfs.sha256
        preserved.append({'path': item.rfilename, 'previous_blob_id': item.blob_id,
                          'current_blob_id': current.blob_id, 'hub_metadata_additions': item.rfilename == '.gitattributes' and bool(hub_added)})
    fixed_tag(api, repo, kind, LEGACY, baseline['revision'])
    fixed_tag(api, repo, kind, TAG, revision)
    anonymous = HfApi(token=False)
    visible = anonymous.repo_info(repo, repo_type=kind, revision=TAG)
    assert visible.private is False and visible.sha == revision
    assert anonymous.repo_info(repo, repo_type=kind, revision=LEGACY).sha == baseline['revision']
    samples = ['README.md', 'TRAINING_RECIPE.md', 'SHA256_MANIFEST.json']
    samples += ['config.json', 'backbone_config/config.json', 'training_initialization/config.json'] if kind == 'model' else [
        'DATA_STATISTICS.json', 'unified/hard/manifest.json', 'unified/soft/manifest.json', 'configs/hard_navigation_demo_v1_cases.jsonl']
    expected = {row['path']: row for row in rows}; downloaded = []
    for name in samples:
        path = hf_hub_download(repo, filename=name, repo_type=kind, revision=revision, token=False,
                               force_download=True, cache_dir=receipts / 'anonymous_readback')
        assert sha(path) == expected[name]['sha256']
        downloaded.append({'path': name, 'sha256': sha(path), 'authenticated': False})
    ranges = [range_check(repo, revision, name, folder / name) for name in
              ('best.safetensors', 'training_initialization/best.safetensors')] if kind == 'model' else []
    result = {'repository': repo, 'repo_type': kind, 'url': 'https://huggingface.co/' + ('datasets/' if kind == 'dataset' else '') + repo,
        'private': False, 'revision': revision, 'tag': TAG, 'previous_revision': baseline['revision'], 'legacy_tag': LEGACY,
        'files': verified, 'uploaded_files': len(rows), 'uploaded_bytes': sum(row['bytes'] for row in rows),
        'remote_total_files': len(after.siblings), 'remote_total_bytes': sum(item.size for item in after.siblings),
        'retained_paths': preserved, 'hub_added_lfs_rules': hub_added, 'no_files_deleted': True,
        'all_uploaded_hashes_verified': True, 'old_versions_preserved': True,
        'anonymous_readback': downloaded, 'anonymous_weight_ranges': ranges,
        'remote_weight_sha256': {name: item.lfs.sha256 for name, item in remote.items()
                                if name in names and name.endswith('.safetensors') and item.lfs is not None},
        'package_manifest_sha256': sha(folder / 'SHA256_MANIFEST.json')}
    save(receipts / (kind + '_public_upload_receipt.json'), result)
    log(phase='public_release_verified', repo=repo, revision=revision, files=len(rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    credentials = parser.add_mutually_exclusive_group(required=True)
    credentials.add_argument('--env', type=Path)
    credentials.add_argument('--token-stdin', action='store_true')
    parser.add_argument('--packages', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-existing', action='store_true', help='Resume publication: verify an updated head, or upload if still at the original head')
    args = parser.parse_args()
    if args.token_stdin:
        token = sys.stdin.readline().strip()
        assert token.startswith('hf_') and token.replace('_', '').isalnum()
    else:
        tokens = {v for v in dotenv_values(args.env).values() if isinstance(v, str) and v.startswith('hf_')}
        assert len(tokens) == 1; token = tokens.pop()
    api = HfApi(token=token); assert api.whoami()['name'] == 'C-Tianyu'
    inventory = json.loads(args.inventory.read_text())
    source_guards(api, inventory)
    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {pool.submit(publish_one, api, token, kind, args.packages / kind, inventory, args.receipts, args.verify_existing): kind for kind in REPOSITORIES}
        for task in as_completed(tasks):
            results[tasks[task]] = task.result()
    source_guards(api, inventory)
    result = {'schema': 'nanojev-unified-public-hf-release-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'passed': True, 'selected_checkpoint_sha256': WEIGHT_SHA, 'selected_checkpoint_step': 400,
        'private_sources_unchanged': inventory['private_sources'], 'model': results['model'], 'dataset': results['dataset']}
    save(args.output, result)
    log(phase='public_release_complete', output=str(args.output), model_revision=results['model']['revision'], dataset_revision=results['dataset']['revision'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        log(phase='failed', error_type=type(error).__name__, http_status=getattr(getattr(error, 'response', None), 'status_code', None),
            traceback=[{'file': Path(frame.filename).name, 'line': frame.lineno, 'function': frame.name}
                       for frame in traceback.extract_tb(error.__traceback__)])
        raise SystemExit(1)

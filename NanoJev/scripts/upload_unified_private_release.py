#!/usr/bin/env python3
"""Upload and verify the explicitly prepared private unified development release.

Credentials are read into memory from an explicit .env path, never printed or
stored by this script. Only the two declared development repositories can be
created or updated. Public repositories are observed solely as unchanged guards.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_VERBOSITY'] = 'error'
os.environ['RUST_LOG'] = 'error'
from dotenv import dotenv_values
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

REPOSITORIES = {'model': 'C-Tianyu/NanoJev-dev', 'dataset': 'C-Tianyu/NanoJev-Data-dev'}
PUBLIC = {'model': 'C-Tianyu/NanoJev', 'dataset': 'C-Tianyu/NanoJev-Data'}
TAG = 'unified-games-v1'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def blob_sha(path):
    digest = hashlib.sha1(b'blob ' + str(path.stat().st_size).encode() + b'\0')
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, path)


def log(**data):
    print(json.dumps(data), flush=True)


def validate_lfs_attribute_additions(previous, updated, uploaded_lfs_paths):
    """Allow Hub-added LFS rules for uploaded files, preserving existing rules."""
    old_lines, new_lines = previous.splitlines(keepends=True), updated.splitlines(keepends=True)
    allowed = {name + ' filter=lfs diff=lfs merge=lfs -text' for name in uploaded_lfs_paths}
    additions = []
    for operation, _, _, start, end in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
        assert operation in ('equal', 'insert'), 'Existing Git attributes changed'
        if operation == 'insert':
            for line in new_lines[start:end]:
                rule = line.rstrip('\r\n')
                assert rule in allowed, 'Unexpected Git attribute rule'
                additions.append(rule)
    return additions


def upload_one(api, token, kind, folder, receipts):
    rid = REPOSITORIES[kind]
    manifest = json.loads((folder / 'SHA256_MANIFEST.json').read_text())
    assert manifest['kind'] == kind and manifest['private'] is True
    rows = manifest['files'] + [{'path': 'SHA256_MANIFEST.json', 'bytes': (folder / 'SHA256_MANIFEST.json').stat().st_size,
                                 'sha256': sha(folder / 'SHA256_MANIFEST.json')}]
    names = [row['path'] for row in rows]
    assert len(names) == len(set(names)) and 'README.md' in names
    for row in rows:
        relative = Path(row['path']); path = folder / relative
        assert not relative.is_absolute() and '..' not in relative.parts and path.is_file() and not path.is_symlink()
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], row['path']
    try:
        before = api.repo_info(rid, repo_type=kind, files_metadata=True)
        assert before.private is True, 'Only private development repositories may be updated'
    except RepositoryNotFoundError:
        api.create_repo(rid, repo_type=kind, private=True, exist_ok=False)
        before = api.repo_info(rid, repo_type=kind, files_metadata=True)
        assert before.private is True
    log(phase='upload_started', repo=rid, private=True, files=len(rows), bytes=sum(r['bytes'] for r in rows))
    commit = api.upload_folder(repo_id=rid, repo_type=kind, folder_path=folder,
        allow_patterns=names, ignore_patterns=['.cache/**'], parent_commit=before.sha,
        commit_message='Add complete unified game model development release' if kind == 'model' else
                       'Add complete mixed game supervision and verified recordings')
    after = api.repo_info(rid, repo_type=kind, revision=commit.oid, files_metadata=True)
    assert after.private is True
    remote = {item.rfilename: item for item in after.siblings}
    verified = []
    for row in rows:
        item = remote[row['path']]
        assert item.size == row['bytes'], row['path']
        if item.lfs is not None:
            assert item.lfs.sha256 == row['sha256'], row['path']
            method = 'remote_lfs_sha256'
        else:
            assert item.blob_id == blob_sha(folder / row['path']), row['path']
            method = 'remote_git_blob_identity_of_local_sha256_verified_bytes'
        verified.append({**row, 'remote_verification': method})
    managed_attribute_rules = []
    for item in before.siblings:
        if item.rfilename not in names:
            assert item.rfilename in remote
            if item.rfilename == '.gitattributes' and remote[item.rfilename].blob_id != item.blob_id:
                versions = [Path(hf_hub_download(rid, filename='.gitattributes', repo_type=kind,
                            revision=revision, token=token, cache_dir=receipts / 'readback_cache')).read_text()
                            for revision in (before.sha, commit.oid)]
                managed_attribute_rules = validate_lfs_attribute_additions(
                    *versions, {name for name in names if remote[name].lfs is not None})
            else:
                assert remote[item.rfilename].blob_id == item.blob_id
    samples = ['README.md', 'SHA256_MANIFEST.json', 'TRAINING_RECIPE.md']
    samples += ['config.json', 'backbone_config/config.json', 'training_initialization/config.json'] if kind == 'model' else [
        'DATA_STATISTICS.json', 'unified/hard/manifest.json', 'unified/soft/manifest.json',
        'configs/hard_navigation_demo_v1_cases.jsonl']
    expected = {row['path']: row for row in rows}
    downloads = []
    for name in samples:
        path = hf_hub_download(rid, filename=name, repo_type=kind, revision=commit.oid, token=token,
                               cache_dir=receipts / 'readback_cache')
        assert sha(path) == expected[name]['sha256'], name
        downloads.append({'path': name, 'sha256': sha(path)})
    refs = api.list_repo_refs(rid, repo_type=kind)
    existing_tag = next((item for item in refs.tags if item.name == TAG), None)
    if existing_tag is None:
        api.create_tag(rid, repo_type=kind, tag=TAG, revision=commit.oid,
                       tag_message='Verified private unified game development release')
    # Resolve annotated tags to commits; target_commit can identify the tag object.
    tagged = api.repo_info(rid, repo_type=kind, revision=TAG)
    assert tagged.sha == commit.oid and tagged.private is True, 'Release tag must resolve to the verified private commit'
    result = {'repository': rid, 'repo_type': kind, 'url': 'https://huggingface.co/' + ('datasets/' if kind == 'dataset' else '') + rid,
        'revision': commit.oid, 'tag': TAG, 'private': True, 'previous_revision': before.sha,
        'files': verified, 'uploaded_files': len(rows), 'uploaded_bytes': sum(r['bytes'] for r in rows),
        'remote_total_files': len(after.siblings),
        'remote_total_bytes': sum(item.size for item in after.siblings),
        'remote_weight_sha256': {name: item.lfs.sha256 for name, item in remote.items()
                                 if name.endswith('.safetensors') and item.lfs is not None},
        'all_uploaded_hashes_verified': True, 'previous_unrelated_files_preserved': True,
        'hub_added_lfs_rules': managed_attribute_rules,
        'authenticated_readback': downloads, 'full_weights_redownloaded': False,
        'package_manifest_sha256': sha(folder / 'SHA256_MANIFEST.json')}
    save(receipts / (kind + '_upload_receipt.json'), result)
    log(phase='verified', repo=rid, revision=commit.oid, private=True, files=len(rows), tag=TAG)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    credentials = parser.add_mutually_exclusive_group(required=True)
    credentials.add_argument('--env', type=Path)
    credentials.add_argument('--token-stdin', action='store_true',
                             help='Read one credential from standard input without storing it')
    parser.add_argument('--packages', type=Path, required=True,
                        help='Explicit directory containing the reviewed model/ and dataset/ packages')
    parser.add_argument('--inventory', type=Path, default=Path('runs/hf_unified_release_v1/account_inventory.json'))
    parser.add_argument('--receipts', type=Path, default=Path('runs/hf_unified_release_v1'))
    parser.add_argument('--output', type=Path, default=Path('results/huggingface_unified_dev_release.json'))
    args = parser.parse_args(argv)
    if args.token_stdin:
        token = sys.stdin.readline().strip()
        assert token.startswith('hf_') and token.isascii() and token.replace('_', '').isalnum()
    else:
        values = dotenv_values(args.env)
        tokens = {value for value in values.values() if isinstance(value, str) and value.startswith('hf_')}
        assert len(tokens) == 1, 'Expected exactly one Hugging Face credential in the supplied .env'
        token = tokens.pop()
    api = HfApi(token=token)
    assert api.whoami()['name'] == 'C-Tianyu'
    inventory = json.loads(args.inventory.read_text())
    guards = {item['repo_type']: item for item in inventory['repositories'] if item['repo_id'] in PUBLIC.values()}
    assert set(guards) == set(PUBLIC)
    for kind, rid in PUBLIC.items():
        current = api.repo_info(rid, repo_type=kind)
        assert current.sha == guards[kind]['revision'] and current.private is False
    started = time.monotonic()
    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(upload_one, api, token, kind, args.packages / kind, args.receipts): kind for kind in REPOSITORIES}
        for future in as_completed(futures): results[futures[future]] = future.result()
    for kind, rid in PUBLIC.items():
        current = api.repo_info(rid, repo_type=kind)
        assert current.sha == guards[kind]['revision'] and current.private is False
    result = {'schema': 'nanojev-unified-private-hf-release-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'passed': True, 'account': 'C-Tianyu', 'phase': 'private_development_review',
        'public_repositories_unchanged': {kind: {'repository': item['repo_id'], 'revision': item['revision']} for kind, item in guards.items()},
        'selected_checkpoint_sha256': 'f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b',
        'selected_checkpoint_step': 400, 'initial_checkpoint_included': True,
        'model': results['model'], 'dataset': results['dataset'], 'elapsed_seconds': time.monotonic() - started}
    save(args.output, result)
    log(phase='release_complete', output=str(args.output), model_revision=results['model']['revision'],
        dataset_revision=results['dataset']['revision'], private=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Exception text and HTTP payloads may include signed URLs: report only safe metadata.
        log(phase='failed', error_type=type(error).__name__,
            http_status=getattr(getattr(error, 'response', None), 'status_code', None),
            traceback=[{'file': Path(frame.filename).name, 'line': frame.lineno, 'function': frame.name}
                       for frame in traceback.extract_tb(error.__traceback__)])
        raise SystemExit(1)

#!/usr/bin/env python3
"""Stage the independent development viewer in a separate static site checkout."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--web-root', type=Path, default=Path('web'))
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--site-url', help='Exact independent origin returned by the hosting provider')
    args = parser.parse_args()
    source, project = args.web_root.resolve(), args.project.resolve()
    marker = project / '.nanojev-development.json'
    if project.exists() and any(project.iterdir()) and not marker.is_file():
        raise ValueError('Use a new checkout or a previously marked development site')
    if source == project or source.is_relative_to(project) or project.is_relative_to(source):
        raise ValueError('Keep the static hosting checkout separate from source assets')
    old_marker = json.loads(marker.read_text()) if marker.exists() else {}
    site_url = args.site_url or old_marker.get('site_url')
    if site_url and (urlparse(site_url).scheme != 'https' or urlparse(site_url).path not in ('', '/')):
        raise ValueError('Use the exact HTTPS site origin')
    data = json.loads((source / 'dev/shooting_results.json').read_text())
    if data.get('schema') != 'nanojev-shooting-demo-v1' or not data.get('cases'):
        raise ValueError('A completed shooting recording is required')
    for case in data['cases']:
        if {s['id'] for s in case['systems']} != {'jev', 'nanojev', 'base'}:
            raise ValueError('Each case must include all three recorded models')
    project.mkdir(parents=True, exist_ok=True)
    hosting_file = project / '.openai/hosting.json'
    if hosting_file.exists():
        hosting = json.loads(hosting_file.read_text())
        if hosting.get('static', {}).get('directory') != 'dist':
            raise ValueError('Development hosting must use the dist directory')
    else:
        hosting_file.parent.mkdir()
        hosting_file.write_text(json.dumps({'d1': None, 'r2': None,
                                           'static': {'directory': 'dist'}}, indent=2) + '\n')
    dist = project / 'dist'
    dist.mkdir(exist_ok=True)
    entry_files = ('index.html', 'shooting.css', 'shooting.js', 'shooting_results.json',
                   'basic.js', 'predict-position.css')
    files = [(source / 'dev' / name, Path(name)) for name in entry_files]
    media = sorted((source / 'dev/media').glob('shooting_*.webp'))
    if not media or any(not p.is_file() for p, _ in files):
        raise ValueError('The development viewer and real frame atlases must be complete')
    files.extend((p, p.relative_to(source / 'dev')) for p in media)
    prediction = source / 'dev/predict_position_results.json'
    if prediction.exists():
        pp = json.loads(prediction.read_text())
        if pp.get('schema') != 'nanojev-shooting-demo-v1' or not pp.get('cases'):
            raise ValueError('Predict Position must contain complete recorded cases')
        for case in pp['cases']:
            if {s['id'] for s in case['systems']} != {'jev', 'nanojev', 'base'}:
                raise ValueError('Predict Position requires all three recorded models')
        prediction_files = ('predict-position.html',
                            'predict-position.js', 'predict_position_results.json')
        for name in prediction_files:
            original = source / 'dev' / name
            if not original.is_file():
                raise ValueError('Missing Predict Position viewer asset: ' + name)
            files.append((original, Path(name)))
        prediction_media = sorted((source / 'dev/media').glob('predict_position_*.webp'))
        if not prediction_media:
            raise ValueError('Missing real Predict Position frame atlases')
        files.extend((p, p.relative_to(source / 'dev')) for p in prediction_media)
    # Current unified-policy recordings have their own development-only viewer.
    navigation = json.loads((source / 'dev/side_by_side_results.json').read_text())
    checkpoint = data.get('protocol', {}).get('selected_checkpoint_sha256')
    if not checkpoint or navigation.get('protocol', {}).get('selected_checkpoint_sha256') != checkpoint:
        raise ValueError('Navigation and Basic must use the same current checkpoint')
    if prediction.exists() and pp.get('protocol', {}).get('selected_checkpoint_sha256') != checkpoint:
        raise ValueError('All four tasks must use the same current checkpoint')
    files.extend((source / 'dev' / name, Path(name)) for name in (
        'side-by-side.html', 'side-by-side.css', 'side-by-side.js', 'side_by_side_results.json'))
    allowed = {relative.as_posix() for _, relative in files} | {'source_manifest.json'}
    unexpected = [p.relative_to(dist).as_posix() for p in dist.rglob('*')
                  if p.is_file() and p.relative_to(dist).as_posix() not in allowed]
    # Retire only unchanged game atlases owned by the previous staging receipt.
    # Unknown files or locally edited assets are never removed.
    previous_manifest = dist / 'source_manifest.json'
    previous = json.loads(previous_manifest.read_text()) if previous_manifest.exists() else {}
    owned = {row['asset']: row for row in previous.get('files', [])}
    obsolete = []
    for name in unexpected:
        row, path = owned.get(name), dist / name
        owned_prefix = next((prefix for prefix in ('media/shooting_', 'media/predict_position_')
                             if name.startswith(prefix)), None)
        if (not row or not owned_prefix or not name.endswith('.webp')
                or not row.get('source', '').startswith('dev/' + owned_prefix) or path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest() != row.get('sha256')):
            raise ValueError('Unexpected or modified pre-existing development asset: ' + name)
        obsolete.append(path)
    records = []
    for original, relative in files:
        if original.is_symlink():
            raise ValueError('Static assets must be regular files')
        target = dist / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        transforms = []
        if relative.as_posix() == 'side-by-side.html':
            text = target.read_text()
            text = re.sub(r'  <meta property="og:url"[^>]*>\n', '', text)
            text = re.sub(r'  <link rel="canonical"[^>]*>\n', '', text)
            if site_url:
                url = html.escape(site_url.rstrip('/') + '/side-by-side.html', quote=True)
                text = text.replace('</head>', f'  <link rel="canonical" href="{url}">\n</head>')
            text = text.replace('href="https://github.com/TianyuCodings/NanoJev" aria-label="NanoJev repository"',
                                'href="./" aria-label="NanoJev development home"')
            text = text.replace('<span class="wordmark-note">A nano replica of Jev</span>',
                                '<span class="wordmark-note">Development arcade</span>')
            text = text.replace('<span class="recorded"><i aria-hidden="true"></i> REAL GAME REPLAYS</span>',
                                '<a class="recorded" href="./">← Shooting</a>')
            target.write_text(text)
            transforms = ['development_canonical_and_return_navigation']
        elif relative.as_posix() in ('index.html', 'predict-position.html'):
            # A source page lives in web/dev; its independent hosting copy is at root.
            text = target.read_text().replace('../side-by-side.html', 'side-by-side.html')
            target.write_text(text)
            transforms = ['development_root_navigation']
        records.append({'source': str(original.relative_to(source)), 'asset': relative.as_posix(),
                        'bytes': target.stat().st_size,
                        'transforms': transforms,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    for path in obsolete:
        path.unlink()
    (dist / 'source_manifest.json').write_text(json.dumps(
        {'schema': 'nanojev-development-static-v1', 'files': records}, indent=2) + '\n')
    marker.write_text(json.dumps({'role': 'independent_development_viewer',
                                 'site_url': site_url,
                                 'public_site_modified': False}, indent=2) + '\n')
    print(json.dumps({'project': str(project), 'assets': len(records),
                      'static_bytes': sum(r['bytes'] for r in records),
                      'retired_game_atlases': len(obsolete)}))


if __name__ == '__main__':
    main()

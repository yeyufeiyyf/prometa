#!/usr/bin/env python3
"""Download or concatenate verified processed ProMeta artifact volumes."""
from pathlib import Path
import argparse, hashlib, json, urllib.request
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory',type=Path,default=Path('.'))
parser.add_argument('--offline',action='store_true')
args=parser.parse_args();root=args.directory;root.mkdir(parents=True,exist_ok=True)
manifest_path=root/'zenodo_volume_manifest.json'
record_id=23618575
prefix=f'https://zenodo.org/records/{record_id}/files/'
if not manifest_path.exists():
    if args.offline:raise SystemExit('Missing zenodo_volume_manifest.json')
    urllib.request.urlretrieve(prefix+manifest_path.name,manifest_path)
m=json.loads(manifest_path.read_text())
if m['original']['name'] != 'prometa_artifacts_processed_v2.1.0.zip' or m['original']['sha256'] != '98db503094a04af616745fa56b715e309d67f8e601682885c1cb07ab03fa0aa6':
    raise SystemExit('Unexpected archive manifest; use the v2.1.0 processed-release manifest.')
for row in m['parts']:
    p=root/row['name']
    if not p.exists():
        if args.offline:raise SystemExit('Missing '+p.name)
        urllib.request.urlretrieve(prefix+p.name,p)
    assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
out=root/m['original']['name'];h=hashlib.sha256()
with out.open('wb') as f:
    for row in m['parts']:
        with (root/row['name']).open('rb') as part:
            while block:=part.read(1024*1024):f.write(block);h.update(block)
assert out.stat().st_size==m['original']['bytes'] and h.hexdigest()==m['original']['sha256']
print('PASS: restored and verified',out)

"""下载 UCI bank-full 原始文件，不覆盖现有文件；不解压任意 ZIP 路径。"""
from datetime import datetime, timezone
import hashlib
import io
import json
from urllib.request import urlopen
from zipfile import ZipFile
from bank_preprocessing import ROOT, SOURCE

if __name__ == '__main__':
    destination = ROOT / 'data'
    destination.mkdir(exist_ok=True)
    targets = ['bank-full.csv', 'bank-names.txt']
    if any((destination / name).exists() for name in targets):
        raise SystemExit('原始文件已存在，停止以防覆盖；请先核对现有数据。')
    with urlopen(SOURCE, timeout=60) as response:
        archive = response.read()
    with ZipFile(io.BytesIO(archive)) as zipped:
        contents = {name: zipped.read(name) for name in targets}
    for name, content in contents.items():
        with (destination / name).open('xb') as handle:
            handle.write(content)
    provenance = {'url': SOURCE, 'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
                  'archive_sha256': hashlib.sha256(archive).hexdigest(),
                  'files': {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}}
    (destination / 'download_manifest.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('Downloaded bank-full.csv and bank-names.txt to', destination)

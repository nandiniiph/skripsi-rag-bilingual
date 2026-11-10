"""Install code only, back up modified files, leave data/indexes/results intact."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime

parser=argparse.ArgumentParser()
parser.add_argument('--target',required=True,help='Folder proyek, misalnya /home/nandini/skripsi_baru')
args=parser.parse_args()
package=Path(__file__).resolve().parent
target=Path(args.target).expanduser().resolve()
if not (target/'config/config.yaml').is_file() or not (target/'src/retrieval.py').is_file():
    raise SystemExit('Target bukan folder proyek yang diharapkan.')
manifest=json.loads((package/'PATCH_MANIFEST.json').read_text())
changed=manifest['changed_files']
conflicts=[]
for rel in changed:
    old=target/rel
    expected=manifest['original_sha256'].get(rel)
    if old.exists():
        actual=hashlib.sha256(old.read_bytes()).hexdigest()
        new=hashlib.sha256((package/rel).read_bytes()).hexdigest()
        if actual!=expected and actual!=new:conflicts.append(rel)
if conflicts:
    raise SystemExit('Kode target berubah sejak arsip dikirim. Instalasi dihentikan agar perubahan tidak tertimpa:\n'+'\n'.join(conflicts))
backup=target/('backup_code_auto_language_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
backup.mkdir(parents=True,exist_ok=False)
created=[];replaced=[]
try:
    for rel in changed:
        source=package/rel;dest=target/rel
        if dest.exists():
            saved=backup/rel;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,saved);replaced.append(rel)
        else:created.append(rel)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
except Exception:
    for rel in replaced:shutil.copy2(backup/rel,target/rel)
    for rel in created:(target/rel).unlink(missing_ok=True)
    raise
(backup/'installation.json').write_text(json.dumps({'created_files':created,'replaced_files':replaced},indent=2))
print('Kode terpasang:',target)
print('Backup kode lama:',backup)
print('Data, indeks lama, dan hasil lama tidak diubah.')
print('Hasil baru: outputs/auto_language_v1/')
print('Selanjutnya baca README_REVISI.md lalu jalankan 08_check_inputs.py.')

"""把 `boss_capture` 的整屏截图裁成候选图、整理成训练集。

命令行用法：

    .venv/Scripts/python.exe dev_tools/boss_dataset.py crop     # 截图 → 候选图
    .venv/Scripts/python.exe dev_tools/boss_dataset.py build    # 标注 → train/val
    .venv/Scripts/python.exe dev_tools/boss_dataset.py stats    # 看各类数量

网页标注用 `dev_tools/boss_label_server.py`，它直接调用这里的函数，逻辑只有一份。

只做"三个候选里哪个更稀有"的分类，所以标注只填稀有度，不用画框、不用填名字：
1 = UR/SP      2 = SSR      3 = SR      4 = R/N/G（都不值得选）
0 = 跳过（画面没加载完、看不清、不是候选）
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_ROOT = ROOT/'log/hyakkiyakou/boss_capture'
DATASET_ROOT = ROOT/'data/boss_rarity'
# 三个候选槽位，按 1280x720 画面切；界面分辨率变了改这里即可，不用重跑游戏
SLOTS = [(100, 60, 470, 690), (470, 60, 850, 690), (850, 60, 1180, 690)]
LABELS = ['UR_SP', 'SSR', 'SR', 'below']
LABEL_HINT = {'UR_SP': 'UR/SP 级', 'SSR': 'SSR', 'SR': 'SR', 'below': 'R/N/G'}
SKIP = 'skip'


def fingerprint(image):
    """感知指纹：8x8 均值哈希 + 16x16 灰度缩略图（用于二次确认，避免误去重）。

    只用 PIL/numpy，不依赖 cv2 —— 工具链和 ABI 无关，换 numpy 版本也不会挂。
    """
    gray = Image.fromarray(image).convert('L')
    small = np.asarray(gray.resize((8, 8), Image.BILINEAR), dtype=np.float32)
    mean = small.mean()
    bits = ''.join('1' if value > mean else '0' for value in small.flatten())
    thumb = np.asarray(gray.resize((16, 16), Image.BILINEAR), dtype=np.int16)
    return int(bits, 2), thumb


def hamming(left: int, right: int) -> int:
    return bin(left ^ right).count('1')


def is_duplicate(left, right) -> bool:
    """只有感知哈希和像素都几乎一样才算重复。

    同一槽位的背景是固定的，只看哈希会把不同式神误判成重复，所以再用 16x16
    灰度图的平均绝对差确认一次。
    """
    left_hash, left_thumb = left
    right_hash, right_thumb = right
    if hamming(left_hash, right_hash) > 2:
        return False
    return float(abs(left_thumb - right_thumb).mean()) <= 2.0


def crop_dataset(captures=None, out=None, dedupe: bool = True, quiet: bool = False) -> dict:
    """整屏截图 → 候选图 + manifest。返回统计，不抛业务异常。"""
    capture_root = Path(captures or CAPTURE_ROOT)
    dataset = Path(out or DATASET_ROOT)
    crops_dir = dataset/'crops'
    crops_dir.mkdir(parents=True, exist_ok=True)

    runs = sorted([p for p in capture_root.iterdir() if p.is_dir()]) if capture_root.is_dir() else []
    if not runs:
        return {'error': f'没有找到截图：{capture_root}（先跑一局，确认"保存鬼王候选截图"已开启）',
                'runs': 0, 'kept': 0, 'skipped': 0, 'dataset': str(dataset)}

    kept, skipped, fingerprints = [], 0, []
    for run in runs:
        for index, (x0, y0, x1, y1) in enumerate(SLOTS, 1):
            source = run/f'candidate_{index}.jpg'
            if not source.is_file():
                continue
            try:
                image = Image.open(source).convert('RGB')
            except Exception:
                continue
            width, height = image.size
            box = (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            piece = image.crop(box)
            current = fingerprint(np.asarray(piece))
            if dedupe and any(is_duplicate(current, existing) for existing in fingerprints):
                skipped += 1
                continue
            fingerprints.append(current)
            name = f'{run.name}_{index}.png'
            piece.save(crops_dir/name)
            kept.append({'id': name, 'src': f'crops/{name}', 'run': run.name, 'index': index})

    manifest = {'schema_version': 1, 'slots': SLOTS, 'items': kept}
    (dataset/'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    result = {'runs': len(runs), 'kept': len(kept), 'skipped': skipped, 'dataset': str(dataset)}
    if not quiet:
        print(f'截图 {result["runs"]} 局 → 候选图 {result["kept"]} 张（去重跳过 {skipped} 张）')
        print(f'数据集：{dataset.resolve()}')
    return result


def labels_path(out=None) -> Path:
    return Path(out or DATASET_ROOT)/'labels.json'


def load_labels(out=None) -> dict:
    path = labels_path(out)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    data = payload.get('labels', payload) if isinstance(payload, dict) else {}
    return data if isinstance(data, dict) else {}


def save_labels(labels: dict, out=None) -> None:
    path = labels_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'dataset': path.parent.name,
               'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               'labels': labels}
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    temporary.replace(path)


def load_manifest(out=None) -> dict:
    path = Path(out or DATASET_ROOT)/'manifest.json'
    if not path.is_file():
        return {'items': []}
    return json.loads(path.read_text(encoding='utf-8'))


def dataset_stats(out=None) -> dict:
    """实时统计：不依赖 build，直接看 labels.json 与 manifest。"""
    dataset = Path(out or DATASET_ROOT)
    items = load_manifest(dataset).get('items', [])
    labels = load_labels(dataset)
    counts = {label: 0 for label in LABELS}
    labeled = skipped = 0
    for item in items:
        label = labels.get(item['id'])
        if label in LABELS:
            counts[label] += 1
            labeled += 1
        elif label == SKIP:
            skipped += 1
    summary_path = dataset/'summary.json'
    built = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else None
    return {
        'total': len(items),
        'labeled': labeled,
        'unlabeled': len(items) - labeled - skipped,
        'skipped': skipped,
        'counts': counts,
        'built': built,
        'dataset': str(dataset),
    }


def build_dataset(out=None, labels_file=None) -> dict:
    """标注结果 → train/val 文件夹。"""
    dataset = Path(out or DATASET_ROOT)
    labels = load_labels(dataset)
    if labels_file:
        payload = json.loads(Path(labels_file).read_text(encoding='utf-8'))
        labels = payload.get('labels', payload) if isinstance(payload, dict) else {}

    manifest = load_manifest(dataset)
    if not manifest.get('items'):
        return {'error': '缺少 manifest.json，先裁图'}

    for split in ('train', 'val'):
        shutil.rmtree(dataset/split, ignore_errors=True)

    rows, skipped = [], 0
    for item in manifest['items']:
        label = labels.get(item['id'])
        if label in (None, SKIP) or label not in LABELS:
            skipped += 1
            continue
        digest = hashlib.md5(item['id'].encode()).hexdigest()
        split = 'train' if int(digest[:2], 16) < 218 else 'val'  # 约 85% / 15%
        target_dir = dataset/split/label
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset/item['src'], target_dir/item['id'])
        rows.append({'file': f'{split}/{label}/{item["id"]}', 'label': label,
                     'split': split, 'run': item['run']})

    header = 'file,label,split,run\n'
    (dataset/'labels.csv').write_text(
        header + '\n'.join(f"{r['file']},{r['label']},{r['split']},{r['run']}" for r in rows) + '\n',
        encoding='utf-8')

    counts = {}
    for row in rows:
        counts.setdefault(row['split'], {}).setdefault(row['label'], 0)
        counts[row['split']][row['label']] += 1
    summary = {'total': len(rows), 'skipped': skipped, 'counts': counts, 'labels': LABELS}
    (dataset/'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    return summary


def render_label_page_standalone(items: list[dict], dataset_name: str) -> str:
    """纯前端标注页（服务器没启动时也能用，标签只存浏览器本地）。"""
    payload = json.dumps(items, ensure_ascii=False)
    labels = json.dumps(LABELS, ensure_ascii=False)
    hints = json.dumps(LABEL_HINT, ensure_ascii=False)
    return f"""<!doctype html>
<meta charset="utf-8">
<title>鬼王稀有度标注 - {dataset_name}</title>
<style>
 body{{margin:0;font-family:system-ui,Segoe UI,Microsoft YaHei;background:#14161a;color:#e6e6e6}}
 header{{padding:10px 16px;background:#1d2026;display:flex;gap:16px;align-items:center;flex-wrap:wrap}}
 h1{{font-size:16px;margin:0;font-weight:600}}
 #progress{{font-variant-numeric:tabular-nums;color:#9fd}}
 main{{display:flex;gap:20px;padding:16px;align-items:flex-start}}
 #stage{{flex:0 0 auto;background:#0b0c0f;border:1px solid #2a2e36;border-radius:8px;padding:8px}}
 img{{display:block;max-height:64vh}}
 aside{{flex:1 1 auto;min-width:280px}}
 .btn{{display:block;width:100%;margin:8px 0;padding:12px;border-radius:8px;border:1px solid #3a4048;
       background:#232830;color:#e6e6e6;font-size:15px;cursor:pointer;text-align:left}}
 .btn:hover{{background:#2d333d}}
 .btn.cur{{outline:2px solid #4c9}}
 kbd{{background:#111;border:1px solid #444;border-radius:4px;padding:1px 6px;margin-right:8px;font-family:monospace}}
 .row{{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}}
 .small{{color:#9aa4b2;font-size:13px;line-height:1.7}}
 #export{{padding:10px 18px;border-radius:8px;border:0;background:#2f7d4f;color:#fff;font-size:15px;cursor:pointer}}
</style>
<header><h1>鬼王稀有度标注</h1><div id="progress">0 / 0</div>
<div class="small">键盘 <kbd>1</kbd>~<kbd>4</kbd> 打标签，<kbd>0</kbd> 跳过，<kbd>←</kbd><kbd>→</kbd> 翻页</div></header>
<main>
  <div id="stage"><img id="pic" alt="candidate"></div>
  <aside>
    <div id="buttons"></div>
    <div class="row"><button id="export">导出 labels.json</button></div>
    <p class="small" id="done"></p>
    <p class="small">纯离线页：标签只存在浏览器，导出后放回本目录再 build。</p>
  </aside>
</main>
<script>
const ITEMS = {payload}, LABELS = {labels}, HINTS = {hints};
const STORE = 'boss_rarity:{dataset_name}';
let labels = JSON.parse(localStorage.getItem(STORE) || '{{}}'), cursor = 0;
const pic = document.getElementById('pic'), progress = document.getElementById('progress');
LABELS.forEach((label, index) => {{
  const button = document.createElement('button');
  button.className = 'btn'; button.dataset.label = label;
  button.innerHTML = '<kbd>' + (index + 1) + '</kbd>' + label + ' · ' + HINTS[label];
  button.onclick = () => setLabel(label); document.getElementById('buttons').appendChild(button);
}});
function save() {{ localStorage.setItem(STORE, JSON.stringify(labels)); }}
function setLabel(label) {{
  labels[ITEMS[cursor].id] = label; save();
  if (cursor < ITEMS.length - 1) cursor += 1; render();
}}
function move(step) {{ cursor = Math.max(0, Math.min(ITEMS.length - 1, cursor + step)); render(); }}
function render() {{
  const item = ITEMS[cursor]; pic.src = item.src;
  progress.textContent = (cursor + 1) + ' / ' + ITEMS.length + ' · 已标注 ' + Object.keys(labels).length;
  document.querySelectorAll('.btn[data-label]').forEach(b => b.classList.toggle('cur', labels[item.id] === b.dataset.label));
  document.getElementById('done').textContent = labels[item.id] ? ('当前标签：' + labels[item.id]) : '当前未标注';
}}
document.getElementById('export').onclick = () => {{
  const blob = new Blob([JSON.stringify({{dataset: '{dataset_name}', labels: labels}}, null, 1)], {{type: 'application/json'}});
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob);
  link.download = 'labels.json'; link.click();
}};
document.addEventListener('keydown', event => {{
  if (event.key >= '1' && event.key <= String(LABELS.length)) setLabel(LABELS[Number(event.key) - 1]);
  else if (event.key === '0') setLabel('skip');
  else if (event.key === 'ArrowRight') move(1);
  else if (event.key === 'ArrowLeft') move(-1);
}});
render();
</script>
"""


def write_standalone_page(out=None) -> Path:
    dataset = Path(out or DATASET_ROOT)
    items = load_manifest(dataset).get('items', [])
    target = dataset/'label.html'
    target.write_text(render_label_page_standalone(items, dataset.name), encoding='utf-8')
    return target


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)

    crop_parser = subparsers.add_parser('crop', help='整屏截图 → 候选图 + 标注页')
    crop_parser.add_argument('--captures', default=str(CAPTURE_ROOT))
    crop_parser.add_argument('--out', default=str(DATASET_ROOT))
    crop_parser.add_argument('--no-dedupe', dest='dedupe', action='store_false')

    build_parser = subparsers.add_parser('build', help='标注结果 → train/val 目录')
    build_parser.add_argument('--labels', help='（可选）外部 labels.json')
    build_parser.add_argument('--out', default=str(DATASET_ROOT))

    stats_parser = subparsers.add_parser('stats', help='查看各类数量')
    stats_parser.add_argument('--out', default=str(DATASET_ROOT))

    args = parser.parse_args()
    if args.command == 'crop':
        result = crop_dataset(args.captures, args.out, dedupe=args.dedupe)
        if result.get('error'):
            print(result['error'])
            return 1
        print(f'标注页：{write_standalone_page(args.out)}（双击用浏览器打开）')
        return 0
    if args.command == 'build':
        summary = build_dataset(args.out, args.labels)
        if summary.get('error'):
            print(summary['error'])
            return 1
        print(f"已整理 {summary['total']} 张，跳过 {summary['skipped']} 张")
        for split, counts in summary['counts'].items():
            print(f'  {split}: {counts}')
        if summary['total'] < 60:
            print('提示：少于 60 张时模型容易过拟合，建议再跑几局补量。')
        return 0

    stats = dataset_stats(args.out)
    if not stats['total']:
        print('还没有裁图结果，先运行 crop')
        return 1
    print(f"候选图 {stats['total']} 张：已标注 {stats['labeled']}，跳过 {stats['skipped']}，"
          f"未标注 {stats['unlabeled']}")
    print(f"  {stats['counts']}")
    thin = [label for label, count in stats['counts'].items() if count < 20]
    if thin:
        print(f'⚠ 训练样本不足 20 的类别：{thin}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

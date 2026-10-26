"""把 X-AnyLabeling 的标注整理成 YOLO 训练集（支持 YOLO11/YOLOv8）。

三个子命令：

    # 1. 生成一个小类别集（别一上来标 268 类）
    .venv/Scripts/python.exe dev_tools/bean_labels.py classes \
        --labels sp_051,ssr_075,ssr_076,r_007 --out data/bean/classes.txt

    # 2. 标注完（X-AnyLabeling 保存的 json 和 jpg 同名），转成 YOLO
    .venv/Scripts/python.exe dev_tools/bean_labels.py convert \
        --raw data/bean_label/raw --classes data/bean/classes.txt --out data/bean

    # 3. 看统计（哪些框是人工新增的 = 新式神样本）
    .venv/Scripts/python.exe dev_tools/bean_labels.py check --out data/bean

设计说明：

- 类别**只用你指定的子集**，不是 268 类全量。理由：你不可能把没有样本的类都标出来，
  而 YOLO 的类别索引只要训练和推理两端一致就行。
- `convert` 会顺手对比 `<stem>.prelabel.json`（旧模型的原始建议），统计"人工新增的框"，
  这些框最可能对应新式神，是真值密度最高的部分。
- 空的 json（确认无目标）会作为负样本保留，避免模型在空场景里瞎框。
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_table():
    from tasks.Hyakkiyakou.shikigami_table import SHIKIGAMI_TABLE
    return SHIKIGAMI_TABLE


def read_classes(path: Path) -> list[str]:
    labels = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        labels.append(line.split()[0])
    return labels


def box_of(shape) -> tuple[float, float, float, float] | None:
    points = shape.get('points') or []
    if len(points) < 2:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1)*(inter_y2 - inter_y1)
    union = (ax2 - ax1)*(ay2 - ay1) + (bx2 - bx1)*(by2 - by1) - inter
    return inter/union if union > 0 else 0.0


# --------------------------------------------------------------------------- classes
def cmd_classes(args) -> int:
    table = load_table()
    wanted = []
    if args.labels:
        wanted = [item.strip() for item in args.labels.split(',') if item.strip()]
    elif args.names:
        by_name = {row['name']: label for label, row in table.items()}
        for name in [item.strip() for item in args.names.split(',') if item.strip()]:
            if name not in by_name:
                print(f'!! 类别表里没有「{name}」，先用 gen_shikigami_table.py 追加')
                return 1
            wanted.append(by_name[name])
    else:
        print('给 --labels 或 --names')
        return 1

    lines = []
    for label in wanted:
        row = next((r for r in table.values() if r['label'] == label), None)
        if row is None:
            print(f'!! 未知类别 {label}')
            return 1
        lines.append(f"{label} {row['name']}")
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'已写 {len(lines)} 类到 {target}')
    for index, line in enumerate(lines):
        print(f'  {index}: {line}')
    print(f'训练时 data.yaml 的 names 顺序必须和这里一致（{target.name} 的顺序即索引）')
    return 0


# --------------------------------------------------------------------------- convert
def cmd_convert(args) -> int:
    raw = Path(args.raw)
    out = Path(args.out)
    labels = read_classes(Path(args.classes))
    index_of = {label: index for index, label in enumerate(labels)}
    table = load_table()
    name_of = {}
    for row in table.values():
        name_of[row['label']] = row['name']

    images = sorted(raw.glob('*.jpg'))
    if not images:
        print(f'{raw} 里没有 jpg')
        return 1

    for split in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        (out/split).mkdir(parents=True, exist_ok=True)

    stats = {'images': 0, 'boxes': 0, 'manual_boxes': 0, 'skipped_no_json': 0,
             'empty_json': 0, 'classes': Counter(), 'manual_classes': Counter(),
             'manual_frames': []}

    for image in images:
        annotation = image.with_suffix('.json')
        if not annotation.is_file():
            stats['skipped_no_json'] += 1
            continue
        payload = json.loads(annotation.read_text(encoding='utf-8'))
        try:
            from PIL import Image as PILImage
            with PILImage.open(image) as handle:
                width, height = handle.size
        except Exception:
            continue

        pristine_path = image.with_suffix('.prelabel.json')
        pristine = []
        if pristine_path.is_file():
            pristine_payload = json.loads(pristine_path.read_text(encoding='utf-8'))
            pristine = [(shape.get('label'), box_of(shape))
                        for shape in pristine_payload.get('shapes', [])]

        lines, manual = [], 0
        for shape in payload.get('shapes', []):
            label = shape.get('label')
            if label not in index_of:
                continue
            box = box_of(shape)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            x1, x2 = max(0.0, x1), min(float(width), x2)
            y1, y2 = max(0.0, y1), min(float(height), y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            cx = (x1 + x2)/2/width
            cy = (y1 + y2)/2/height
            bw = (x2 - x1)/width
            bh = (y2 - y1)/height
            lines.append(f'{index_of[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
            stats['classes'][label] += 1
            # 人工新增 = 同类旧框里找不到 IoU>0.5 的
            if not any(plabel == label and pbox and iou(box, pbox) > 0.5
                       for plabel, pbox in pristine):
                manual += 1
                stats['manual_classes'][label] += 1
        if not lines and not args.drop_empty:
            stats['empty_json'] += 1  # 空标注当负样本，仍写入空 txt

        digest = hashlib.md5(image.stem.encode()).hexdigest()
        split = 'train' if int(digest[:2], 16) < 218 else 'val'
        (out/'labels'/split/f'{image.stem}.txt').write_text(
            '\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        shutil.copyfile(image, out/'images'/split/image.name)
        stats['images'] += 1
        stats['boxes'] += len(lines)
        stats['manual_boxes'] += manual
        if manual:
            stats['manual_frames'].append({'image': image.name, 'manual': manual})

    names = '\n'.join(f'  {index}: "{label}"' for index, label in enumerate(labels))
    (out/'data.yaml').write_text(
        f'# YOLO11/YOLOv8 数据集；类别顺序与 classes.txt 一致\n'
        f'path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n\n'
        f'names:\n{names}\n', encoding='utf-8')
    (out/'classes.txt').write_text(
        '\n'.join(f'{label} {name_of.get(label, label)}' for label in labels) + '\n',
        encoding='utf-8')

    summary = {
        'images': stats['images'],
        'boxes': stats['boxes'],
        'manual_boxes': stats['manual_boxes'],
        'empty_json': stats['empty_json'],
        'skipped_no_json': stats['skipped_no_json'],
        'per_class': dict(stats['classes']),
        'manual_per_class': dict(stats['manual_classes']),
        'manual_frames': stats['manual_frames'],
    }
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                    encoding='utf-8')

    print(f"导入 {stats['images']} 张图、{stats['boxes']} 个框（跳过无 json {stats['skipped_no_json']} 张）")
    print(f"其中人工新增 {stats['manual_boxes']} 个框，分布在 {len(stats['manual_frames'])} 张图")
    for label, count in stats['classes'].most_common():
        manual = stats['manual_classes'].get(label, 0)
        print(f"  {label:<10} {name_of.get(label,'?'):<8} {count:>4} 框（新增 {manual}）")
    thin = [label for label in labels if stats['classes'].get(label, 0) < 20]
    if thin:
        print(f'⚠ 不足 20 框的类别：{thin}')
    print(f'数据集：{out}（data.yaml 已生成）')
    return 0


# --------------------------------------------------------------------------- check
def cmd_check(args) -> int:
    out = Path(args.out)
    summary = out/'summary.json'
    if not summary.is_file():
        print('还没有转换结果，先跑 convert')
        return 1
    data = json.loads(summary.read_text(encoding='utf-8'))
    print(f"图片 {data['images']}，框 {data['boxes']}，人工新增 {data['manual_boxes']}"
          f"，空标注（负样本）{data['empty_json']}")
    print('每类框数：', data['per_class'])
    if data['manual_per_class']:
        print('人工新增（= 最可能的新式神样本）：', data['manual_per_class'])
    return 0


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    classes = sub.add_parser('classes', help='生成训练用的类别子集')
    classes.add_argument('--labels', help='逗号分隔的类别代号，如 sp_051,ssr_075')
    classes.add_argument('--names', help='逗号分隔的式神名，如 百羽凤凰火,咲耶')
    classes.add_argument('--out', default=str(ROOT/'data/bean/classes.txt'))

    convert = sub.add_parser('convert', help='X-AnyLabeling json → YOLO 数据集')
    convert.add_argument('--raw', default=str(ROOT/'data/bean_label/raw'))
    convert.add_argument('--classes', default=str(ROOT/'data/bean/classes.txt'))
    convert.add_argument('--out', default=str(ROOT/'data/bean'))
    convert.add_argument('--drop-empty', action='store_true', help='不保留空标注（默认保留当负样本）')

    check = sub.add_parser('check', help='看转换结果')
    check.add_argument('--out', default=str(ROOT/'data/bean'))

    args = parser.parse_args()
    return {'classes': cmd_classes, 'convert': cmd_convert, 'check': cmd_check}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())

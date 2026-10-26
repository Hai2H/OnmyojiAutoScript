"""整理撒豆采集：看统计、挑可疑帧、铺成 X-AnyLabeling 能直接打开的目录。

    # 1. 看这 10 局存了什么
    .venv/Scripts/python.exe dev_tools/bean_dataset.py stats

    # 2. 先只导出"可疑帧"（旧模型置信低 / 类别跳变 / 空场景），优先标注这些
    .venv/Scripts/python.exe dev_tools/bean_dataset.py prepare --suspects-only

    # 3. 或者全量导出
    .venv/Scripts/python.exe dev_tools/bean_dataset.py prepare

导出的目录 `data/bean_label/raw/` 里，每张 jpg 旁边有一个同名 json（旧模型预标注），
X-AnyLabeling 打开这个文件夹就能看到框，改完导出 YOLO 即可。
"""

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 直接跑脚本时也能 import tasks.*

CAPTURES = ROOT/'data/bean_capture'
OUT = ROOT/'data/bean_label'
LOW_CONF = 0.60
VERY_LOW_CONF = 0.45


def read_round(folder: Path) -> list[dict]:
    log = folder/'frames.jsonl'
    if not log.is_file():
        return []
    rows = []
    for line in log.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def rounds(captures: Path) -> list[Path]:
    if not captures.is_dir():
        return []
    return sorted([p for p in captures.iterdir() if p.is_dir()])


def analyse(captures: Path) -> dict:
    """逐帧打分：旧模型越不确定、类别越爱跳变，越可能是新式神。"""
    summary = {'rounds': [], 'frames': 0, 'bytes': 0, 'classes': Counter(), 'suspects': []}
    for folder in rounds(captures):
        rows = read_round(folder)
        if not rows:
            continue
        size = sum(f.stat().st_size for f in (folder/'raw').glob('*.jpg')) if (folder/'raw').is_dir() else 0
        last_class = {}
        round_suspects = 0
        for row in rows:
            tracks = row.get('tracks') or []
            flips = 0
            for track in tracks:
                track_id, class_id = track[0], track[1]
                if track_id in last_class and last_class[track_id] != class_id:
                    flips += 1
                last_class[track_id] = class_id
                summary['classes'][class_id] += 1
            confidence = min((t[2] for t in tracks), default=0.0)
            score = 0
            if flips:
                score += 2
            if tracks and confidence < LOW_CONF:
                score += 1
            if tracks and confidence < VERY_LOW_CONF:
                score += 1
            if not tracks:
                score += 0
            row['flips'] = flips
            row['min_conf'] = round(confidence, 3)
            row['score'] = score
            if score:
                round_suspects += 1
                summary['suspects'].append({'round': folder.name, 'frame': row['frame'],
                                            'score': score, 'flips': flips,
                                            'min_conf': row['min_conf']})
        summary['rounds'].append({'round': folder.name, 'frames': len(rows),
                                  'suspects': round_suspects, 'mb': round(size/1024/1024, 1)})
        summary['frames'] += len(rows)
        summary['bytes'] += size
    summary['suspects'].sort(key=lambda item: -item['score'])
    return summary


def cmd_stats(args) -> int:
    data = analyse(Path(args.captures))
    if not data['rounds']:
        print(f'没有采集数据：{args.captures}（先在设置里开「采集整局撒豆画面」跑几局）')
        return 1
    print(f"共 {len(data['rounds'])} 局，{data['frames']} 帧，{data['bytes']/1024/1024:.0f} MB")
    for item in data['rounds']:
        print(f"  {item['round']}  {item['frames']:>5} 帧  {item['mb']:>6} MB  可疑 {item['suspects']}")
    top = data['classes'].most_common(12)
    print('旧模型认出的类别（次数 top12）：')
    from tasks.Hyakkiyakou.shikigami_table import SHIKIGAMI_TABLE
    for class_id, count in top:
        name = SHIKIGAMI_TABLE.get(class_id, {}).get('name', '?')
        label = SHIKIGAMI_TABLE.get(class_id, {}).get('label', '?')
        print(f'  {label:<10} {name:<8} {count}')
    print(f"可疑帧 {len(data['suspects'])} 张（类别跳变 / 置信度低）")
    return 0


def cmd_prepare(args) -> int:
    captures, out = Path(args.captures), Path(args.out)
    data = analyse(captures)
    if not data['rounds']:
        print(f'没有采集数据：{captures}')
        return 1

    suspect_keys = {(item['round'], item['frame']) for item in data['suspects']}
    raw_dir = out/'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    report = {'copied': 0, 'rounds': [], 'origin': {}}
    for folder in rounds(captures):
        rows = read_round(folder)
        if not rows:
            continue
        copied = 0
        for row in rows:
            if args.suspects_only and (folder.name, row['frame']) not in suspect_keys:
                continue
            name = f"{folder.name.replace(' ', '')}_{row['frame']}.jpg"
            source = folder/'raw'/f"{row['frame']}.jpg"
            if not source.is_file():
                continue
            stem = Path(name).stem
            # 图不会被改，可以直接覆盖；json 是标注结果，已经存在就保留（除非 --force）
            if args.force or not (raw_dir/name).exists():
                shutil.copyfile(source, raw_dir/name)
            prelabel = folder/'prelabel'/f"{row['frame']}.json"
            if prelabel.is_file():
                payload = json.loads(prelabel.read_text(encoding='utf-8'))
                payload['imagePath'] = name
                text = json.dumps(payload, ensure_ascii=False, indent=1)
                # 留一份只读原件：X-AnyLabeling 保存时会覆盖 <stem>.json，
                # 之后靠 <stem>.prelabel.json 判断哪些框是人工新增的
                backup = raw_dir/f'{stem}.prelabel.json'
                if not backup.exists():
                    backup.write_text(text, encoding='utf-8')
                editable = raw_dir/f'{stem}.json'
                if args.force or not editable.exists():
                    editable.write_text(text, encoding='utf-8')
            report['origin'][name] = {'round': folder.name, 'frame': row['frame']}
            copied += 1
        if copied:
            report['rounds'].append({'round': folder.name, 'copied': copied})
            report['copied'] += copied

    from tasks.Hyakkiyakou.shikigami_table import SHIKIGAMI_TABLE
    (out/'classes.txt').write_text(
        '\n'.join(f"{row['label']} {row['name']}" for _, row in sorted(SHIKIGAMI_TABLE.items())) + '\n',
        encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')

    print(f"已导出 {report['copied']} 张到 {raw_dir}")
    for item in report['rounds']:
        print(f"  {item['round']}：{item['copied']} 张")
    print(f'类别表：{out/"classes.txt"}')
    print(f'用 X-AnyLabeling 打开：{raw_dir}')
    print('提醒：prelabel 里的框只是旧模型建议，新式神多半没被框到，要手动补。')
    return 0 if report['copied'] else 1


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)

    stats_parser = subparsers.add_parser('stats', help='看采集了多少帧、旧模型认出了什么')
    stats_parser.add_argument('--captures', default=str(CAPTURES))

    prepare_parser = subparsers.add_parser('prepare', help='导出成 X-AnyLabeling 目录')
    prepare_parser.add_argument('--captures', default=str(CAPTURES))
    prepare_parser.add_argument('--out', default=str(OUT))
    prepare_parser.add_argument('--suspects-only', action='store_true',
                                help='只导出可疑帧（类别跳变/置信低），优先标注')
    prepare_parser.add_argument('--force', action='store_true',
                                help='覆盖已存在的 json（会丢掉标注结果，慎用）')
    args = parser.parse_args()
    return cmd_stats(args) if args.command == 'stats' else cmd_prepare(args)


if __name__ == '__main__':
    raise SystemExit(main())

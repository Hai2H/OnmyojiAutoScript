"""生成/追加式神类别表，并导出标注用的类别空间。

oashya 的类别表固定 219 类，模型无法表示之后上线的式神。这个脚本把
`check_shikigami_gap.py` 产出的缺口清单**追加**到本地类别表尾部：

- 旧 id 0~218 保持与 oashya 完全一致，历史标注不会错位；
- 新式神按稀有度分配 `ur_001` / `sp_036` / `ssr_047` / `sr_066` 这类编号；
- 已经存在的条目不会被删除或改名，重复执行只追加新增项（幂等）。

用法：
    .venv/Scripts/python.exe dev_tools/gen_shikigami_table.py            # 生成/追加
    .venv/Scripts/python.exe dev_tools/gen_shikigami_table.py --check    # 只检查
    .venv/Scripts/python.exe dev_tools/gen_shikigami_table.py --gap 其他缺口.json
"""

import argparse
import importlib.util
import json
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT/'tasks/Hyakkiyakou/shikigami_table.py'
DEFAULT_GAP = ROOT/'tasks/Hyakkiyakou/docs/shikigami_gap.json'
LABEL_SPACE_DIR = ROOT/'data/shikigami_dataset/label_space'
PREFIXES = {'UR': 'ur', 'SP': 'sp', 'SSR': 'ssr', 'SR': 'sr', 'R': 'r', 'N': 'n', 'G': 'g', 'BUFF': 'buff'}


def normalize(text: str) -> str:
    return unicodedata.normalize('NFC', text).strip()


def load_table(path: Path) -> list[dict]:
    """读取已有类别表；不存在时用 oashya 的 219 类作为起点。"""
    if path.is_file():
        spec = importlib.util.spec_from_file_location('shikigami_table', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = []
        for item in module.ROWS:
            if isinstance(item, dict):
                rows.append(dict(item))
            else:  # 生成文件里是 (label, name, rarity, released) 元组
                label, name, rarity, released = item
                rows.append({'label': label, 'name': name,
                             'rarity': rarity, 'released': released})
        return rows
    from oashya.labels import CLASSIFY
    return [{'label': item['class'], 'name': item['name'],
             'rarity': 'BUFF' if item['class'].startswith('buff_') else item['class'].split('_')[0].upper(),
             'released': None} for item in CLASSIFY]


def next_index(rows: list[dict], rarity: str) -> int:
    used = [int(row['label'].split('_')[1]) for row in rows if row['label'].startswith(PREFIXES[rarity] + '_')]
    return max(used, default=0) + 1


def append_gap(rows: list[dict], units: list[dict]) -> list[dict]:
    known = {normalize(row['name']) for row in rows}
    added = []
    for unit in units:
        name = normalize(unit['name'])
        if name in known:
            continue
        rarity = unit['rarity']
        if rarity not in PREFIXES or rarity == 'BUFF':
            print(f'  跳过未知稀有度：{rarity} {name}')
            continue
        label = f"{PREFIXES[rarity]}_{next_index(rows, rarity):03d}"
        rows.append({'label': label, 'name': name, 'rarity': rarity,
                     'released': unit.get('date')})
        known.add(name)
        added.append(rows[-1])
    return added


def render(rows: list[dict]) -> str:
    lines = [
        '"""式神类别表：前 219 类与 oashya 完全一致，新式神按稀有度追加在尾部。',
        '',
        '本文件由 dev_tools/gen_shikigami_table.py 生成，id 只追加、不改动，',
        '避免历史标注（YOLO 的 class_id）错位。rarity 为 BUFF 的条目是场上增益，不是式神。',
        '"""',
        '',
        '# (label, name, rarity, released)',
        'ROWS = [',
    ]
    for row in rows:
        released = f"'{row['released']}'" if row.get('released') else 'None'
        lines.append(f"    ('{row['label']}', '{row['name']}', '{row['rarity']}', {released}),")
    lines += [
        ']',
        '',
        'SHIKIGAMI_TABLE = {',
        '    index: {\'label\': label, \'name\': name, \'rarity\': rarity, \'released\': released}',
        '    for index, (label, name, rarity, released) in enumerate(ROWS)',
        '}',
        '',
        '# 稀有度排序：用于"谁更值得砸"和鬼王三选一，与 oashya 的 0~5 保持一致并新增 UR',
        "RARITY_SCORE = {'G': 0, 'N': 1, 'R': 2, 'SR': 3, 'SSR': 4, 'SP': 5, 'UR': 6}",
        '# 模型没有的类别（含新式神）统一落到这个分数，避免"识别不到就崩"',
        "UNKNOWN_RARITY = 'SSR'",
        '',
        '',
        'def rarity_of(class_id: int) -> str | None:',
        '    row = SHIKIGAMI_TABLE.get(class_id)',
        '    return row[\'rarity\'] if row else None',
        '',
        '',
        'def name_of(class_id: int) -> str | None:',
        '    row = SHIKIGAMI_TABLE.get(class_id)',
        '    return row[\'name\'] if row else None',
        '',
        '',
        'def rarity_score(class_id: int, unknown: str = UNKNOWN_RARITY) -> int:',
        '    """模型类别 → 稀有度分；不在表里（新式神/误检）按 unknown 处理。"""',
        '    rarity = rarity_of(class_id)',
        '    return RARITY_SCORE.get(rarity, RARITY_SCORE[unknown])',
        '',
        '',
        '# 按稀有度取全部类别 id。新类别按上线时间追加，所以同一稀有度的 id 不连续，',
        '# 不要用 MIN/MAX 区间判断稀有度，一律查表。',
        'def ids_of(rarity: str) -> list[int]:',
        '    return [index for index, row in SHIKIGAMI_TABLE.items() if row[\'rarity\'] == rarity]',
        '',
    ]
    return '\n'.join(lines)


def export_label_space(rows: list[dict], directory: Path) -> None:
    """导出 X-AnyLabeling 的类别清单与 YOLO 的 data.yaml，供标注/训练使用。"""
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'classes.txt').write_text(
        '\n'.join(row['label'] for row in rows) + '\n', encoding='utf-8')
    (directory/'xany_classes.txt').write_text(
        '\n'.join(f"{row['label']} {row['name']}" for row in rows) + '\n', encoding='utf-8')
    (directory/'classes.json').write_text(json.dumps([
        {'id': index, 'label': row['label'], 'name': row['name'],
         'rarity': row['rarity'], 'released': row.get('released')}
        for index, row in enumerate(rows)
    ], ensure_ascii=False, indent=1), encoding='utf-8')
    names = '\n'.join(f'  {index}: "{row["label"]}"' for index, row in enumerate(rows))
    (directory/'data.yaml').write_text(
        f'# 类别空间 {len(rows)} 类，路径按实际整理结果修改\n'
        f'path: .\ntrain: images/train\nval: images/val\n\nnames:\n{names}\n',
        encoding='utf-8')


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gap', type=Path, default=DEFAULT_GAP, help='缺口清单 JSON')
    parser.add_argument('--check', action='store_true', help='只统计，不写入')
    args = parser.parse_args()

    rows = load_table(TABLE_PATH)
    base_count = len(rows)
    added = []
    if args.gap.is_file():
        units = json.loads(args.gap.read_text(encoding='utf-8')).get('units', [])
        added = append_gap(rows, units)
    else:
        print(f'未找到缺口清单：{args.gap}（只做检查）')

    counts = {}
    for row in rows:
        counts[row['rarity']] = counts.get(row['rarity'], 0) + 1
    print(f'类别表：{base_count} → {len(rows)} 类，新增 {len(added)} 个  {counts}')
    for row in added:
        print(f"  + {row['label']:<10} {row['rarity']:<3} {row['name']}  ({row['released']})")
    if not args.check:
        TABLE_PATH.write_text(render(rows), encoding='utf-8')
        export_label_space(rows, LABEL_SPACE_DIR)
        print(f'已写入 {TABLE_PATH.relative_to(ROOT)}')
        print(f'已写入 {LABEL_SPACE_DIR.relative_to(ROOT)}/{{classes.txt, xany_classes.txt, classes.json, data.yaml}}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Immutable, round-grouped YOLO dataset snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from .catalog import Catalog, _iter_directories, _iter_frames


TRAIN_GATE = 20
ASSIST_GATE = 50
PRIMARY_GATE = 100
MAX_EMPTY_NEGATIVES_PER_ROUND = 10
ELIGIBLE_STATUSES = {'correct', 'changed'}


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def _group_key(event: dict) -> str:
    return f"{event['source_kind']}:{event['round_id']}"


def _gate(count: int) -> str:
    if count >= PRIMARY_GATE:
        return 'primary'
    if count >= ASSIST_GATE:
        return 'assist'
    if count >= TRAIN_GATE:
        return 'train'
    return 'insufficient'


class DatasetBuilder:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def _reviewed_events(self) -> list[dict]:
        placeholders = ', '.join('?' for _ in ELIGIBLE_STATUSES)
        rows = self.catalog.connection.execute(
            f'SELECT * FROM events WHERE status IN ({placeholders}) '
            'ORDER BY round_id, source_kind, track_key',
            tuple(ELIGIBLE_STATUSES),
        ).fetchall()
        return [self.catalog._event_payload(row) for row in rows]

    def readiness(self) -> dict:
        self.catalog._ensure_class_table()
        events = self._reviewed_events()
        by_class: dict[int, list[dict]] = defaultdict(list)
        missing_box = Counter()
        for event in events:
            class_id = event.get('reviewed_class_id')
            if class_id is None:
                continue
            class_id = int(class_id)
            by_class[class_id].append(event)
            if not self._samples_for_event(event):
                missing_box[class_id] += 1

        rows = []
        for class_id, members in by_class.items():
            registry = self.catalog.class_table.get(class_id, {})
            ready = sum(1 for event in members if self._samples_for_event(event))
            rows.append({
                'class_id': class_id,
                'label': registry.get('label'),
                'name': registry.get('name'),
                'rarity': registry.get('rarity'),
                'events': len(members),
                'box_ready_events': ready,
                'missing_box_events': missing_box[class_id],
                'gate': _gate(ready),
                'train_gate': TRAIN_GATE,
            })
        rarity_order = {'UR': 0, 'SP': 1, 'SSR': 2, 'SR': 3, 'R': 4, 'N': 5, 'G': 6, 'BUFF': 7}
        rows.sort(key=lambda row: (
            rarity_order.get(row['rarity'], 99),
            -row['box_ready_events'],
            row['class_id'],
        ))
        groups = {_group_key(event) for event in events}
        selected_default = [row['class_id'] for row in rows if row['gate'] != 'insufficient']
        return {
            'reviewed_events': len(events),
            'round_groups': len(groups),
            'round_groups_detail': sorted(groups),
            'classes': rows,
            'default_classes': selected_default,
            'gates': {
                'train': TRAIN_GATE,
                'assist': ASSIST_GATE,
                'primary': PRIMARY_GATE,
            },
            'can_build': len(groups) >= 2 and bool(rows),
            'warnings': [
                *(['至少需要来自两个不同局的事件才能拆分 train/val。'] if len(groups) < 2 else []),
                *(['当前没有达标类别，继续审核或降低试训门槛。'] if not selected_default else []),
            ],
        }

    def _samples_for_event(self, event: dict) -> list[dict]:
        imported = self.catalog.frame_labels(event['id'])
        class_id = event.get('reviewed_class_id')
        if imported:
            return [
                {
                    'image_index': item['image_index'],
                    'image_path': item['image_path'],
                    'box': item['box'],
                    'class_id': item['class_id'],
                    'label_source': 'xanylabeling',
                }
                for item in imported
                if class_id is not None and item['class_id'] == int(class_id)
            ]
        if class_id is None:
            return []
        for index, representative in enumerate(event.get('representatives') or []):
            box = representative.get('box')
            image_path = representative.get('image_path')
            if box and image_path and Path(image_path).is_file():
                return [{
                    'image_index': index,
                    'image_path': image_path,
                    'box': box,
                    'class_id': int(class_id),
                    'label_source': 'old_track_box',
                }]
        return []

    @staticmethod
    def _split_groups(groups: list[str]) -> tuple[dict[str, str], list[str]]:
        if len(groups) < 2:
            raise ValueError('至少需要来自两个不同局的数据才能拆分 train/val')
        ordered = sorted(groups, key=lambda value: hashlib.md5(value.encode()).hexdigest())
        count = len(ordered)
        warnings = []
        if count >= 10:
            train_end = max(1, round(count * .7))
            val_end = max(train_end + 1, round(count * .9))
            mapping = {}
            for index, group in enumerate(ordered):
                mapping[group] = 'train' if index < train_end else ('val' if index < val_end else 'test')
            return mapping, warnings
        if count >= 3:
            mapping = {
                group: ('val' if index == count - 1 else 'train')
                for index, group in enumerate(ordered)
            }
            warnings.append('局数少于 10，本版本不生成 test；仅拆出独立 val。')
            return mapping, warnings
        warnings.append('只有两局：一局 train、一局 val，不能形成正式上线验收。')
        return {ordered[0]: 'train', ordered[1]: 'val'}, warnings

    def _empty_negatives(self) -> list[dict]:
        result = []
        for folder in _iter_directories(self.catalog.paths.bean_capture):
            empty = []
            for row in _iter_frames(folder / 'frames.jsonl'):
                if row.get('tracks'):
                    continue
                image_path = folder / 'raw' / f"{row.get('frame')}.jpg"
                if image_path.is_file():
                    empty.append({
                        'id': f"negative:bean:{folder.name}:{row.get('frame')}",
                        'source_kind': 'bean',
                        'round_id': folder.name,
                        'frame': row.get('frame'),
                        'image_path': str(image_path.resolve()),
                    })
            result.extend(empty[:MAX_EMPTY_NEGATIVES_PER_ROUND])
        return result

    @staticmethod
    def _clamp_box(box: dict, width: int, height: int) -> dict | None:
        try:
            x1 = max(0.0, min(float(width), float(box['x1'])))
            y1 = max(0.0, min(float(height), float(box['y1'])))
            x2 = max(0.0, min(float(width), float(box['x2'])))
            y2 = max(0.0, min(float(height), float(box['y2'])))
        except (KeyError, TypeError, ValueError):
            return None
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

    @staticmethod
    def _yolo_line(class_index: int, box: dict, width: int, height: int) -> str:
        cx = (box['x1'] + box['x2']) / 2 / width
        cy = (box['y1'] + box['y2']) / 2 / height
        bw = (box['x2'] - box['x1']) / width
        bh = (box['y2'] - box['y1']) / height
        return f'{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}'

    @staticmethod
    def _copy_or_link(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def build(self, class_ids: list[int] | None = None) -> dict:
        readiness = self.readiness()
        selected = [int(item) for item in (class_ids or readiness['default_classes'])]
        if not selected:
            raise ValueError('没有达标类别可训练')
        unknown = [class_id for class_id in selected if class_id not in self.catalog.class_table]
        if unknown:
            raise ValueError(f'未知的 class_id: {unknown}')
        ready_by_class = {
            int(row['class_id']): int(row['box_ready_events'])
            for row in readiness['classes']
        }
        below_gate = [
            class_id for class_id in selected
            if ready_by_class.get(class_id, 0) < TRAIN_GATE
        ]
        if below_gate:
            raise ValueError(
                f'类别未达到每类 {TRAIN_GATE} 个独立事件门槛：{below_gate}'
            )

        events = [
            event for event in self._reviewed_events()
            if event.get('reviewed_class_id') in selected
        ]
        if not events:
            raise ValueError('所选类别没有已审核事件')

        samples = []
        excluded = []
        digest_builder = hashlib.sha256()
        for event in events:
            resolved = self._samples_for_event(event)
            if not resolved:
                excluded.append({
                    'event_id': event['id'],
                    'reason': 'missing_reviewed_box',
                })
                continue
            for sample in resolved:
                source = Path(sample['image_path'])
                if not source.is_file():
                    excluded.append({
                        'event_id': event['id'],
                        'reason': 'missing_image',
                        'image_path': str(source),
                    })
                    continue
                samples.append({
                    **sample,
                    'event_id': event['id'],
                    'source_kind': event['source_kind'],
                    'round_id': event['round_id'],
                    'old_class_id': event.get('old_class_id'),
                    'rarity': self.catalog.class_table[int(sample['class_id'])]['rarity'],
                    'label': self.catalog.class_table[int(sample['class_id'])]['label'],
                })

        if not samples:
            raise ValueError('所选类别没有包含可用图片和矩形框')

        empty_negatives = self._empty_negatives()
        groups = sorted(
            {_group_key(sample) for sample in samples}
            | {_group_key(negative) for negative in empty_negatives}
        )
        split_of_group, warnings = self._split_groups(groups)

        local_index = {class_id: index for index, class_id in enumerate(sorted(selected))}
        dataset_root = self.catalog.paths.datasets_root
        dataset_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        temporary = dataset_root / f'.building-{stamp}-{id(self)}'
        temporary.mkdir(parents=False, exist_ok=False)

        rows = []
        per_split = Counter()
        per_class = Counter()
        per_domain = Counter()
        try:
            for sample in samples:
                source = Path(sample['image_path'])
                try:
                    with Image.open(source) as image:
                        width, height = image.size
                except Exception as exception:
                    excluded.append({
                        'event_id': sample['event_id'],
                        'reason': 'invalid_image',
                        'error': str(exception),
                    })
                    continue
                box = self._clamp_box(sample['box'], width, height)
                if not box:
                    excluded.append({
                        'event_id': sample['event_id'],
                        'reason': 'invalid_box',
                    })
                    continue
                split = split_of_group[_group_key(sample)]
                class_index = local_index[int(sample['class_id'])]
                safe_event = sample['event_id'].replace(':', '_').replace('/', '_')
                filename = f'{safe_event}_{sample["image_index"] + 1:02d}{source.suffix.lower()}'
                image_target = temporary / 'images' / split / filename
                self._copy_or_link(source, image_target)
                label_target = temporary / 'labels' / split / image_target.with_suffix('.txt').name
                label_target.parent.mkdir(parents=True, exist_ok=True)
                label_target.write_text(
                    self._yolo_line(class_index, box, width, height) + '\n',
                    encoding='utf-8',
                )
                image_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                digest_builder.update(
                    f'{sample["event_id"]}|{sample["image_index"]}|{sample["class_id"]}|'
                    f'{json.dumps(box, sort_keys=True)}|{image_hash}\n'.encode()
                )
                rows.append({
                    'event_id': sample['event_id'],
                    'image_index': sample['image_index'],
                    'source_path': str(source),
                    'dataset_image': str(image_target.relative_to(temporary)),
                    'dataset_label': str(label_target.relative_to(temporary)),
                    'class_id': int(sample['class_id']),
                    'class_index': class_index,
                    'label': sample['label'],
                    'rarity': sample['rarity'],
                    'source_kind': sample['source_kind'],
                    'round_id': sample['round_id'],
                    'split': split,
                    'box': box,
                    'label_source': sample['label_source'],
                })
                per_split[split] += 1
                per_class[sample['label']] += 1
                per_domain[sample['source_kind']] += 1

            for negative in empty_negatives:
                split = split_of_group[_group_key(negative)]
                source = Path(negative['image_path'])
                safe_round = negative['round_id'].replace(':', '_').replace('/', '_')
                filename = f'negative_{safe_round}_{negative["frame"]}.jpg'
                image_target = temporary / 'images' / split / filename
                self._copy_or_link(source, image_target)
                label_target = temporary / 'labels' / split / image_target.with_suffix('.txt').name
                label_target.parent.mkdir(parents=True, exist_ok=True)
                label_target.write_text('', encoding='utf-8')
                per_split[split] += 1
                per_domain[negative['source_kind']] += 1

            if not rows:
                raise ValueError('所有候选样本都在图片或框校验时被排除')

            classes = [self.catalog.class_table[class_id] for class_id in sorted(selected)]
            (temporary / 'classes.txt').write_text(
                '\n'.join(f"{row['label']} {row['name']}" for row in classes) + '\n',
                encoding='utf-8',
            )
            digest = digest_builder.hexdigest()[:8]
            version = f'v{stamp}-{digest}'
            target = dataset_root / version
            if target.exists():
                raise ValueError(f'数据集版本已存在，拒绝覆盖：{target.name}')
            names = '\n'.join(
                f'  {index}: "{row["label"]}"' for index, row in enumerate(classes)
            )
            data_yaml = (
                '# Generated by dev_tools/hyakki_workbench/dataset.py\n'
                f'path: {target.as_posix()}\n'
                'train: images/train\n'
                'val: images/val\n'
                + ('test: images/test\n' if per_split.get('test') else '')
                + f'\nnames:\n{names}\n'
            )
            (temporary / 'data.yaml').write_text(data_yaml, encoding='utf-8')
            manifest = {
                'schema_version': 1,
                'version': version,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'content_hash': digest,
                'classes': [
                    {
                        'class_index': index,
                        'class_id': class_id,
                        **self.catalog.class_table[class_id],
                    }
                    for index, class_id in enumerate(sorted(selected))
                ],
                'split_groups': split_of_group,
                'samples': rows,
                'stats': {
                    'images': len(rows) + len(empty_negatives),
                    'positive_images': len(rows),
                    'empty_negative_images': len(empty_negatives),
                    'boxes': len(rows),
                    'per_split': dict(per_split),
                    'per_class': dict(per_class),
                    'per_domain': dict(per_domain),
                },
                'excluded': excluded,
                'warnings': warnings,
                'policy': {
                    'gate_train': TRAIN_GATE,
                    'gate_assist': ASSIST_GATE,
                    'gate_primary': PRIMARY_GATE,
                    'split_by_round': True,
                    'max_empty_negatives_per_round': MAX_EMPTY_NEGATIVES_PER_ROUND,
                },
            }
            (temporary / 'manifest.json').write_text(
                json.dumps(manifest, ensure_ascii=False, indent=1),
                encoding='utf-8',
            )
            temporary.rename(target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        self.catalog.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                version TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                manifest_json TEXT NOT NULL
            )
            """
        )
        self.catalog.connection.execute(
            'INSERT INTO datasets(version, path, created_at, manifest_json) VALUES (?, ?, ?, ?)',
            (version, str(target), manifest['created_at'], _json(manifest)),
        )
        self.catalog.connection.commit()
        return {
            'version': version,
            'path': str(target),
            'manifest': manifest,
        }

    def list_datasets(self) -> list[dict]:
        self.catalog.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                version TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                manifest_json TEXT NOT NULL
            )
            """
        )
        rows = self.catalog.connection.execute(
            'SELECT * FROM datasets ORDER BY created_at DESC, version DESC'
        ).fetchall()
        result = []
        for row in rows:
            try:
                manifest = json.loads(row['manifest_json'])
            except json.JSONDecodeError:
                manifest = {}
            result.append({
                'version': row['version'],
                'path': row['path'],
                'created_at': row['created_at'],
                'stats': manifest.get('stats', {}),
                'classes': manifest.get('classes', []),
            })
        return result

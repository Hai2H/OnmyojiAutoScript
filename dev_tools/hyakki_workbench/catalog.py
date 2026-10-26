"""Index captured Hyakki rounds into reviewable target events."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .paths import WorkspacePaths


HIGH_RARITIES = {'SSR', 'SP', 'UR'}
LOW_CONFIDENCE = 0.60
REPRESENTATIVE_LIMIT = 3


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return default


def _iter_directories(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    return sorted(item for item in root.iterdir() if item.is_dir())


def _iter_frames(path: Path) -> Iterable[dict]:
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _track_box(track: list | tuple) -> dict[str, float]:
    cx, cy, width, height = (float(track[3]), float(track[4]),
                             float(track[5]), float(track[6]))
    return {
        'x1': round(cx - width / 2, 2),
        'y1': round(cy - height / 2, 2),
        'x2': round(cx + width / 2, 2),
        'y2': round(cy + height / 2, 2),
    }


def _box_area(track: list | tuple) -> float:
    return float(track[5]) * float(track[6])


@dataclass(frozen=True)
class ScanResult:
    bean_rounds: int
    boss_rounds: int
    events_seen: int
    events_upserted: int
    high_rarity_events: int
    suspicious_events: int
    scanned_at: str

    def as_dict(self) -> dict:
        return {
            'bean_rounds': self.bean_rounds,
            'boss_rounds': self.boss_rounds,
            'events_seen': self.events_seen,
            'events_upserted': self.events_upserted,
            'high_rarity_events': self.high_rarity_events,
            'suspicious_events': self.suspicious_events,
            'scanned_at': self.scanned_at,
        }


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS scan_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scanned_at TEXT NOT NULL,
    bean_rounds INTEGER NOT NULL,
    boss_rounds INTEGER NOT NULL,
    events_seen INTEGER NOT NULL,
    events_upserted INTEGER NOT NULL,
    result_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    source_kind TEXT NOT NULL,
    round_id TEXT NOT NULL,
    track_key TEXT NOT NULL,
    old_class_id INTEGER,
    old_label TEXT,
    old_name TEXT,
    rarity TEXT,
    confidence REAL,
    first_frame TEXT,
    last_frame TEXT,
    frame_count INTEGER NOT NULL DEFAULT 1,
    image_path TEXT NOT NULL,
    box_json TEXT,
    representatives_json TEXT NOT NULL,
    suspicion TEXT NOT NULL DEFAULT '[]',
    needs_review INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'pending',
    reviewed_class_id INTEGER,
    reviewed_label TEXT,
    review_note TEXT,
    reviewed_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_review
    ON events(status, needs_review, rarity, source_kind);
CREATE INDEX IF NOT EXISTS idx_events_round ON events(round_id);
CREATE INDEX IF NOT EXISTS idx_events_updated ON events(updated_at);

CREATE TABLE IF NOT EXISTS review_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    action TEXT NOT NULL,
    previous_status TEXT,
    new_status TEXT NOT NULL,
    old_class_id INTEGER,
    reviewed_class_id INTEGER,
    reviewed_label TEXT,
    note TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_history_event
    ON review_history(event_id, id DESC);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_frame_labels (
    event_id TEXT NOT NULL,
    image_index INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    class_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    box_json TEXT NOT NULL,
    source TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    PRIMARY KEY(event_id, image_index)
);

CREATE INDEX IF NOT EXISTS idx_frame_labels_event
    ON event_frame_labels(event_id);
"""


class Catalog:
    """Scanning and query layer shared by the API and dataset builder."""

    def __init__(self, paths: WorkspacePaths, connection: sqlite3.Connection):
        self.paths = paths
        self.connection = connection
        self.class_table: dict[int, dict] = {}
        self.label_to_class: dict[str, int] = {}

    @classmethod
    def connect(cls, paths: WorkspacePaths) -> 'Catalog':
        paths.ensure()
        connection = sqlite3.connect(paths.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA)
        return cls(paths, connection)

    def close(self) -> None:
        self.connection.close()

    def scan(self) -> ScanResult:
        self._load_class_table()
        bean_events, bean_rounds = self._scan_bean()
        boss_events, boss_rounds = self._scan_boss()
        events = bean_events + boss_events
        now = _now()
        upserted = 0
        high_rarity = 0
        suspicious = 0
        for event in events:
            existing = self.connection.execute(
                'SELECT status, reviewed_class_id, reviewed_label, review_note, reviewed_at '
                'FROM events WHERE id = ?',
                (event['id'],),
            ).fetchone()
            if existing:
                event.update({
                    'status': existing['status'],
                    'reviewed_class_id': existing['reviewed_class_id'],
                    'reviewed_label': existing['reviewed_label'],
                    'review_note': existing['review_note'],
                    'reviewed_at': existing['reviewed_at'],
                })
            self._upsert_event(event, now)
            upserted += 1
            if event['rarity'] in HIGH_RARITIES:
                high_rarity += 1
            if event['suspicion']:
                suspicious += 1
        result = ScanResult(
            bean_rounds=bean_rounds,
            boss_rounds=boss_rounds,
            events_seen=len(events),
            events_upserted=upserted,
            high_rarity_events=high_rarity,
            suspicious_events=suspicious,
            scanned_at=now,
        )
        self.connection.execute(
            'INSERT INTO scan_runs '
            '(scanned_at, bean_rounds, boss_rounds, events_seen, events_upserted, result_json) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (now, bean_rounds, boss_rounds, len(events), upserted, _json(result.as_dict())),
        )
        self.connection.commit()
        return result

    def _load_class_table(self) -> None:
        from tasks.Hyakkiyakou.shikigami_table import SHIKIGAMI_TABLE

        self.class_table = SHIKIGAMI_TABLE
        self.label_to_class = {
            row['label']: class_id for class_id, row in SHIKIGAMI_TABLE.items()
        }

    def _ensure_class_table(self) -> None:
        if not self.class_table:
            self._load_class_table()

    def _row_for(self, class_id: int | None) -> dict:
        if class_id is None:
            return {}
        return self.class_table.get(int(class_id), {})

    def _event_from_track(
        self,
        *,
        round_id: str,
        track_id: int,
        history: list[dict],
    ) -> dict:
        classes = [int(item['track'][1]) for item in history]
        class_counts = Counter(classes)
        class_id = class_counts.most_common(1)[0][0]
        row = self._row_for(class_id)

        flips = sum(left != right for left, right in zip(classes, classes[1:]))
        lowest_confidence = min(float(item['track'][2]) for item in history)
        suspicion = []
        if flips:
            suspicion.append('class_flip')
        if lowest_confidence < LOW_CONFIDENCE:
            suspicion.append('low_confidence')
        if row.get('rarity') in HIGH_RARITIES:
            suspicion.append('high_rarity')
        if int(class_id) > 218:
            suspicion.append('unsupported_by_old_model')

        ranked = sorted(
            history,
            key=lambda item: (
                -int(item['track'][1] == class_id),
                -float(item['track'][2]),
                -_box_area(item['track']),
            ),
        )
        representatives = []
        seen_frames = set()
        for item in ranked:
            frame = item['frame']
            if frame in seen_frames:
                continue
            representatives.append({
                'frame': frame,
                'captured_at': item.get('captured_at'),
                'image_path': item['image_path'],
                'box': _track_box(item['track']),
            })
            seen_frames.add(frame)
            if len(representatives) >= REPRESENTATIVE_LIMIT:
                break

        best = ranked[0]
        metadata = {
            'track_id': track_id,
            'phase': best.get('phase'),
            'class_counts': dict(class_counts),
            'min_confidence': round(lowest_confidence, 4),
            'class_flips': flips,
        }
        return {
            'id': f'bean:{round_id}:{track_id}',
            'source_kind': 'bean',
            'round_id': round_id,
            'track_key': str(track_id),
            'old_class_id': class_id,
            'old_label': row.get('label'),
            'old_name': row.get('name'),
            'rarity': row.get('rarity'),
            'confidence': float(best['track'][2]),
            'first_frame': min(item['frame'] for item in history),
            'last_frame': max(item['frame'] for item in history),
            'frame_count': len(history),
            'image_path': str(Path(best['image_path']).resolve()),
            'box_json': _json(_track_box(best['track'])),
            'representatives_json': _json(representatives),
            'suspicion': suspicion,
            'needs_review': 1,
            'metadata_json': _json(metadata),
        }

    def _scan_bean(self) -> tuple[list[dict], int]:
        events = []
        rounds = list(_iter_directories(self.paths.bean_capture))
        for folder in rounds:
            history_by_track: dict[int, list[dict]] = defaultdict(list)
            for row in _iter_frames(folder / 'frames.jsonl'):
                image_path = folder / 'raw' / f"{row.get('frame')}.jpg"
                if not image_path.is_file():
                    continue
                for track in row.get('tracks') or []:
                    if not isinstance(track, (list, tuple)) or len(track) < 8:
                        continue
                    history_by_track[int(track[0])].append({
                        'frame': str(row.get('frame')),
                        'captured_at': row.get('captured_at'),
                        'phase': row.get('phase'),
                        'image_path': str(image_path),
                        'track': track,
                    })
            for track_id, history in history_by_track.items():
                if history:
                    events.append(self._event_from_track(
                        round_id=folder.name,
                        track_id=track_id,
                        history=history,
                    ))
        return events, len(rounds)

    def _scan_boss(self) -> tuple[list[dict], int]:
        events = []
        rounds = list(_iter_directories(self.paths.boss_capture))
        for folder in rounds:
            meta = _read_json(folder / 'meta.json', {}) or {}
            entries = meta.get('entries') or []
            by_index = {
                int(entry.get('index')): entry
                for entry in entries
                if entry.get('index') is not None
            }
            for index in range(1, 4):
                image_path = folder / f'candidate_{index}.jpg'
                if not image_path.is_file():
                    continue
                entry = by_index.get(index, {})
                class_id = entry.get('old_class_id')
                class_id = int(class_id) if class_id is not None else None
                row = self._row_for(class_id)
                suspicion = []
                if not row:
                    suspicion.append('unknown_identity')
                elif row.get('rarity') in HIGH_RARITIES:
                    suspicion.append('high_rarity')
                event = {
                    'id': f'boss:{folder.name}:{index}',
                    'source_kind': 'boss',
                    'round_id': folder.name,
                    'track_key': str(index),
                    'old_class_id': class_id,
                    'old_label': row.get('label'),
                    'old_name': entry.get('old_name') or row.get('name'),
                    'rarity': row.get('rarity'),
                    'confidence': None,
                    'first_frame': f'candidate_{index}',
                    'last_frame': f'candidate_{index}',
                    'frame_count': 1,
                    'image_path': str(image_path.resolve()),
                    'box_json': None,
                    'representatives_json': _json([{
                        'frame': f'candidate_{index}',
                        'image_path': str(image_path.resolve()),
                        'box': None,
                    }]),
                    'suspicion': suspicion,
                    'needs_review': 1,
                    'metadata_json': _json({
                        'candidate_index': index,
                        'old_score': entry.get('old_score'),
                        'old_raw': entry.get('old_raw'),
                        'round_status': meta.get('status'),
                    }),
                }
                events.append(event)
        return events, len(rounds)

    def _upsert_event(self, event: dict, updated_at: str) -> None:
        self.connection.execute(
            """
            INSERT INTO events (
                id, source_kind, round_id, track_key, old_class_id, old_label,
                old_name, rarity, confidence, first_frame, last_frame, frame_count,
                image_path, box_json, representatives_json, suspicion, needs_review,
                status, reviewed_class_id, reviewed_label, review_note, reviewed_at,
                metadata_json, updated_at
            ) VALUES (
                :id, :source_kind, :round_id, :track_key, :old_class_id, :old_label,
                :old_name, :rarity, :confidence, :first_frame, :last_frame, :frame_count,
                :image_path, :box_json, :representatives_json,
                :suspicion_json, :needs_review, :status,
                :reviewed_class_id, :reviewed_label, :review_note, :reviewed_at,
                :metadata_json, :updated_at
            )
            ON CONFLICT(id) DO UPDATE SET
                source_kind=excluded.source_kind,
                round_id=excluded.round_id,
                track_key=excluded.track_key,
                old_class_id=excluded.old_class_id,
                old_label=excluded.old_label,
                old_name=excluded.old_name,
                rarity=excluded.rarity,
                confidence=excluded.confidence,
                first_frame=excluded.first_frame,
                last_frame=excluded.last_frame,
                frame_count=excluded.frame_count,
                image_path=excluded.image_path,
                box_json=excluded.box_json,
                representatives_json=excluded.representatives_json,
                suspicion=excluded.suspicion,
                needs_review=excluded.needs_review,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            {
                **event,
                'suspicion_json': _json(event['suspicion']),
                'updated_at': updated_at,
                'status': event.get('status', 'pending'),
                'reviewed_class_id': event.get('reviewed_class_id'),
                'reviewed_label': event.get('reviewed_label'),
                'review_note': event.get('review_note'),
                'reviewed_at': event.get('reviewed_at'),
            },
        )

    def summary(self) -> dict:
        status = {
            row['status']: row['count']
            for row in self.connection.execute(
                'SELECT status, COUNT(*) AS count FROM events GROUP BY status'
            )
        }
        rarity = {
            row['rarity'] or 'UNKNOWN': row['count']
            for row in self.connection.execute(
                'SELECT rarity, COUNT(*) AS count FROM events GROUP BY rarity'
            )
        }
        sources = {
            row['source_kind']: row['count']
            for row in self.connection.execute(
                'SELECT source_kind, COUNT(*) AS count FROM events GROUP BY source_kind'
            )
        }
        pending_high = self.connection.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE status IN ('pending', 'needs_label') AND rarity IN ('SSR', 'SP', 'UR')"
        ).fetchone()[0]
        scanned = self.connection.execute(
            'SELECT scanned_at FROM scan_runs ORDER BY id DESC LIMIT 1'
        ).fetchone()
        rounds = {
            row['source_kind']: row['count']
            for row in self.connection.execute(
                'SELECT source_kind, COUNT(DISTINCT round_id) AS count '
                'FROM events GROUP BY source_kind'
            )
        }
        return {
            'events': sum(status.values()),
            'status': status,
            'rarity': rarity,
            'sources': sources,
            'rounds': rounds,
            'pending_high_rarity': pending_high,
            'last_scan': scanned['scanned_at'] if scanned else None,
            'database': str(self.paths.database),
            'bean_capture': str(self.paths.bean_capture),
            'boss_capture': str(self.paths.boss_capture),
        }

    @staticmethod
    def _event_payload(row: sqlite3.Row) -> dict:
        payload = dict(row)
        decoders = {
            'box_json': ('box', None),
            'representatives_json': ('representatives', []),
            'metadata_json': ('metadata', {}),
        }
        for key, (target, fallback) in decoders.items():
            raw = payload.pop(key, None)
            try:
                payload[target] = json.loads(raw) if raw else fallback
            except json.JSONDecodeError:
                payload[target] = fallback
        try:
            payload['suspicion'] = json.loads(payload.get('suspicion') or '[]')
        except json.JSONDecodeError:
            payload['suspicion'] = []
        payload['needs_review'] = bool(payload.get('needs_review'))
        payload['image_url'] = f"/api/events/{payload['id']}/image/0"
        return payload

    def list_events(
        self,
        *,
        status: str | None = None,
        source: str | None = None,
        rarity: str | None = None,
        pending_high_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        clauses = []
        values: list[Any] = []
        if status:
            statuses = [item.strip() for item in status.split(',') if item.strip()]
            if len(statuses) == 1:
                clauses.append('status = ?')
                values.append(statuses[0])
            elif statuses:
                placeholders = ', '.join('?' for _ in statuses)
                clauses.append(f'status IN ({placeholders})')
                values.extend(statuses)
        if source:
            clauses.append('source_kind = ?')
            values.append(source)
        if rarity:
            clauses.append('rarity = ?')
            values.append(rarity)
        if pending_high_only:
            clauses.append("status = 'pending' AND rarity IN ('SSR', 'SP', 'UR')")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ''
        total = self.connection.execute(
            f'SELECT COUNT(*) FROM events {where}', values
        ).fetchone()[0]
        rows = self.connection.execute(
            f'SELECT * FROM events {where} '
            "ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, "
            "CASE rarity WHEN 'UR' THEN 0 WHEN 'SP' THEN 1 WHEN 'SSR' THEN 2 ELSE 3 END, "
            'updated_at DESC LIMIT ? OFFSET ?',
            [*values, max(1, min(int(limit), 500)), max(0, int(offset))],
        ).fetchall()
        return {
            'total': total,
            'limit': limit,
            'offset': offset,
            'items': [self._event_payload(row) for row in rows],
        }

    def get_event(self, event_id: str) -> dict | None:
        row = self.connection.execute(
            'SELECT * FROM events WHERE id = ?', (event_id,)
        ).fetchone()
        return self._event_payload(row) if row else None

    def list_classes(
        self,
        *,
        query: str = '',
        rarity: str | None = None,
        limit: int = 300,
    ) -> list[dict]:
        self._ensure_class_table()
        needle = query.strip().lower()
        rows = []
        for class_id, row in self.class_table.items():
            if rarity and row['rarity'] != rarity:
                continue
            haystack = f"{row['label']} {row['name']} {row['rarity']}".lower()
            if needle and needle not in haystack:
                continue
            rows.append({
                'class_id': class_id,
                'label': row['label'],
                'name': row['name'],
                'rarity': row['rarity'],
                'released': row['released'],
            })
        rarity_order = {'UR': 0, 'SP': 1, 'SSR': 2, 'SR': 3, 'R': 4, 'N': 5, 'G': 6, 'BUFF': 7}
        rows.sort(key=lambda item: (
            rarity_order.get(item['rarity'], 99),
            item['released'] or '',
            item['class_id'],
        ))
        return rows[:max(1, min(int(limit), 500))]

    def review_event(
        self,
        event_id: str,
        *,
        action: str,
        class_id: int | None = None,
        note: str = '',
    ) -> dict:
        self._ensure_class_table()
        event = self.get_event(event_id)
        if not event:
            raise KeyError(event_id)
        action = action.strip().lower()
        previous_status = event['status']
        reviewed_class_id = event.get('reviewed_class_id')
        reviewed_label = event.get('reviewed_label')
        status = previous_status

        if action == 'correct':
            reviewed_class_id = event.get('old_class_id')
            if reviewed_class_id is None:
                raise ValueError('旧模型没有给出类别，请选择具体式神或标记 UNKNOWN')
            row = self.class_table.get(int(reviewed_class_id))
            if not row:
                raise ValueError('旧类别不在 268 类登记表中，请手动选择类别')
            reviewed_label = row['label']
            status = 'correct'
        elif action == 'change':
            if class_id is None:
                raise ValueError('改类别时必须提供 class_id')
            row = self.class_table.get(int(class_id))
            if not row:
                raise ValueError('未知的 class_id')
            reviewed_class_id = int(class_id)
            reviewed_label = row['label']
            status = 'changed'
        elif action == 'unknown':
            reviewed_class_id = None
            reviewed_label = 'UNKNOWN'
            status = 'unknown'
        elif action == 'needs_label':
            reviewed_class_id = None
            reviewed_label = 'UNKNOWN'
            status = 'needs_label'
        elif action == 'skip':
            status = 'skipped'
        elif action == 'reopen':
            reviewed_class_id = None
            reviewed_label = None
            status = 'pending'
        else:
            raise ValueError('不支持的审核动作')

        now = _now()
        self.connection.execute(
            """
            UPDATE events SET
                status = ?, reviewed_class_id = ?, reviewed_label = ?,
                review_note = ?, reviewed_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                status,
                reviewed_class_id,
                reviewed_label,
                note.strip() or None,
                None if status == 'pending' else now,
                now,
                event_id,
            ),
        )
        self.connection.execute(
            """
            INSERT INTO review_history (
                event_id, action, previous_status, new_status, old_class_id,
                reviewed_class_id, reviewed_label, note, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                action,
                previous_status,
                status,
                event.get('old_class_id'),
                reviewed_class_id,
                reviewed_label,
                note.strip() or None,
                now,
            ),
        )
        self.connection.commit()
        return self.get_event(event_id)

    def review_history(self, event_id: str) -> list[dict]:
        return [
            dict(row)
            for row in self.connection.execute(
                'SELECT * FROM review_history WHERE event_id = ? ORDER BY id DESC',
                (event_id,),
            ).fetchall()
        ]

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.connection.execute(
            'SELECT value FROM settings WHERE key = ?', (key,)
        ).fetchone()
        return row['value'] if row else default

    def set_setting(self, key: str, value: str) -> None:
        self.connection.execute(
            """
            INSERT INTO settings(key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, value, _now()),
        )
        self.connection.commit()

    def prepare_label_package(self, event_id: str) -> dict:
        """Create a stable X-AnyLabeling folder without replacing reviewed JSON."""
        self._ensure_class_table()
        event = self.get_event(event_id)
        if not event:
            raise KeyError(event_id)
        slug = event_id.replace(':', '_').replace('/', '_')
        target = self.paths.state_root / 'review' / slug
        target.mkdir(parents=True, exist_ok=True)
        (target / 'classes.txt').write_text(
            '\n'.join(
                f"{row['label']} {row['name']}"
                for _, row in sorted(self.class_table.items())
            ) + '\n',
            encoding='utf-8',
        )
        images = event.get('representatives') or []
        files = []
        for index, representative in enumerate(images):
            source = Path(representative['image_path'])
            suffix = source.suffix.lower() or '.jpg'
            image_name = f'{index + 1:02d}_{source.stem}{suffix}'
            image_target = target / image_name
            if not image_target.exists():
                try:
                    os.link(source, image_target)
                except OSError:
                    shutil.copy2(source, image_target)
            annotation_path = image_target.with_suffix('.json')
            if not annotation_path.exists():
                box = representative.get('box')
                label = event.get('reviewed_label') or event.get('old_label') or 'UNKNOWN'
                shapes = []
                if box:
                    shapes.append({
                        'label': label,
                        'points': [[box['x1'], box['y1']], [box['x2'], box['y2']]],
                        'group_id': None,
                        'shape_type': 'rectangle',
                        'flags': {'source': 'old_model'},
                        'description': '旧模型预标注，必须人工确认',
                    })
                payload = {
                    'version': '4.1.1',
                    'flags': {
                        'model_suggestion': True,
                        'reviewed': False,
                        'workbench_event': event_id,
                    },
                    'shapes': shapes,
                    'imagePath': image_name,
                    'imageData': None,
                    'imageHeight': None,
                    'imageWidth': None,
                }
                annotation_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=1),
                    encoding='utf-8',
                )
            files.append({
                'image': str(image_target),
                'annotation': str(annotation_path),
            })
        (target / 'event.json').write_text(
            json.dumps(event, ensure_ascii=False, indent=1),
            encoding='utf-8',
        )
        return {
            'folder': str(target),
            'classes': str(target / 'classes.txt'),
            'files': files,
            'xanylabeling_url': 'https://github.com/CVHub520/X-AnyLabeling',
        }

    @staticmethod
    def _shape_box(shape: dict) -> dict | None:
        points = shape.get('points') or []
        if len(points) < 2:
            return None
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        box = {
            'x1': min(xs),
            'y1': min(ys),
            'x2': max(xs),
            'y2': max(ys),
        }
        if box['x2'] <= box['x1'] or box['y2'] <= box['y1']:
            return None
        return {key: round(value, 2) for key, value in box.items()}

    @staticmethod
    def _iou(left: dict, right: dict) -> float:
        x1 = max(left['x1'], right['x1'])
        y1 = max(left['y1'], right['y1'])
        x2 = min(left['x2'], right['x2'])
        y2 = min(left['y2'], right['y2'])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        left_area = (left['x2'] - left['x1']) * (left['y2'] - left['y1'])
        right_area = (right['x2'] - right['x1']) * (right['y2'] - right['y1'])
        return intersection / max(1e-6, left_area + right_area - intersection)

    def import_label_package(self, event_id: str) -> dict:
        """Import reviewed X-AnyLabeling rectangles back into the event index."""
        self._ensure_class_table()
        event = self.get_event(event_id)
        if not event:
            raise KeyError(event_id)
        slug = event_id.replace(':', '_').replace('/', '_')
        folder = self.paths.state_root / 'review' / slug
        if not folder.is_dir():
            raise ValueError('尚未生成 X-AnyLabeling 标注目录')

        imported = 0
        skipped = 0
        chosen_classes: Counter[int] = Counter()
        preferred_class_id = event.get('reviewed_class_id')
        preferred_label = event.get('reviewed_label')
        if preferred_label == 'UNKNOWN':
            preferred_label = None

        for image_index, representative in enumerate(event.get('representatives') or []):
            candidates = sorted(folder.glob(f'{image_index + 1:02d}_*.json'))
            if not candidates:
                skipped += 1
                continue
            payload = _read_json(candidates[0], {}) or {}
            shapes = payload.get('shapes') or []
            resolved = []
            for shape in shapes:
                label = shape.get('label')
                class_id = self.label_to_class.get(label)
                box = self._shape_box(shape)
                if class_id is None or box is None:
                    continue
                resolved.append((shape, int(class_id), box))
            if not resolved:
                skipped += 1
                continue

            preferred = [
                item for item in resolved
                if (preferred_class_id is not None and item[1] == int(preferred_class_id))
                or (preferred_label and item[0].get('label') == preferred_label)
            ]
            pool = preferred or resolved
            old_box = representative.get('box')
            if old_box:
                _, class_id, box = max(pool, key=lambda item: self._iou(old_box, item[2]))
            else:
                _, class_id, box = max(
                    pool,
                    key=lambda item: (item[2]['x2'] - item[2]['x1']) * (item[2]['y2'] - item[2]['y1']),
                )
            row = self.class_table[class_id]
            image_name = payload.get('imagePath') or f'{image_index + 1:02d}'
            image_path = folder / image_name
            if not image_path.is_file():
                matches = sorted(folder.glob(f'{image_index + 1:02d}_*.jpg'))
                if not matches:
                    skipped += 1
                    continue
                image_path = matches[0]
            self.connection.execute(
                """
                INSERT INTO event_frame_labels (
                    event_id, image_index, image_path, class_id, label,
                    box_json, source, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id, image_index) DO UPDATE SET
                    image_path=excluded.image_path,
                    class_id=excluded.class_id,
                    label=excluded.label,
                    box_json=excluded.box_json,
                    source=excluded.source,
                    imported_at=excluded.imported_at
                """,
                (
                    event_id,
                    image_index,
                    str(image_path.resolve()),
                    class_id,
                    row['label'],
                    _json(box),
                    'xanylabeling',
                    _now(),
                ),
            )
            imported += 1
            chosen_classes[class_id] += 1

        if not imported:
            raise ValueError('标注目录中还没有可用类别和矩形框')
        chosen_class_id = chosen_classes.most_common(1)[0][0]
        chosen_row = self.class_table[chosen_class_id]
        status = 'correct' if chosen_class_id == event.get('old_class_id') else 'changed'
        now = _now()
        self.connection.execute(
            """
            UPDATE events SET status=?, reviewed_class_id=?, reviewed_label=?,
                reviewed_at=?, updated_at=?
            WHERE id=?
            """,
            (status, chosen_class_id, chosen_row['label'], now, now, event_id),
        )
        self.connection.execute(
            """
            INSERT INTO review_history (
                event_id, action, previous_status, new_status, old_class_id,
                reviewed_class_id, reviewed_label, note, created_at
            ) VALUES (?, 'import_xany', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event['status'],
                status,
                event.get('old_class_id'),
                chosen_class_id,
                chosen_row['label'],
                f'导入 {imported} 张人工标注',
                now,
            ),
        )
        self.connection.commit()
        return {
            'event': self.get_event(event_id),
            'imported': imported,
            'skipped': skipped,
        }

    def frame_labels(self, event_id: str) -> list[dict]:
        rows = self.connection.execute(
            'SELECT * FROM event_frame_labels WHERE event_id = ? ORDER BY image_index',
            (event_id,),
        ).fetchall()
        result = []
        for row in rows:
            payload = dict(row)
            try:
                payload['box'] = json.loads(payload.pop('box_json'))
            except json.JSONDecodeError:
                continue
            result.append(payload)
        return result

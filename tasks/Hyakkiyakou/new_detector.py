"""Runtime adapter that keeps the old tracker authoritative while adding YOLO11.

The YOLO model only produces detections. This module owns tracking, matching and
the four rollout modes used by the workbench.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


LOGGER = logging.getLogger(__name__)
VALID_MODES = {'auto', 'old', 'shadow', 'assist', 'primary'}


def _xyxy(track: list | tuple) -> tuple[float, float, float, float]:
    cx, cy, width, height = map(float, track[3:7])
    return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2


def box_iou(left, right) -> float:
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
    x1, y1 = max(lx1, rx1), max(ly1, ry1)
    x2, y2 = min(lx2, rx2), min(ly2, ry2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    return intersection / max(1e-9, left_area + right_area - intersection)


def merge_tracks(old_tracks: list[tuple], new_tracks: list[tuple], mode: str) -> list[tuple]:
    """Combine old and new tracks with the agreed conflict policy."""
    mode = mode.lower()
    if mode == 'old':
        return old_tracks
    if mode in {'shadow', 'assist'}:
        if mode == 'shadow':
            return old_tracks
        additions = [
            new for new in new_tracks
            if not any(box_iou(_xyxy(new), _xyxy(old)) >= .35 for old in old_tracks)
        ]
        return [*old_tracks, *additions]
    if mode == 'primary':
        fallbacks = [
            old for old in old_tracks
            if not any(box_iou(_xyxy(old), _xyxy(new)) >= .35 for new in new_tracks)
        ]
        return [*new_tracks, *fallbacks]
    return old_tracks


@dataclass
class TrackState:
    track_id: int
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    last_center_x: float
    velocity: float
    last_seen: float


class IouTracker:
    """Small IoU tracker for new-model detections only."""

    def __init__(self, first_id: int = 1_000_000, ttl_seconds: float = 1.5):
        self.next_id = first_id
        self.ttl_seconds = ttl_seconds
        self.tracks: dict[int, TrackState] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self.tracks.clear()

    def update(self, detections: list[dict], now: float | None = None) -> list[tuple]:
        now = now if now is not None else time.monotonic()
        with self._lock:
            self.tracks = {
                track_id: track
                for track_id, track in self.tracks.items()
                if now - track.last_seen <= self.ttl_seconds
            }
            unmatched = set(self.tracks)
            output = []
            ordered = sorted(
                detections,
                key=lambda item: float(item.get('confidence', 0)),
                reverse=True,
            )
            for detection in ordered:
                x1, y1, x2, y2 = map(float, detection['bbox'])
                class_id = int(detection['class_id'])
                confidence = float(detection['confidence'])
                best_id = None
                best_iou = 0.0
                for track_id in unmatched:
                    state = self.tracks[track_id]
                    if state.class_id != class_id:
                        continue
                    overlap = box_iou(
                        (x1, y1, x2, y2),
                        (state.x1, state.y1, state.x2, state.y2),
                    )
                    if overlap > best_iou:
                        best_iou = overlap
                        best_id = track_id
                center_x = (x1 + x2) / 2
                if best_id is not None and best_iou >= .3:
                    unmatched.remove(best_id)
                    state = self.tracks[best_id]
                    velocity = center_x - state.last_center_x
                    state.x1, state.y1, state.x2, state.y2 = x1, y1, x2, y2
                    state.class_id = class_id
                    state.confidence = confidence
                    state.last_center_x = center_x
                    state.velocity = velocity
                    state.last_seen = now
                    track_id = best_id
                else:
                    track_id = self.next_id
                    self.next_id += 1
                    state = TrackState(
                        track_id=track_id,
                        class_id=class_id,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=confidence,
                        last_center_x=center_x,
                        velocity=0.0,
                        last_seen=now,
                    )
                    self.tracks[track_id] = state
                output.append((
                    track_id,
                    class_id,
                    round(confidence, 4),
                    round((x1 + x2) / 2),
                    round((y1 + y2) / 2),
                    max(1, round(x2 - x1)),
                    max(1, round(y2 - y1)),
                    round(state.velocity, 4),
                ))
            return output


class RemoteDetector:
    def __init__(self, endpoint: str, timeout_ms: int = 120):
        self.endpoint = endpoint.rstrip('/')
        self.timeout = max(.01, min(timeout_ms, 500) / 1000)
        self.last_error = 0.0
        self.error_cooldown = 5.0

    def detect(self, image: Any) -> list[dict]:
        if time.monotonic() - self.last_error < self.error_cooldown:
            return []
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [
            int(cv2.IMWRITE_JPEG_QUALITY), 90,
        ])
        if not ok:
            return []
        request = urllib.request.Request(
            f'{self.endpoint}/predict',
            data=encoded.tobytes(),
            headers={'Content-Type': 'image/jpeg'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode('utf-8'))
            return list(payload.get('detections') or [])
        except (OSError, ValueError, urllib.error.URLError, TimeoutError) as exception:
            self.last_error = time.monotonic()
            LOGGER.warning('New detector unavailable, fallback to old tracker: %s', exception)
            return []


class DetectionRouter:
    """Drop-in replacement around oashya Tracker with safe fallback."""

    def __init__(
        self,
        old_tracker,
        *,
        project_root: Path,
        mode: str = 'auto',
        manifest_path: str = 'weights/hyakkiyakou/active.json',
        timeout_ms: int = 120,
        observation_log: bool = False,
        remote: RemoteDetector | None = None,
    ):
        self.old_tracker = old_tracker
        self.project_root = Path(project_root).resolve()
        self.manifest_path = self.project_root / manifest_path
        self.manifest = self._load_manifest()
        requested_mode = mode.lower() if mode.lower() in VALID_MODES else 'auto'
        manifest_mode = str(self.manifest.get('mode') or 'old').lower()
        self.mode = manifest_mode if requested_mode == 'auto' else requested_mode
        if self.mode not in VALID_MODES - {'auto'}:
            self.mode = 'old'
        self.endpoint = self.manifest.get('service_url') or 'http://127.0.0.1:8899'
        self.remote = remote or RemoteDetector(self.endpoint, timeout_ms=timeout_ms)
        self.tracker = IouTracker()
        self.observation_log = observation_log
        self.log_path = self.project_root / 'log' / 'hyakkiyakou' / 'detector_router.jsonl'
        self.last_observation = 0.0

    def _load_manifest(self) -> dict:
        try:
            return json.loads(self.manifest_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return {}

    def clear_tracks(self) -> None:
        self.tracker.clear()
        clear = getattr(self.old_tracker, 'clear_tracks', None)
        if callable(clear):
            clear()

    def __call__(self, image, response) -> list[tuple]:
        old_tracks = self.old_tracker(image=image, response=response)
        if self.mode == 'old' or not self.manifest:
            return old_tracks
        detections = self.remote.detect(image)
        new_tracks = self.tracker.update(detections)
        merged = merge_tracks(old_tracks, new_tracks, self.mode)
        self._record(old_tracks, new_tracks, merged)
        return merged

    def _record(self, old_tracks: list, new_tracks: list, merged: list) -> None:
        if not self.observation_log:
            return
        now = time.monotonic()
        if now - self.last_observation < 1:
            return
        self.last_observation = now
        row = {
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mode': self.mode,
            'version': self.manifest.get('model_version'),
            'old': len(old_tracks),
            'new': len(new_tracks),
            'merged': len(merged),
            'added': max(0, len(merged) - len(old_tracks)) if self.mode == 'assist' else 0,
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        except OSError:
            pass

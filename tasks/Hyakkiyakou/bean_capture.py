"""撒豆过程的整局截图采集（给新式神做检测训练集）。

存两份：

- `raw/frame_000001.jpg`：**原图**，不画框、不压缩到有损外观，以后想用任何模型
  重新预标注都可以。
- `prelabel/frame_000001.json`：旧模型（oashya）给出的框，X-AnyLabeling / LabelMe
  格式。只是"建议"，人工核对后才会变成训练标签。

参数取舍：

- 默认每 1 秒存一帧：一只式神横穿约 3~5 秒，能拿到 3~5 个角度/位置，够训检测；
  再密就是同一个人重复几十张，对训练没帮助，只涨磁盘。
- 画面上一个目标都没有时，每 5 秒才存一张（当负样本），否则空场景会占掉一半空间。
- 质量 88：1280x720 的游戏截图大约 120~180KB/帧。
  一局 5 分钟 ≈ 300 帧 ≈ 40~55MB，10 局约 0.5GB。

目录：`data/bean_capture/<时间>_r001/`，每局一个，互不覆盖。
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from module.logger import logger
from tasks.Hyakkiyakou.shikigami_table import SHIKIGAMI_TABLE


class BeanCapture:
    schema_version = 1

    def __init__(self, root='data/bean_capture', interval_ms: int = 1000, quality: int = 88,
                 prelabel: bool = True, empty_every: int = 5, max_frames: int = 3000,
                 settings: dict = None):
        self.root = Path(root)
        self.interval = max(0.2, interval_ms/1000)
        self.quality = int(quality)
        self.prelabel = prelabel
        self.empty_every = max(0, int(empty_every))
        self.max_frames = max(0, int(max_frames))
        self.settings = settings or {}
        self.folder: Path | None = None
        self.round = 0
        self.count = 0
        self.saved = 0
        self.last_time = 0.0
        self.last_empty = 0.0

    # ------------------------------------------------------------------ 生命周期
    def begin_round(self) -> Path:
        self.round += 1
        self.folder = self.root / f'{datetime.now():%Y-%m-%d_%H-%M-%S}_r{self.round:03d}'
        (self.folder/'raw').mkdir(parents=True, exist_ok=False)
        if self.prelabel:
            (self.folder/'prelabel').mkdir()
        self.count = 0
        self.saved = 0
        self.last_time = 0.0
        self.last_empty = 0.0
        self._write_round('capturing')
        self._write_classes()
        return self.folder

    def finish(self, status: str = 'complete') -> None:
        self._write_round(status)

    def _write_round(self, status: str) -> None:
        if self.folder is None:
            return
        data = {
            'schema_version': self.schema_version,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'round': self.round,
            'status': status,
            'frames_seen': self.count,
            'frames_saved': self.saved,
            'settings': {**self.settings, 'interval_ms': int(self.interval*1000),
                         'quality': self.quality, 'prelabel': self.prelabel,
                         'empty_every_s': self.empty_every, 'max_frames': self.max_frames},
            'note': 'raw 是原图；prelabel 是旧模型建议，需在 X-AnyLabeling 里核对后导出 YOLO 标签。',
        }
        (self.folder/'round.json').write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')

    def _write_classes(self) -> None:
        """X-AnyLabeling 的类别清单，放在采集根目录，所有局共用。"""
        target = self.root/'xany_classes.txt'
        if target.is_file():
            return
        lines = [f"{row['label']} {row['name']}" for _, row in sorted(SHIKIGAMI_TABLE.items())]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    # ------------------------------------------------------------------ 每帧
    def tick(self, image, tracks, phase: str = 'running') -> None:
        """在主循环里每帧调用一次；内部自己决定存不存，异常不外抛。"""
        try:
            self._tick(image, tracks or [], phase)
        except Exception as exception:
            logger.warning(f'Bean capture failed: {exception}')

    def _tick(self, image, tracks, phase: str) -> None:
        if self.folder is None:
            self.begin_round()
        self.count += 1
        if self.max_frames and self.saved >= self.max_frames:
            return
        now = time.monotonic()
        if now - self.last_time < self.interval:
            return
        if not tracks and self.empty_every and now - self.last_empty < self.empty_every:
            return
        self.last_time = now
        if not tracks:
            self.last_empty = now

        self.saved += 1
        name = f'frame_{self.saved:06d}'
        cv2.imwrite(str(self.folder/'raw'/f'{name}.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        record = {
            'frame': name,
            'captured_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'phase': phase,
            'tracks': [[int(t[0]), int(t[1]), round(float(t[2]), 4),
                        int(t[3]), int(t[4]), int(t[5]), int(t[6]), round(float(t[7]), 4)]
                       for t in tracks],
        }
        with (self.folder/'frames.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(record, ensure_ascii=False) + '\n')
        if self.prelabel:
            self._write_prelabel(name, image.shape, tracks)
        self._write_round('capturing')

    def _write_prelabel(self, name: str, shape, tracks) -> None:
        height, width = int(shape[0]), int(shape[1])
        shapes = []
        for track in tracks:
            _, class_id, confidence, cx, cy, box_w, box_h, _ = track
            x1 = max(0., cx - box_w/2)
            y1 = max(0., cy - box_h/2)
            x2 = min(float(width), cx + box_w/2)
            y2 = min(float(height), cy + box_h/2)
            if x2 <= x1 or y2 <= y1:
                continue
            label = SHIKIGAMI_TABLE.get(int(class_id), {}).get('label', f'unknown_{class_id}')
            shapes.append({
                'label': label,
                'points': [[round(x1, 1), round(y1, 1)], [round(x2, 1), round(y2, 1)]],
                'group_id': None, 'shape_type': 'rectangle', 'flags': {},
                'description': f'旧模型预标注 {float(confidence):.3f}，需人工核对',
            })
        payload = {
            'version': '4.1.1',
            'flags': {'model_suggestion': True, 'reviewed': False},
            'shapes': shapes,
            'imagePath': f'{name}.jpg',
            'imageData': None,
            'imageHeight': height,
            'imageWidth': width,
        }
        (self.folder/'prelabel'/f'{name}.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')

"""鬼王候选自动截图。

每局开始时新建一个文件夹，三个候选各存一张整屏画面（JPEG），并在 `meta.json`
里记下旧模型的判断结果。裁图不在这里做：分辨率或界面变了以后可以用
`dev_tools/boss_dataset.py crop` 重新裁，不用重跑。

目录形如 `log/hyakkiyakou/boss_capture/2026-10-25_14-28-59_001/candidate_1.jpg`，
时间按本地时区，最后三位是本局序号。一局 3 张，约 0.7MB。
"""

import json
from datetime import datetime
from pathlib import Path

import cv2

from module.logger import logger


class BossCapture:
    schema_version = 1

    def __init__(self, root='log/hyakkiyakou/boss_capture', quality: int = 95):
        self.root = Path(root)
        self.quality = quality
        self.folder: Path | None = None
        self.entries: list[dict] = []
        self.round = 0

    def begin_round(self) -> Path:
        """每局开始时调用：新建本局的文件夹，避免和上一局互相覆盖。"""
        self.round += 1
        now = datetime.now()
        # 文件名给人看，用 2026-10-25_14-28-59 这种格式，避免 T/无分隔符
        self.folder = self.root / f'{now:%Y-%m-%d_%H-%M-%S}_{self.round:03d}'
        self.folder.mkdir(parents=True, exist_ok=False)
        self.entries = []
        self._write('capturing')
        return self.folder

    def _write(self, status: str = 'capturing') -> None:
        if self.folder is None:
            return
        data = {
            'schema_version': self.schema_version,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'round': self.round,
            'status': status,
            'note': '旧模型的识别结果只作参考，不是训练标签；稀有度需人工确认。',
            'entries': self.entries,
        }
        target = self.folder / 'meta.json'
        temporary = self.folder / 'meta.json.tmp'
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(target)

    def save(self, image, index: int, score: int = None, class_id: int = None,
             name: str = None, raw=None) -> None:
        """保存一个候选画面。任何异常都不能影响主线流程。"""
        try:
            if self.folder is None:
                self.begin_round()
            filename = f'candidate_{index + 1}.jpg'
            # device.image 是 RGB，cv2.imwrite 需要 BGR
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(self.folder / filename), bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            self.entries.append({
                'index': index + 1,
                'file': filename,
                'image_size': [int(image.shape[1]), int(image.shape[0])],
                'old_score': None if score is None else int(score),
                'old_class_id': None if class_id is None else int(class_id),
                'old_name': name,
                'old_raw': repr(raw) if raw is not None else None,
            })
            self._write()
        except Exception as exception:  # 采集失败不阻塞百鬼夜行
            logger.warning(f'Boss capture failed: {exception}')

    def finish(self, status: str = 'complete') -> None:
        try:
            self._write(status)
        except Exception as exception:
            logger.warning(f'Boss capture finish failed: {exception}')

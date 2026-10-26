"""Local GPU YOLO11 inference process used by the OAS runtime adapter."""

from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np


_DLL_HANDLES = []


def configure_cuda_dlls() -> None:
    """Make CUDA DLLs from the training environment available to ORT if needed."""
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / '.venv-train' / 'Lib' / 'site-packages' / 'torch' / 'lib',
    ]
    nvidia_root = project_root / '.venv-infer' / 'Lib' / 'site-packages' / 'nvidia'
    if nvidia_root.is_dir():
        candidates.extend(sorted(nvidia_root.glob('*/bin')))
    existing = [str(path) for path in candidates if Path(path).is_dir()]
    if existing:
        os.environ['PATH'] = os.pathsep.join([*existing, os.environ.get('PATH', '')])
    if os.name == 'nt':
        for path in existing:
            try:
                _DLL_HANDLES.append(os.add_dll_directory(path))
            except OSError:
                pass


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding='utf-8'))
    model_file = payload.get('model_file')
    if not model_file:
        raise ValueError('模型清单缺少 model_file')
    payload['_model_path'] = str((path.parent / model_file).resolve())
    classes = payload.get('classes') or []
    payload['_classes'] = {int(row['class_index']): row for row in classes}
    payload['_thresholds'] = {
        int(index): float(value)
        for index, value in (payload.get('thresholds') or {}).items()
    }
    return payload


def nms(boxes: list[list[float]], scores: list[float], threshold: float = .7) -> list[int]:
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    keep = []
    while order:
        current = order.pop(0)
        keep.append(current)
        a = boxes[current]
        remaining = []
        for index in order:
            b = boxes[index]
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            overlap = intersection / max(1e-9, area_a + area_b - intersection)
            if overlap <= threshold:
                remaining.append(index)
        order = remaining
    return keep


class OnnxYoloDetector:
    """YOLO11 ONNX postprocessing without importing PyTorch/Ultralytics."""

    def __init__(self, path: str, manifest: dict, imgsz: int):
        import onnxruntime as ort

        available = ort.get_available_providers()
        preferred = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available else []
        self.session = ort.InferenceSession(
            path,
            providers=preferred + ['CPUExecutionProvider'],
        )
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        self.input_width = int(shape[3]) if isinstance(shape[3], int) else int(imgsz)
        self.input_height = int(shape[2]) if isinstance(shape[2], int) else int(imgsz)
        self.classes = manifest['_classes']
        self.thresholds = manifest['_thresholds']
        self.minimum_confidence = min(self.thresholds.values(), default=.25)
        self.provider = self.session.get_providers()[0]

    def predict(self, image) -> list[dict]:
        height, width = image.shape[:2]
        ratio = min(self.input_width / width, self.input_height / height)
        resized_width = max(1, round(width * ratio))
        resized_height = max(1, round(height * ratio))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = self.session.run(None, {self.input.name: tensor})[0]
        predictions = np.asarray(output)
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise ValueError(f'Unexpected YOLO output shape: {predictions.shape}')
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.transpose(1, 0)

        candidates: dict[int, list[dict]] = {}
        for row in predictions:
            if row.shape[0] < 5:
                continue
            scores = row[4:]
            class_index = int(np.argmax(scores))
            confidence = float(scores[class_index])
            if confidence < self.thresholds.get(class_index, .95):
                continue
            center_x, center_y, box_width, box_height = map(float, row[:4])
            x1 = (center_x - box_width / 2 - pad_x) / ratio
            y1 = (center_y - box_height / 2 - pad_y) / ratio
            x2 = (center_x + box_width / 2 - pad_x) / ratio
            y2 = (center_y + box_height / 2 - pad_y) / ratio
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.setdefault(class_index, []).append({
                'box': [x1, y1, x2, y2],
                'confidence': confidence,
            })

        result = []
        for class_index, items in candidates.items():
            keep = nms([item['box'] for item in items], [item['confidence'] for item in items])
            meta = self.classes.get(class_index)
            if meta is None:
                continue
            for index in keep:
                item = items[index]
                result.append({
                    'class_id': int(meta['class_id']),
                    'class_index': class_index,
                    'label': meta.get('label'),
                    'rarity': meta.get('rarity'),
                    'confidence': round(item['confidence'], 6),
                    'bbox': [round(value, 2) for value in item['box']],
                })
        return result


class UltralyticsDetector:
    """Development fallback for .pt models."""

    def __init__(self, path: str, manifest: dict, imgsz: int, device: int):
        from ultralytics import YOLO

        self.model = YOLO(path)
        self.imgsz = imgsz
        self.device = device
        self.classes = manifest['_classes']
        self.thresholds = manifest['_thresholds']
        self.minimum_confidence = min(self.thresholds.values(), default=.25)
        self.provider = 'PyTorch'

    def predict(self, image) -> list[dict]:
        result = self.model.predict(
            source=image,
            conf=self.minimum_confidence,
            iou=.7,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            max_det=300,
        )[0]
        detections = []
        if result.boxes is None or not len(result.boxes):
            return detections
        xyxy = result.boxes.xyxy.cpu().tolist()
        confidence = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        for box, score, class_index in zip(xyxy, confidence, classes):
            class_index = int(class_index)
            meta = self.classes.get(class_index)
            threshold = self.thresholds.get(class_index, .95)
            if meta is None or float(score) < threshold:
                continue
            detections.append({
                'class_id': int(meta['class_id']),
                'class_index': class_index,
                'label': meta.get('label'),
                'rarity': meta.get('rarity'),
                'confidence': round(float(score), 6),
                'bbox': [round(float(value), 2) for value in box],
            })
        return detections


def build_model(manifest: dict, imgsz: int, device: int):
    path = manifest['_model_path']
    if Path(path).suffix.lower() == '.onnx':
        return OnnxYoloDetector(path, manifest, imgsz)
    return UltralyticsDetector(path, manifest, imgsz, device)


def build_handler(manifest: dict, model):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'HyakkiInference/1.0'

        def _send(self, code: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args) -> None:
            print(f'{self.address_string()} - {format % args}', flush=True)

        def do_GET(self) -> None:
            if self.path != '/health':
                self._send(404, {'error': 'not found'})
                return
            self._send(200, {
                'ok': True,
                'model_version': manifest.get('model_version'),
                'provider': model.provider,
                'gpu': {'available': 'CPU' not in model.provider, 'provider': model.provider},
                'classes': len(manifest['_classes']),
            })

        def do_POST(self) -> None:
            if self.path != '/predict':
                self._send(404, {'error': 'not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > 12 * 1024 * 1024:
                    self._send(400, {'error': 'invalid image size'})
                    return
                image = cv2.imdecode(
                    np.frombuffer(self.rfile.read(length), dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if image is None:
                    self._send(400, {'error': 'cannot decode image'})
                    return
                started = time.perf_counter()
                detections = model.predict(image)
                self._send(200, {
                    'detections': detections,
                    'elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
                    'model_version': manifest.get('model_version'),
                    'provider': model.provider,
                })
            except Exception as exception:
                self._send(500, {'error': f'{type(exception).__name__}: {exception}'})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8899)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    configure_cuda_dlls()
    manifest = load_manifest(Path(args.manifest).resolve())
    model = build_model(manifest, args.imgsz, args.device)
    model.predict(np.zeros((720, 1280, 3), dtype=np.uint8))
    print(
        f"Hyakki inference ready: version={manifest.get('model_version')} "
        f"model={manifest['_model_path']} classes={len(manifest['_classes'])} "
        f"provider={model.provider}",
        flush=True,
    )
    server = HTTPServer((args.host, args.port), build_handler(manifest, model))
    server.serve_forever()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

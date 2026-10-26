"""YOLO11 train/test worker executed inside the dedicated training venv."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path


def iou(left, right) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    left_area = (float(left[2]) - float(left[0])) * (float(left[3]) - float(left[1]))
    right_area = (float(right[2]) - float(right[0])) * (float(right[3]) - float(right[1]))
    return intersection / max(1e-9, left_area + right_area - intersection)


def read_ground_truth(image: Path, classes: int) -> list[list]:
    label = image.parents[2] / 'labels' / image.parent.name / f'{image.stem}.txt'
    boxes = []
    if not label.is_file():
        return boxes
    for line in label.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        if class_id < 0 or class_id >= classes:
            continue
        cx, cy, width, height = map(float, parts[1:5])
        boxes.append([
            class_id,
            cx - width / 2,
            cy - height / 2,
            cx + width / 2,
            cy + height / 2,
        ])
    return boxes


def match_predictions(predictions: list[list], truths: list[list], threshold: float) -> tuple[int, int, int]:
    truth_used = set()
    tp = fp = 0
    for prediction in sorted(predictions, key=lambda item: -float(item[1])):
        class_id, confidence, *box = prediction
        if float(confidence) < threshold:
            continue
        candidates = [
            (iou(box, truth[1:]), index)
            for index, truth in enumerate(truths)
            if index not in truth_used and int(truth[0]) == int(class_id)
        ]
        if candidates:
            overlap, index = max(candidates)
            if overlap >= 0.5:
                truth_used.add(index)
                tp += 1
                continue
        fp += 1
    fn = len(truths) - len(truth_used)
    return tp, fp, fn


def predictions_from_result(result) -> list[list]:
    if result.boxes is None or not len(result.boxes):
        return []
    height, width = result.orig_shape
    xyxy = result.boxes.xyxy.cpu().tolist()
    confidence = result.boxes.conf.cpu().tolist()
    class_id = result.boxes.cls.cpu().tolist()
    normalized = []
    for box, conf, cls in zip(xyxy, confidence, class_id):
        normalized.append([
            int(cls),
            float(conf),
            float(box[0]) / width,
            float(box[1]) / height,
            float(box[2]) / width,
            float(box[3]) / height,
        ])
    return normalized


def split_stats(root: Path, split: str, classes: int, thresholds: dict[str, float], model) -> dict:
    image_dir = root / 'images' / split
    images = sorted(
        path for path in image_dir.glob('*') if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    )
    totals = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    durations = []
    for image_index, image in enumerate(images):
        started = time.perf_counter()
        result = model.predict(
            source=str(image),
            conf=0.01,
            iou=0.7,
            imgsz=int(getattr(model, '_hyakki_imgsz', 960)),
            device=0,
            verbose=False,
            max_det=300,
        )[0]
        if image_index:
            durations.append((time.perf_counter() - started) * 1000)
        predictions = predictions_from_result(result)
        truths = read_ground_truth(image, classes)
        for class_id in range(classes):
            class_predictions = [item for item in predictions if int(item[0]) == class_id]
            class_truth = [item for item in truths if int(item[0]) == class_id]
            threshold = float(thresholds.get(str(class_id), .25))
            tp, fp, fn = match_predictions(class_predictions, class_truth, threshold)
            totals[class_id]['tp'] += tp
            totals[class_id]['fp'] += fp
            totals[class_id]['fn'] += fn
    rows = []
    overall = {'tp': 0, 'fp': 0, 'fn': 0}
    for class_id in range(classes):
        item = totals[class_id]
        precision = item['tp'] / max(1, item['tp'] + item['fp'])
        recall = item['tp'] / max(1, item['tp'] + item['fn'])
        rows.append({
            'class_index': class_id,
            'precision': round(precision, 6),
            'recall': round(recall, 6),
            **item,
        })
        for key in overall:
            overall[key] += item[key]
    precision = overall['tp'] / max(1, overall['tp'] + overall['fp'])
    recall = overall['tp'] / max(1, overall['tp'] + overall['fn'])
    ordered = sorted(durations)
    p50 = ordered[max(0, math.ceil(len(ordered) * .50) - 1)] if ordered else None
    p95 = ordered[max(0, math.ceil(len(ordered) * .95) - 1)] if ordered else None
    return {
        'split': split,
        'images': len(images),
        'precision': round(precision, 6),
        'recall': round(recall, 6),
        'latency_ms_p50': round(p50, 2) if p50 is not None else None,
        'latency_ms_p95': round(p95, 2) if p95 is not None else None,
        'classes': rows,
        **overall,
    }


def calibrate_thresholds(root: Path, classes: int, model) -> dict[str, float]:
    images = sorted(
        path for path in (root / 'images' / 'val').glob('*')
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    )
    cached = []
    for image in images:
        result = model.predict(
            source=str(image),
            conf=0.01,
            iou=0.7,
            imgsz=int(getattr(model, '_hyakki_imgsz', 960)),
            device=0,
            verbose=False,
            max_det=300,
        )[0]
        predictions = predictions_from_result(result)
        cached.append((predictions, read_ground_truth(image, classes)))
    thresholds = {}
    for class_id in range(classes):
        best = None
        for step in range(5, 100, 5):
            threshold = step / 100
            tp = fp = fn = 0
            for predictions, truths in cached:
                class_predictions = [item for item in predictions if int(item[0]) == class_id]
                class_truth = [item for item in truths if int(item[0]) == class_id]
                a, b, c = match_predictions(class_predictions, class_truth, threshold)
                tp += a
                fp += b
                fn += c
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            admissible = precision >= .95
            score = (admissible, recall, -threshold, precision)
            if best is None or score > best[0]:
                best = (score, threshold)
        thresholds[str(class_id)] = round(best[1] if best else .60, 2)
    return thresholds


def load_classes(root: Path) -> int:
    import yaml

    payload = yaml.safe_load((root / 'data.yaml').read_text(encoding='utf-8'))
    return len(payload['names'])


def run_train(args) -> dict:
    from ultralytics import YOLO

    root = Path(args.data).resolve().parent
    run_dir = Path(args.run_dir).resolve()
    project = run_dir.parent
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        amp=True,
        pretrained=True,
        project=str(project),
        name=run_dir.name,
        exist_ok=True,
        patience=args.patience,
        workers=4,
        seed=0,
        deterministic=True,
        plots=True,
    )
    best = Path(model.trainer.best).resolve()
    metrics = {str(key): float(value) if isinstance(value, (int, float)) else value
               for key, value in (results.results_dict or {}).items()}
    eval_model = YOLO(str(best))
    eval_model._hyakki_imgsz = args.imgsz
    classes = load_classes(root)
    thresholds = calibrate_thresholds(root, classes, eval_model)
    val = split_stats(root, 'val', classes, thresholds, eval_model)
    thresholds_path = run_dir / 'thresholds.json'
    thresholds_path.write_text(json.dumps(thresholds, ensure_ascii=False, indent=1), encoding='utf-8')
    val_path = run_dir / 'val_metrics.json'
    val_path.write_text(json.dumps(val, ensure_ascii=False, indent=1), encoding='utf-8')
    return {
        'ok': True,
        'best_weight': str(best),
        'metrics': metrics,
        'val': val,
        'thresholds': thresholds,
        'thresholds_path': str(thresholds_path),
        'val_metrics_path': str(val_path),
    }


def run_test(args) -> dict:
    from ultralytics import YOLO

    root = Path(args.data).resolve().parent
    thresholds = json.loads(Path(args.thresholds).read_text(encoding='utf-8'))
    model = YOLO(args.weight)
    model._hyakki_imgsz = args.imgsz
    test = split_stats(root, 'test', load_classes(root), thresholds, model)
    path = Path(args.run_dir) / 'test_metrics.json'
    path.write_text(json.dumps(test, ensure_ascii=False, indent=1), encoding='utf-8')
    return {'ok': True, 'test': test, 'test_metrics_path': str(path)}


def run_export(args) -> dict:
    from ultralytics import YOLO

    output = Path(args.run_dir) / 'export'
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weight, task='detect')
    exported = Path(model.export(
        format='onnx',
        imgsz=args.imgsz,
        dynamic=False,
        simplify=True,
        opset=12,
        device=0,
    )).resolve()
    target = (output / 'model.onnx').resolve()
    if exported != target:
        shutil.copy2(exported, target)
    return {'ok': True, 'onnx': str(target)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--job-id', required=True)
    train.add_argument('--data', required=True)
    train.add_argument('--run-dir', required=True)
    train.add_argument('--model', default='yolo11s.pt')
    train.add_argument('--epochs', type=int, default=100)
    train.add_argument('--imgsz', type=int, default=960)
    train.add_argument('--batch', type=int, default=4)
    train.add_argument('--patience', type=int, default=20)
    test = sub.add_parser('test')
    test.add_argument('--job-id', required=True)
    test.add_argument('--data', required=True)
    test.add_argument('--run-dir', required=True)
    test.add_argument('--weight', required=True)
    test.add_argument('--thresholds', required=True)
    test.add_argument('--imgsz', type=int, default=960)
    test.add_argument('--batch', type=int, default=4)
    export = sub.add_parser('export')
    export.add_argument('--job-id', required=True)
    export.add_argument('--run-dir', required=True)
    export.add_argument('--weight', required=True)
    export.add_argument('--imgsz', type=int, default=960)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / 'result.json'
    try:
        if args.command == 'train':
            result = run_train(args)
        elif args.command == 'test':
            result = run_test(args)
        else:
            result = run_export(args)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
        print('HYAKKI_RESULT=' + json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exception:
        result = {'ok': False, 'error': f'{type(exception).__name__}: {exception}'}
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
        print('HYAKKI_ERROR=' + result['error'], file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

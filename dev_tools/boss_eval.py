"""稀有度模型的评估逻辑（供命令行和工作台共用）。

一次评估同时给出三类信息：

- 单张：真值 vs 预测、置信度、是否低于阈值（会回退）
- 分组：每局三个候选放在一起，模型会选哪个、是不是真的最稀有
- 阈值：换一个阈值，回退率和选择结果怎么变（工作台里用滑块实时看）
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HIGH = {'UR_SP', 'SSR', 'SR'}


def load_dataset(dataset_dir: Path):
    dataset = Path(dataset_dir)
    manifest = json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    payload = json.loads((dataset/'labels.json').read_text(encoding='utf-8'))
    labels = payload.get('labels', payload)
    return dataset, manifest.get('items', []), labels


def evaluate(model, dataset_dir, threshold: float = None) -> dict:
    """model 是 tasks.Hyakkiyakou.boss_rarity.BossRarityModel。"""
    dataset, items, labels = load_dataset(dataset_dir)
    threshold = float(threshold if threshold is not None else model.threshold)

    rows = []
    for item in items:
        truth_label = labels.get(item['id'])
        if truth_label not in HIGH and truth_label != 'below':
            continue
        path = dataset/item['src']
        try:
            image = np.asarray(Image.open(path).convert('RGB'))
        except Exception:
            continue
        score, predicted, confidence = model.score_patch(image)
        rows.append({
            'id': item['id'],
            'truth_score': 1 if truth_label in HIGH else 0,
            'run': item.get('run'),
            'index': item.get('index'),
            'truth': 'SR_plus' if truth_label in HIGH else 'below',
            'truth_label': truth_label,
            'pred': predicted,
            'score': score,
            'confidence': round(confidence, 4),
            'fallback': confidence < threshold,
        })

    if not rows:
        return {'error': '没有已标注的候选图，先标注再评估', 'rows': [], 'rounds': [],
                'threshold': threshold, 'total': 0}

    correct = sum(row['truth'] == row['pred'] for row in rows)
    fallback = sum(row['fallback'] for row in rows)
    confusion = {}
    for row in rows:
        confusion.setdefault(row['truth'], Counter())[row['pred']] += 1

    grouped = defaultdict(list)
    for row in rows:
        grouped[row['run'] or 'unknown'].append(row)

    rounds = []
    for run, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda row: row['index'] or 0)
        # 注意别用字符串比较：'SR_plus' < 'below'，max() 会永远得到 below
        best_true = 'SR_plus' if any(row['truth'] == 'SR_plus' for row in candidates) else 'below'
        picked = _pick(candidates, threshold)
        rounds.append({
            'run': run,
            'candidates': candidates,
            'picked': picked['id'] if picked else None,
            'best_true': best_true,
            'correct': bool(picked) and picked['truth'] == best_true,
            'fallback_count': sum(row['fallback'] for row in candidates),
        })

    usable_rounds = [r for r in rounds if r['picked']]
    return {
        'threshold': threshold,
        'note': '评估用的是训练时的同一批图，单张准确率会虚高；泛化能力看 model.json 里的 CV。',
        'total': len(rows),
        'accuracy': round(correct/len(rows), 4),
        'fallback': fallback,
        'fallback_rate': round(fallback/len(rows), 4),
        'distribution': dict(Counter(row['truth'] for row in rows)),
        'confusion': {key: dict(value) for key, value in confusion.items()},
        'rows': rows,
        'rounds': rounds,
        'round_accuracy': round(sum(r['correct'] for r in usable_rounds)/len(usable_rounds), 4)
        if usable_rounds else None,
        'rounds_total': len(usable_rounds),
        'rounds_hit': sum(r['correct'] for r in usable_rounds),
    }


def _pick(candidates, threshold: float):
    """和运行时一致：低于阈值的候选不采信；都不采信时返回 None（=回退旧检测）。"""
    usable = [row for row in candidates if not row['fallback']]
    if not usable:
        return None
    return max(usable, key=lambda row: (row['score'], -(row['index'] or 0)))

"""命令行评估稀有度模型（工作台的同一套逻辑，方便脚本化对比）。

    .venv/Scripts/python.exe dev_tools/boss_model_eval.py
    .venv/Scripts/python.exe dev_tools/boss_model_eval.py --threshold 0.8
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

import boss_eval  # noqa: E402
from tasks.Hyakkiyakou.boss_rarity import BossRarityModel  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default=str(ROOT/'data/boss_rarity'))
    parser.add_argument('--model', default=str(ROOT/'data/boss_rarity/model/boss_rarity.onnx'))
    parser.add_argument('--threshold', type=float, default=None)
    args = parser.parse_args()

    model = BossRarityModel(args.model, threshold=args.threshold or 0.6)
    result = boss_eval.evaluate(model, args.dataset, threshold=args.threshold)
    if result.get('error'):
        print(result['error'])
        return 1

    print(f"样本 {result['total']} 张，阈值 {result['threshold']:.2f}")
    print(f"单张准确率：{result['accuracy']:.3f}   分布：{result['distribution']}")
    print(f"低于阈值会回退：{result['fallback']}/{result['total']} = {result['fallback_rate']:.1%}")
    if result['round_accuracy'] is not None:
        print(f"每局三选一：{result['rounds_hit']}/{result['rounds_total']} = {result['round_accuracy']:.1%}")
    for row in result['rows']:
        mark = '✓' if row['truth'] == row['pred'] else '✗'
        flag = '回退' if row['fallback'] else '   '
        print(f"  {mark} {flag} {row['id']:<32} 真={row['truth']:<8} 预测={row['pred']:<8} p={row['confidence']:.2f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

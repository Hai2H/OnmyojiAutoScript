"""对比 onnxruntime 的 CPU / CUDA 推理延迟，决定哪些模型值得放 GPU。

    .venv/Scripts/python.exe dev_tools/gpu_bench.py --model data/boss_rarity/model/boss_rarity.onnx
    .venv/Scripts/python.exe dev_tools/gpu_bench.py --model xxx.onnx --size 960 --iters 50

只测**计算**：输入是随机张量，不读图，所以数字反映的是模型本身的推理耗时，
包含首次预热与显存分配，但不包含截图和预处理。

如果输出里没有 CUDAExecutionProvider，说明当前环境装的是 CPU 版 onnxruntime：

    .venv/Scripts/python.exe -m pip uninstall -y onnxruntime
    .venv/Scripts/python.exe -m pip install onnxruntime-gpu==1.19.2

⚠ onnxruntime 是全局共用的（OCR 也在用），换 GPU 版前先备份：
    .venv/Scripts/python.exe -m pip freeze > requirements-lock.txt
换完必须回归 OCR：跑一次任意任务，确认文字识别正常，否则退回
    .venv/Scripts/python.exe -m pip install onnxruntime==1.16.3
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def bench(session, size: int, iters: int, warmup: int = 5) -> dict:
    name = session.get_inputs()[0].name
    shape = session.get_inputs()[0].shape
    # 模型可能固定尺寸，按模型声明的来
    height = shape[2] if isinstance(shape[2], int) else size
    width = shape[3] if isinstance(shape[3], int) else size
    dummy = np.random.rand(1, 3, height, width).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {name: dummy})
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        session.run(None, {name: dummy})
        samples.append((time.perf_counter() - started)*1000)
    samples.sort()
    return {
        'input': (height, width),
        'mean': statistics.mean(samples),
        'p50': samples[len(samples)//2],
        'p95': samples[int(len(samples)*0.95) - 1],
        'min': samples[0],
    }


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--iters', type=int, default=50)
    args = parser.parse_args()

    import onnxruntime as ort
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f'模型不存在：{model_path}')
        return 1

    available = ort.get_available_providers()
    print(f'onnxruntime {ort.__version__}，可用 provider：{available}')

    plans = [('CPU', ['CPUExecutionProvider'])]
    if 'CUDAExecutionProvider' in available:
        plans.insert(0, ('CUDA', ['CUDAExecutionProvider', 'CPUExecutionProvider']))
    if 'TensorrtExecutionProvider' in available:
        plans.insert(0, ('TensorRT', ['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                      'CPUExecutionProvider']))

    for label, providers in plans:
        try:
            session = ort.InferenceSession(str(model_path), providers=providers)
            actual = session.get_providers()[0]
            result = bench(session, args.size, args.iters)
            print(f"{label:<8} 实际 {actual:<26} 输入 {result['input']} "
                  f"mean {result['mean']:.2f} ms  p50 {result['p50']:.2f}  "
                  f"p95 {result['p95']:.2f}  min {result['min']:.2f}")
        except Exception as exception:
            print(f'{label:<8} 不可用：{exception}')

    if 'CUDAExecutionProvider' not in available:
        print('\n当前是 CPU 版 onnxruntime。要试 GPU 见文件头部的安装与回滚命令。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

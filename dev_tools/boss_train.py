"""训练鬼王候选的稀有度分类模型（试点：SR+ / below 二分类）。

数据来自 `data/boss_rarity`：`manifest.json` + `labels.json`（网页标注写的那份）。
输出到 `data/boss_rarity/model/boss_rarity.onnx` + `model.json`，运行时由
`tasks/Hyakkiyakou/boss_rarity.py` 加载。

设计要点：

- **用预训练骨干**（mobilenet_v3_small，ImageNet 权重）+ 线性分类头。29 张图从零训
  卷积只会退化成"永远猜多数类"，预训练特征才能区分 SR 与 R/N 的画风。
- 骨干冻结，图片只在开始时解码一次，特征也只算一次（可缓存），之后训练的是
  线性头——所以全程在 GPU 上，CPU 不参与循环。
- 导出的是"归一化 + 骨干 + 头"的完整图，ONNX 输入仍然是 0~1 的 RGB，运行时
  不需要知道 ImageNet 归一化的存在。

样本很少，报告**分层 5 折交叉验证**准确率，并和"永远猜多数类"的基线对比；
不比基线好的模型会被 `tasks/Hyakkiyakou/boss_rarity.py` 判定为不可用而不接管。

用法：
    .venv-train/Scripts/python.exe dev_tools/boss_train.py --device cuda
    .venv-train/Scripts/python.exe dev_tools/boss_train.py --device cuda --backbone none   # 退回复古小 CNN
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.Hyakkiyakou.boss_rarity import INPUT_SIZE, preprocess  # noqa: E402

DATASET = ROOT/'data/boss_rarity'
HIGH = {'UR_SP', 'SSR', 'SR'}
CLASSES = ['below', 'SR_plus']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SmallNet(nn.Module):
    """没有 torchvision 时的退路：三组卷积 + 全局池化。"""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.head(self.features(x))

    def embed(self, x):
        return self.features(x).mean(dim=(2, 3))


class PretrainedNet(nn.Module):
    """mobilenet_v3_small（冻结）+ 线性头，输入 0~1 的 RGB。"""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        channels = 576  # mobilenet_v3_small 最后一层通道数
        self.head = nn.Linear(channels, num_classes)
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def embed(self, x):
        with torch.no_grad():
            x = (x - self.mean)/self.std
            feature = self.features(x)
        return feature.mean(dim=(2, 3))

    def forward(self, x):
        return self.head(self.embed(x))


def resolve_device(requested: str = 'auto') -> torch.device:
    if requested == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if requested == 'cuda':
        raise SystemExit('要求用 GPU，但 torch.cuda.is_available() 是 False：检查驱动或换 CUDA 版 torch')
    return torch.device('cpu')


def describe_device(device: torch.device) -> str:
    if device.type != 'cuda':
        return f'{device.type}（CPU 训练会慢很多，确认是否装了 CUDA 版 torch）'
    capability = torch.cuda.get_device_capability(0)
    return f'cuda（{torch.cuda.get_device_name(0)}, sm_{capability[0]}{capability[1]}）｜torch {torch.__version__}'


def load_items():
    manifest = json.loads((DATASET/'manifest.json').read_text(encoding='utf-8'))
    payload = json.loads((DATASET/'labels.json').read_text(encoding='utf-8'))
    labels = payload.get('labels', payload)
    items = []
    for item in manifest['items']:
        label = labels.get(item['id'])
        if label in HIGH:
            items.append((DATASET/item['src'], 1))
        elif label == 'below':
            items.append((DATASET/item['src'], 0))
    return items


def load_tensors(items):
    """只在这里解码图片，之后全程张量。"""
    images = np.stack([preprocess(np.asarray(Image.open(path).convert('RGB')))[0]
                       for path, _ in items])
    labels = np.array([label for _, label in items], dtype=np.int64)
    return torch.from_numpy(images), torch.from_numpy(labels)


def build_model(backbone: str) -> nn.Module:
    if backbone == 'none':
        return SmallNet()
    try:
        return PretrainedNet()
    except Exception as exception:  # 没装 torchvision / 下载失败
        print(f'预训练骨干不可用（{exception}），退回复古小 CNN')
        return SmallNet()


def embed_all(model, images, device, flip: bool = False) -> torch.Tensor:
    batch = images.to(device)
    if flip:
        batch = torch.flip(batch, dims=[3])
    return model.embed(batch)


def train_head(features, labels, epochs: int, seed: int = 0, device=None) -> nn.Linear:
    """在线性头上训练；特征已经算好，循环里只有矩阵乘。"""
    device = device or torch.device('cpu')
    torch.manual_seed(seed)
    features, labels = features.to(device), labels.to(device)
    head = nn.Linear(features.shape[1], 2).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05, weight_decay=1e-3)
    counts = Counter(labels.tolist())
    weights = torch.tensor([len(labels)/(2*max(1, counts[0])), len(labels)/(2*max(1, counts[1]))],
                           dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(head(features), labels)
        loss.backward()
        optimizer.step()
    return head


@torch.no_grad()
def accuracy_of(head, features, labels, device) -> float:
    prediction = head(features.to(device)).argmax(dim=1).cpu()
    return float((prediction == labels).float().mean().item())


def stratified_folds(labels, folds: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    folds_index = [[] for _ in range(folds)]
    for label in (0, 1):
        index = torch.nonzero(labels == label).flatten()
        index = index[torch.randperm(len(index), generator=generator)]
        for position, value in enumerate(index.tolist()):
            folds_index[position % folds].append(value)
    return folds_index


def export_onnx(model: nn.Module, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    model.cpu()
    model.eval()
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    kwargs = dict(input_names=['input'], output_names=['logits'],
                  opset_version=12, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, str(target), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(target), **kwargs)


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3, help='重复几轮交叉验证，降低小样本的抖动')
    parser.add_argument('--out', default=str(DATASET/'model'))
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--backbone', default='mobilenet', choices=['mobilenet', 'none'])
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f'设备：{describe_device(device)}')

    items = load_items()
    counts = Counter('SR_plus' if label else 'below' for _, label in items)
    print(f'样本 {len(items)} 张：{dict(counts)}')
    if len(items) < 10:
        print('样本太少，先去标注（boss_label_server.py）')
        return 1

    images, labels = load_tensors(items)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    print(f'输入：{tuple(images.shape)}，数据已驻留{"显存" if device.type == "cuda" else "内存"}')

    majority = max(counts.values())/len(items)
    print(f'基线（永远猜多数类）：{majority:.3f}')

    started = time.time()
    model = build_model(args.backbone)
    model.to(device).eval()
    # 特征只算一次：原图 + 左右翻转各一份，兼容一点平移/镜像
    feature_orig = embed_all(model, images, device, flip=False)
    feature_flip = embed_all(model, images, device, flip=True)
    print(f'特征：{tuple(feature_orig.shape)}（骨干冻结，CPU 不再参与）')

    # 折按"原图"划分；翻转图只跟随原图进入训练集，不能跨折，否则同一张图的
    # 两个副本分别落在训练/验证里，准确率会被高估。
    # 小样本下单独一轮 5 折波动极大（一折里只有 2 张 SR），多重复几轮取均值
    scores = []
    for repeat in range(args.repeats):
        folds_index = stratified_folds(labels, args.folds, seed=repeat)
        repeat_scores = []
        for index, fold in enumerate(folds_index):
            train_index = [i for j, part in enumerate(folds_index) if j != index for i in part]
            train_features = torch.cat([feature_orig[train_index], feature_flip[train_index]], dim=0)
            train_labels = torch.cat([labels[train_index], labels[train_index]], dim=0)
            head = train_head(train_features, train_labels, args.epochs,
                              seed=repeat*100 + index, device=device)
            repeat_scores.append(accuracy_of(head, feature_orig[fold], labels[fold], device))
        scores.extend(repeat_scores)
        print(f'  第 {repeat + 1} 轮 {args.folds} 折：{np.mean(repeat_scores):.3f}'
              f'（{", ".join(f"{v:.2f}" for v in repeat_scores)}）')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - started
    print(f'交叉验证准确率：{np.mean(scores):.3f} ± {np.std(scores):.3f}')

    all_features = torch.cat([feature_orig, feature_flip], dim=0)
    all_labels = torch.cat([labels, labels], dim=0)
    final = train_head(all_features, all_labels, args.epochs, seed=0, device=device)
    model.head.load_state_dict(final.state_dict())
    target = Path(args.out)/'boss_rarity.onnx'
    export_onnx(model, target)

    meta = {
        'schema_version': 1,
        'classes': CLASSES,
        'input_size': INPUT_SIZE,
        'backbone': args.backbone,
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'samples': len(items),
        'counts': dict(counts),
        'cv_accuracy': round(float(np.mean(scores)), 4),
        'baseline': round(float(majority), 4),
        'device': device.type,
        'repeats': args.repeats,
        'note': '试点二分类：SR 及以上 vs 以下。样本量小，CV 只作参考。',
    }
    # 保留工作台里调过的阈值，不要因为重新训练被重置
    meta_path = Path(args.out)/'model.json'
    if meta_path.is_file():
        try:
            meta['threshold'] = json.loads(meta_path.read_text(encoding='utf-8')).get('threshold', 0.6)
        except Exception:
            meta['threshold'] = 0.6
    else:
        meta['threshold'] = 0.6
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')

    print(f'训练耗时：{elapsed:.1f} s（{args.repeats} 轮 × {args.folds} 折，各 {args.epochs} epoch）')
    if device.type == 'cuda':
        print(f'显存峰值：{torch.cuda.max_memory_allocated()/1024/1024:.1f} MB')
    print(f'已导出：{target}（{target.stat().st_size/1024/1024:.1f} MB）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

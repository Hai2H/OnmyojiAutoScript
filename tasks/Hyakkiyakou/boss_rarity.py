"""鬼王候选的稀有度模型（运行时）。

只解决一件事：鬼王选择页的三个候选，哪个更值钱。所以输入是候选槽的裁剪图，
输出是稀有度类别；不需要画框，也不需要跟踪。

训练见 `dev_tools/boss_train.py`，训练产物是 `model/*.onnx` + `model.json`。
这里的预处理必须和训练时**完全一致**（同一份 `crop_slot` / `preprocess`），
否则线上精度会莫名其妙地掉。
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

# 三个候选槽位，1280x720 画面坐标；与 dev_tools/boss_dataset.py 共用
SLOTS = [(100, 60, 470, 690), (470, 60, 850, 690), (850, 60, 1180, 690)]
INPUT_SIZE = 224
# 二分类试点：先把 SR 及以上和以下分开；攒够样本后再拆成 4 类
CLASSES = ['below', 'SR_plus']
# 模型类别 → 稀有度分，值必须和 shikigami_table.RARITY_SCORE 的量级一致
CLASS_SCORE = {'below': 2, 'SR_plus': 3}


def crop_slot(image: np.ndarray, index: int) -> np.ndarray:
    """从整屏 RGB 图里切出第 index 个候选槽（0 起）。"""
    x0, y0, x1, y1 = SLOTS[index]
    height, width = image.shape[:2]
    return image[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]


def preprocess(patch: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """RGB 裁剪图 → 1x3xNxN float32，值域 0~1。训练和推理共用。"""
    image = Image.fromarray(patch).convert('RGB').resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array.transpose(2, 0, 1), axis=0)


class BossRarityModel:
    """加载 onnx，给出候选的稀有度。模型不存在或跑不动时抛异常，由调用方回退。"""

    def __init__(self, model_path, threshold: float = 0.6):
        self.model_path = Path(model_path)
        meta_path = self.model_path.parent / 'model.json'
        self.meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.is_file() else {}
        # 阈值可以在工作台里调，存在 model.json 里；没有就用参数/默认值
        self.threshold = float(self.meta.get('threshold', threshold))
        self.classes = self.meta.get('classes', CLASSES)
        self.input_size = int(self.meta.get('input_size', INPUT_SIZE))
        self.cv_accuracy = self.meta.get('cv_accuracy')
        self.provider = ''
        self.baseline = self.meta.get('baseline')
        import onnxruntime as ort
        available = ort.get_available_providers()
        # 装了 onnxruntime-gpu 就优先用 CUDA；否则退回 CPU。日志里能看到实际用的是哪个
        preferred = [p for p in ('TensorrtExecutionProvider', 'CUDAExecutionProvider') if p in available]
        providers = preferred + ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @property
    def usable(self) -> bool:
        """不比"永远猜多数类"更好的模型不应该接管决策。"""
        if self.cv_accuracy is None or self.baseline is None:
            return True  # 老版本 model.json 没有这两个字段，交给阈值兜底
        return float(self.cv_accuracy) > float(self.baseline)

    def predict_patch(self, patch: np.ndarray) -> tuple[str, float]:
        """对已经裁好的候选图做分类，返回 (类别, 置信度)。"""
        if patch.size == 0:
            raise ValueError('empty candidate patch')
        logits = self.session.run(
            [self.output_name], {self.input_name: preprocess(patch, self.input_size)})[0][0]
        logits = np.asarray(logits, dtype=np.float32)
        exp = np.exp(logits - logits.max())
        probability = exp / exp.sum()
        best = int(probability.argmax())
        return self.classes[best], float(probability[best])

    def predict(self, image: np.ndarray, index: int) -> tuple[str, float]:
        """从整屏截图里切出第 index 个候选再分类。"""
        return self.predict_patch(crop_slot(image, index))

    def score_patch(self, patch: np.ndarray) -> tuple[int, str, float]:
        """返回 (稀有度分, 类别, 置信度)，输入是已裁好的候选图。"""
        label, confidence = self.predict_patch(patch)
        return CLASS_SCORE.get(label, 2), label, confidence

    def score(self, image: np.ndarray, index: int) -> tuple[int, str, float]:
        """返回 (稀有度分, 类别, 置信度)，输入是整屏截图。"""
        return self.score_patch(crop_slot(image, index))

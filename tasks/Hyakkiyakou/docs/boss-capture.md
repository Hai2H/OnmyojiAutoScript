# 鬼王候选采集与标注流程

目标：让脚本认得出"三个候选里哪个最稀有"。这一步**不需要认人**，只需要稀有度分类，
所以标注成本很低（每张图按一个键）。

## 1. 开启采集

设置 → 百鬼夜行 → 调试配置 → **保存鬼王候选截图**（默认开启，`hya_boss_capture`）。

每局开始新建一个文件夹，点开每个候选后各存一张整屏截图：

```
log/hyakkiyakou/boss_capture/2026-10-25_14-28-59_001/   ← 本地时间 + 第几局
    candidate_1.jpg
    candidate_2.jpg
    candidate_3.jpg
    meta.json      ← 旧模型的判断结果（仅参考，不是标签）
```

跑 10 局就会有 10 个这样的文件夹、共 30 张图。检查方法：

```bash
ls log/hyakkiyakou/boss_capture/*/candidate_*.jpg | wc -l    # 应该是 局数 × 3
```

约 0.7MB/局（JPEG 95）。30 局 ≈ 21MB。不需要采集时把开关关掉即可。

> 不要在这里手写标签。`meta.json` 里 `old_*` 字段是旧模型猜的，它对新式神基本是错的。

## 2. 工作台：标注 / 数据 / 训练 / 评估

```bash
.venv/Scripts/python.exe dev_tools/boss_workbench.py
# 浏览器自动打开 http://127.0.0.1:8017
```

四个页签：

| 页签 | 能做什么 |
|---|---|
| 标注 | 键盘 `1~4` / `0` 打标签，**立即写入** `labels.json`；已标注的自动跳过 |
| 数据 | 每类数量与达标进度（目标每类 20 / 总数 60）；重新裁图；构建 train/val |
| 训练 | 调 epochs / folds / backbone / device，后台跑 GPU 训练，实时看日志 |
| 评估 | 拖阈值看回退率、单张准确率、每局三选一是否选对；阈值可保存进 `model.json` |

评估页会同时显示 **CV（泛化）** 和 **基线**：CV 不高于基线时页面会标红，任务里也会
自动禁用这个模型（`boss_rarity.BossRarityModel.usable`），避免它把鬼王选坏。

如果不想开服务，也可以用命令行 + 离线页：

```bash
.venv/Scripts/python.exe dev_tools/boss_dataset.py crop
```

- 按三个固定槽位（1280×720）裁剪，输出到 `data/boss_rarity/crops/`
- 生成 `data/boss_rarity/label.html`
- 完全重复的图默认跳过；不同式神一律保留（`--no-dedupe` 可关掉去重）
- 界面分辨率变了才需要改脚本里的 `SLOTS`，不用重跑游戏

## 3. 标注

双击 `data/boss_rarity/label.html`，用浏览器打开：

| 键 | 含义 |
|---|---|
| `1` | UR/SP 级（最值钱，值得选作鬼王） |
| `2` | SSR |
| `3` | SR |
| `4` | R/N/G（不值得选） |
| `0` | 跳过：画面没加载完、半透明还在淡入、看不清 |

- 一张图按一个键就自动跳下一张，90 张大约 3~5 分钟
- 标注存在浏览器本地，中途关掉不会丢
- 标完点 **导出 labels.json**，把文件放到 `data/boss_rarity/`

**为什么把 UR 并进 SP**：现在全服只有 2 个 UR，跑 30 局可能一张都碰不到。样本不够会把
这一类学坏，所以先合成"UR/SP"，等攒到 20 张以上再单独拆出来。

**只标稀有度、不标名字**：鬼王选择只需要比较"谁更值钱"，不需要知道具体是谁。这样
一个分类模型就够，不用检测框、不用跟踪。

## 4. 整理成训练集

```bash
.venv/Scripts/python.exe dev_tools/boss_dataset.py build --labels data/boss_rarity/labels.json
.venv/Scripts/python.exe dev_tools/boss_dataset.py stats
```

输出（标准的按文件夹分好类的图像分类数据集，可以直接喂 ImageFolder / YOLOv8-cls）：

```
data/boss_rarity/
├── train/UR_SP/*.png   train/SSR/*.png   train/SR/*.png   train/below/*.png
├── val/  ...
├── labels.csv          # file,label,split,run
└── summary.json        # 各类数量、跳过数
```

- 按文件名哈希固定切 85% / 15%，重复运行结果一致
- `stats` 会提示训练样本少于 20 的类别

**数量建议**：每类至少 20 张，总数 ≥ 60 张。30 局大约 90 张，分布通常够用；如果
`SSR` 明显偏少，就再多跑几局，或把 `SR` 和 `below` 合并。

## 5. 之后

有数据之后接一个小分类模型（224×224，4 类），再接回 `_detect_current_rarity()`：
裁槽 → 分类 → 稀有度分。在那之前，鬼王选择仍然是"认不出就选第 1 个"，采集不影响现有流程。

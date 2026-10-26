# 撒豆采集与标注流程（新式神检测）

鬼王那一套是"整图分类"，撒豆是"检测 + 跟踪"，所以数据形态不同：要整帧原图 + 框。

## 1. 开启采集

设置 → 百鬼夜行 → 调试配置：

| 参数 | 默认 | 说明 |
|---|---|---|
| 采集整局撒豆画面 | 关 | 打开后每局都会存，建议长期开着 |
| 采集间隔(ms) | 1000 | 一只式神横穿 3~5 秒，1 秒一帧能拿到 3~5 个角度；再密只是重复 |
| 截图 JPEG 质量 | 88 | 1280×720 约 120~180KB/帧 |
| 空场景间隔(秒) | 5 | 画面上一个目标都没有时，每 N 秒存一张当负样本 |
| 写旧模型预标注 | 开 | 在 `prelabel/` 生成 X-AnyLabeling 可读的 JSON |

一局 5 分钟约 **300 帧 / 40~55MB**；10 局约 **0.5GB**。每局一个目录：

```
data/bean_capture/
├── xany_classes.txt                        ← 268 类，X-AnyLabeling 用
└── 2026-10-25_21-03-11_r001/
    ├── raw/frame_000001.jpg                ← 原图，不画框
    ├── prelabel/frame_000001.json          ← 旧模型的框（仅建议）
    ├── frames.jsonl                        ← 每帧的跟踪结果
    └── round.json                          ← 本局参数与帧数
```

**为什么原图和预标注分开存**：

- 原图不带任何涂改，以后换模型可以重新预标注，不会把旧框"焊死"在图上；
- 预标注只是给标注工具一个起点，改起来比从零画快得多；
- `frames.jsonl` 保留每帧的跟踪 id，之后排查"模型把 A 认成 B"时能对上时间线。

## 2. 看这 10 局攒了什么

```bash
.venv/Scripts/python.exe dev_tools/bean_dataset.py stats
```

输出每局的帧数、体积、可疑帧数，以及旧模型认出的类别分布。

**可疑帧**的判定：同一 track id 的类别在相邻帧跳变（+2 分）、最低置信度 < 0.60（+1）、< 0.45（再 +1）。
新式神通常表现为"类别乱跳 + 置信度低"，所以这部分优先标注。

## 3. 导出给 X-AnyLabeling

```bash
# 先只导出可疑帧（推荐，标注量小、命中率高）
.venv/Scripts/python.exe dev_tools/bean_dataset.py prepare --suspects-only

# 或者全量导出
.venv/Scripts/python.exe dev_tools/bean_dataset.py prepare
```

产物在 `data/bean_label/raw/`：每张 jpg 旁边有同名 json，X-AnyLabeling 打开这个文件夹就能看到预框。

## 4. 标注

1. X-AnyLabeling → 打开目录 → 选 `data/bean_label/raw`
2. 导入类别：用 `data/bean_label/classes.txt`（格式 `label 名称`，如 `sp_051 百羽凤凰火`）
3. 逐张核对：
   - 预框 **正确** → 保留
   - 预框 **错类别** → 改类别
   - **新式神没被框到** → 手动补框并选对应类别
   - **误检**（把特效/纸人当式神）→ 删除该框
4. 全部确认后导出 YOLO 格式（矩形框）

> 关键：**没被旧模型框到的目标也要补**。新式神几乎一定属于这一类，漏补等于白标。

## 5. 训练与上线（待做）

- 训练：YOLOv8n（ultralytics 装进 `.venv-train`），导出 ONNX
- 上线：自建 onnxruntime 推理 + 轻量 IoU 跟踪，输出与 `oashya.tracker` 相同的
  `(id, cls, conf, cx, cy, w, h, v)`，`agent/focus.py` 不用改
- 合并：新模型置信度高时覆盖 oashya 的类别，否则保留 oashya 的判断

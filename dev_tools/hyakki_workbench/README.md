# Hyakki Workbench

本机百鬼夜行 YOLO11 数据与训练工作台。

## 启动

```powershell
pwsh -File .\dev_tools\hyakki_workbench\start.ps1
```

默认打开 <http://127.0.0.1:8765>，只监听本机。可通过参数修改端口：

```powershell
pwsh -File .\dev_tools\hyakki_workbench\start.ps1 --port 8877
```

首次配置独立 GPU 推理环境：

```powershell
pwsh -File .\dev_tools\hyakki_workbench\setup_infer_env.ps1
```

## 工作台状态

工作台自己的状态和后续数据集版本写入：

```text
data/hyakki_workbench/
```

该目录是本地运行数据，不提交到 Git。原始撒豆采集仍位于
`data/bean_capture/`，鬼王候选仍位于 `log/hyakkiyakou/boss_capture/`。

## 当前阶段

- 扫描撒豆 `frames.jsonl`，按旧模型 track ID 聚合成目标出现事件。
- 扫描鬼王候选截图及 `meta.json`。
- 使用 268 类登记表补充名称和稀有度。
- 将 SSR、SP、UR、低置信度和类别跳变事件加入审核队列。
- 提供本地 FastAPI API 和流程总览页面。
- 按一次目标出现聚合 1～3 张代表帧，记录审核动作和完整历史。
- 生成 X-AnyLabeling 目录与旧模型预标注，不覆盖已有 JSON。
- 可按 268 类登记表改具体式神，或标记 UNKNOWN、待重标、跳过。
- 导入 X-AnyLabeling 已人工确认的矩形框；没有人工框的鬼王候选不会进入训练集。
- 按独立事件统计 20 / 50 / 100 三档样本门槛。
- 以整局为最小单位拆分 train / val / test，生成带内容哈希的不可变数据集版本。
- 自动加入有限数量的空场景负样本，并写入 data.yaml、manifest 和类别映射。
- 使用 `.venv-train` 在 RTX GPU 上后台训练 YOLO11n / YOLO11s，页面关闭不终止任务。
- 单 GPU 队列、实时日志、取消任务和服务重启后的任务状态恢复。
- 在 val 上按类别校准高精度阈值，只有手动点击后才用冻结 test 集做最终验收。
- 发布前检查每类 Precision ≥ 95%、Recall ≥ 90% 和 P95 延迟 ≤ 180ms，通过后同时保存 PyTorch 与 ONNX 权重。
- 独立本地 GPU 推理进程直接执行 ONNX Runtime；OAS 侧提供旧模型、旁路、补充和新模型优先四种模式。
- 新服务超时、异常或未启动时，当前帧继续使用旧 `oashya.tracker`。

后续维护由“扫描采集 → 高稀有度审核 → 数据版本 → 训练 → test → 发布”循环完成。

## 日常操作

1. 在 OAS 设置中开启“采集整局撒豆画面”和“保存鬼王候选截图”，正常运行百鬼夜行。
2. 打开工作台，扫描采集数据。SSR / SP / UR、低置信度和类别跳变会进入审核队列。
3. 在“审核”页确认旧模型是否正确；需要补框或重新标注时生成标注目录。
4. 在 X-AnyLabeling 中打开生成的目录，核对矩形框和具体式神类别后保存。
5. 回到工作台点击“导入标注结果”。预标注不会被当作人工真值。
6. 在“数据集”页选择达到 20 个独立事件的类别，构建不可变 train / val / test。
7. 在“训练与评估”页选择 `yolo11n` 或 `yolo11s`，GPU 后台训练并查看逐类指标。
8. 训练完成后手动运行最终 test。只有每类 Precision ≥ 95%、Recall ≥ 90%，且 P95 延迟不超过 180ms 才能发布。
9. 在“上线与回退”页发布模型、启动 GPU 推理服务并选择运行模式。
10. OAS 的 `Auto` 模式跟随工作台设置；新服务异常时当前帧继续使用旧模型。

X-AnyLabeling 官方仓库：<https://github.com/CVHub520/X-AnyLabeling>

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

新模型推理与新旧模型共存接入在后续迭代完成。

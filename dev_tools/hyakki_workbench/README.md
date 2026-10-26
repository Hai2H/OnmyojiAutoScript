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

审核、数据集版本、GPU 训练与运行时接入在后续迭代完成。

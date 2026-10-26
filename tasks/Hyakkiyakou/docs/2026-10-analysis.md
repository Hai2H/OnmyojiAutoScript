# 百鬼夜行：现状梳理与"新式神识别"方案

分析对象：当前仓库 `dev` 分支（HEAD `38bed4b9`）
分析时间：2026-10-25（当日已把本地 WIP 全部回滚，工作区与 HEAD 一致；备份见 `log/backup/20261025T130915-local-wip/`）
结论一句话：**任务流程还在维护，真正停更的是模型和类别表（oashya 0.0.7，2024-06-25，219 类）。49 个式神类别模型根本表示不了；"识别不到就整局退出"是回滚前本地 WIP 的行为，HEAD 则是静默选第 1 个——两种都不好，只是坏法不同。**

---

## 一、结论先行

| # | 事实 | 影响 |
|---|---|---|
| 1 | 依赖包 `oashya` 最后一版是 **0.0.7（2024-06-25）**，类别表固定 **219 类**，最新只到 2024-02-07（SP 鲸汐千姬 / 福悦座敷童子、SSR 泷） | 之后上线的式神一律无法表示，只能漏检或被判成最像的旧式神 |
| 2 | 截至 2026-10-25 缺 **49 类**：UR 2、SP 16、SSR 30、SR 1 | 每年还会再加 10~17 类，缺口只会变大 |
| 3 | 游戏在 **2025-10-29 新增 UR 稀有度**，且**UR 碎片可以从百鬼夜行获取** | 现在最值钱的目标恰恰是模型完全学不到的那一类 |
| 4 | 模型文件是加密/封装过的（`onnxruntime` 直接 `INVALID_PROTOBUF`），推理与跟踪都在编译好的 `tracker.cp310-win_amd64.pyd` 里 | **不能简单换一个 ONNX 就完事**，这是所有方案的硬约束 |
| 5 | 鬼王页对 219 类检测器是**分布外**的：实测 conf 降到 0.3、候选槽裁出来放大到 640，仍然 0 检出 | "识别率低"的根因不只是新式神，而是渲染方式不同 |
| 6 | HEAD 的兜底是"三个都认不出就选第 1 个"（不报错、不留证）；回滚前的 WIP 改成"认不出就停" | 一个会选错、一个会停摆，都不能长期用 |
| 7 | 童男/童女（`R_007/R_008`）被排除后稀有度算出 `-1`，在"取最高分"里等价于"未知" | 会让三选一错判（HEAD 不中止，只是选错） |

---

## 二、现有逻辑地图

### 2.1 文件职责

| 文件 | 职责 |
|---|---|
| [script_task.py](tasks/Hyakkiyakou/script_task.py) | 入口、单局流程、鬼王选择、撒豆主循环 |
| [agent/agent.py](tasks/Hyakkiyakou/agent/agent.py) | 势场 `gamma`、`argmax` 选焦点、对外 `decision` |
| [agent/focus.py](tasks/Hyakkiyakou/agent/focus.py) | 单帧收益 `omega`、buff 收益、`r()` 出手阈值与惩罚 |
| [slave/hya_slave.py](tasks/Hyakkiyakou/slave/hya_slave.py) | 豆子数、剩余式神数、buff 槽、5↔10 豆、邀请好友 |
| [slave/hya_device.py](tasks/Hyakkiyakou/slave/hya_device.py) | 快速截图（0.2s 节流）与窗口直发点击 |
| [slave/hya_color.py](tasks/Hyakkiyakou/slave/hya_color.py) | buff 槽颜色快判（空槽短路） |
| [debugger.py](tasks/Hyakkiyakou/debugger.py) | 可视化线程、连续学习存图、结算截图 |
| [config.py](tasks/Hyakkiyakou/config.py) | 权重、优先级、阈值、调试开关 |

### 2.2 单局流程

1. `run()`：`page_onmyodo` → `switch_onmyoji` → `page_hyakkiyakou`，之后按 `hya_limit_count` / `hya_limit_time` 循环调用 `one()`（[script_task.py:120](tasks/Hyakkiyakou/script_task.py#L120)）。
2. `one()`：`I_HACCESS` → 可选邀请好友 → `I_HSTART` → 等 `I_HTITLE`（[script_task.py:239](tasks/Hyakkiyakou/script_task.py#L239)）。
3. 鬼王三选一：逐个点击 `C_HSELECT_1/2/3` → `sleep(1)` 等刷新 → `_detect_current_rarity()`（内部调用 `tracker(...)`，**不是**单帧 detect）→ `_rarity_score` 打分 → 取最高分；**三个都失败时 `best_idx` 保持 0，即静默选第 1 个**（[script_task.py:208](tasks/Hyakkiyakou/script_task.py#L208)）。
4. 回到 `I_HTITLE` 循环，直到标题消失（期间点 `I_HSTART`，并兜底补点一次最佳候选）（[script_task.py:251](tasks/Hyakkiyakou/script_task.py#L251)）。
5. 撒豆主循环：`fast_screenshot` → 出现 `I_HEND` 结束；没有 `I_CHECK_RUN` 就 `continue`；否则 `update_state()` → `tracker(image=..., response=last_action)` → `agent.decision()` → `do_action()`（[script_task.py:268](tasks/Hyakkiyakou/script_task.py#L268)）。
6. 结算：`logger.info('Hyakkiyakou End')` → `I_HEND` → `I_HACCESS` → 回主界面点关闭红叉（[script_task.py:300](tasks/Hyakkiyakou/script_task.py#L300)）。

### 2.3 决策数学（真正决定"砸不砸"的部分）

**候选焦点评分**（[agent.py:82](tasks/Hyakkiyakou/agent/agent.py#L82)）：

- 每个目标在 `z` 上叠一个高斯核，`mu = 0.8 + cx*0.2/1280`（越靠右越优先，给飞行留提前量）。
- 权重：SP/SSR `1.5×`，SR `0.7`，R `0.3`（可配），N/G `0`（可配，默认不砸）；`hya_priorities` 命中的类别**强制 1.7**（会覆盖 SP 的 1.5）。
- `argmax(z)` → 找最近轨迹 → `Focus`；`omega < Agent.OBSERVE_THRESHOLD (0.6)` 就放弃目标（[agent.py:58](tasks/Hyakkiyakou/agent/agent.py#L58)、[agent.py:147](tasks/Hyakkiyakou/agent/agent.py#L147)）。

> 注意这里有个隐性耦合：单目标时 `omega ≈ mu × weight`。默认权重下 **SR 只有右半屏（cx ≳ 364）才可能过 0.6 阈值，R/N/G 永远过不了**。也就是说"权重"和"是否看得见"被同一个 0.6 绑死，改 `hya_r` 想砸 R 卡并不会生效，除非同时改阈值——而阈值是写死的常量，不在配置里。

**出手判定**（[focus.py:127](tasks/Hyakkiyakou/agent/focus.py#L127)）：

```text
omega  = max(目标收益, buff 收益)
tau    = -140 / (max(101, Δt_ms) - 100)          # 冷却项，防止连砸
upsilon= 100 * (豆子/250 - 剩余数/35)² 带符号   # 豆多进度低→更想砸
r      = omega + tau + upsilon - 0.6
惩罚：SSR/SP 且无"概率UP" → -999；冰冻 → -999
```

**buff 收益**（[focus.py:100](tasks/Hyakkiyakou/agent/focus.py#L100)）：豆子获取 2.2、概率UP 2.0、好友UP 2.0（限左半屏且开启邀请）、减速 2.0（仅当自身收益 ≥1.5）、砸豆加速 2.0（左半屏）。

### 2.4 状态读取

- **剩余式神数**：0~9 的模板匹配，两位数用"十位 + 个位"两张图组合（[hya_slave.py:118](tasks/Hyakkiyakou/slave/hya_slave.py#L118)）。
- **豆子数**：先按 `current / -10 / -20` 猜，模板验证，失败退 OCR（[hya_slave.py:134](tasks/Hyakkiyakou/slave/hya_slave.py#L134)）。
- **5/10 豆**：两张图标模板；**buff**：4 个槽，颜色快判 + 5 个模板（概率UP = `BUFF_STATE6`）。
- 一局 36 只、鬼王 3 选 1 三倍碎片，和代码里的初值 `[250, 36, 10, ...]` 对得上。

### 2.5 证据链（回滚后**不存在**，但值得重建）

回滚前那版 WIP 曾实现：鬼王每个候选都存 RGB 截图 + 原始检测 + 稀有度 + `annotation: pending`（`log/hyakkiyakou/boss_selection/<时间>/`），以及整局采集（原图 + X-AnyLabeling 预标注 + `frames.jsonl`，落 `data/shikigami_dataset/captures/<时间>/`）。

现在 HEAD 里既没有 `SelectionRecord`，也没有 `hya_collect_round` 配置。**这是后面所有诊断的前提**：没有证据就只能靠猜。备份里的 `selection_record.py` 可以直接复用（它本身不依赖 `script_task` 的其它改动）。

---

## 三、问题清单（按严重度）

| 级别 | 问题（HEAD 基线） | 位置 |
|---|---|---|
| P0 | 鬼王页 0 检出时**静默选第 1 个**：不报错、不留证据，玩家只会觉得"鬼王老是选错" | [script_task.py:229](tasks/Hyakkiyakou/script_task.py#L229) |
| P0 | 鬼王页对检测器 OOD：候选是半透明全息正面大图，与撒豆页的实体小图不同分布（见附：实测） | [script_task.py:182](tasks/Hyakkiyakou/script_task.py#L182) |
| P0 | 冰冻策略是死代码：外层 `if not appear(I_HFREEZE)` 已保证非冰冻，内层再判一次 `freeze` 恒为 False | [script_task.py:280](tasks/Hyakkiyakou/script_task.py#L280) |
| P1 | 没有 UR 稀有度，也没有"未知"这一类；模型对 49 个类别完全无表示 | 全局 |
| P1 | 童男/童女被排除在 R 之外后 `_rarity_score` 返回 -1，等价"未知"，会让三选一错判 | [script_task.py:170](tasks/Hyakkiyakou/script_task.py#L170) |
| P1 | 主循环 `if not I_CHECK_RUN: continue` 没有退出条件，只有 3 分钟黑屏计时器兜底，弹窗会空转 | [script_task.py:272](tasks/Hyakkiyakou/script_task.py#L272) |
| P1 | 没有任何证据留存：选错也无法复盘，采集开关也不存在 | 全局 |
| P2 | `hya_priorities` 的 1.7 会压过 SP 的 1.5，"SP 最优先"的直觉不成立 | [agent.py:107](tasks/Hyakkiyakou/agent/agent.py#L107) |
| P2 | 观测阈值 0.6 写死，与权重耦合（见 2.3） | [agent.py:58](tasks/Hyakkiyakou/agent/agent.py#L58) |

---

## 四、网络侧：这两年半发生了什么

### 4.1 依赖与上游

| 对象 | 状态 |
|---|---|
| `oashya`（模型 + 跟踪器 + 类别表） | PyPI 只有 0.0.6/0.0.7，均为 **2024-06-24/25** 上传，之后再无版本（[PyPI JSON](https://pypi.org/pypi/oashya/json)、[piwheels](https://www.piwheels.org/project/oashya/)） |
| 上游 `tasks/Hyakkiyakou` | **仍在维护**：2025-09-30 修邀请界面死循环（[#1095](https://github.com/runhey/OnmyojiAutoScript/issues/1095)）、2025-10-29 补 `hya_check_invitation.png`、2025-11-08 加切换阴阳师、2026-01-07 鬼王三选一增强（`xxz7528`）、2026-07-19 召回活动 ROI、2026-09-18 GameUi 迁移 |
| 社区 | 没有新的"百鬼夜行专用"开源模型或项目；OAS 里 56 个提到百鬼夜行的 issue 基本都是 UI/导航/等待类，没人碰模型 |

也就是说：**"很久没人更新"指的是模型和类别表，不是任务代码。** 任务代码跟着游戏 UI 在动，模型停在 2024 年。

### 4.2 模型本身的两个硬事实

1. 打包的 `oashya_fp32.onnx` **不是标准 ONNX**（文件头是 `gAAAAAB...` 的 Fernet 风格 base64 密文，`onnxruntime` 解析直接报 `INVALID_PROTOBUF`），解密逻辑编译在 `tracker.cp310-win_amd64.pyd` 里——属于作者的防篡改/防替换设计。
2. 同目录的 `oashya_int8.onnx` 是一份**误打包的 Python 源码**，可以看出预处理是 640 尺寸输入、按 `conf/iou` 过滤、并丢弃 `w<=50 且 h<=50` 的小目标，末尾还留着 `Watcher().run()` 的调用。

结论：**想换模型，只能绕开 `oashya.tracker`，自己实现推理 + 跟踪。** 这决定了后面所有方案的成本结构。

### 4.3 游戏侧的变化（关键）

- 2024-03 至今新增 **49 个**式神（含联动），其中本家 SSR 19 个、SP 16 个、SR 1 个。
- **2025-10-29 新增稀有度 UR**（妖刀姬·绯夜猎刃），2026-10-28 第二个 UR（大天狗·高天铁羽）——稀有度体系本身变了。
- **UR 碎片可以通过百鬼夜行获取**（[9game 2025-10-22](https://www.9game.cn/yys/11466334.html)），联动碎片也有概率砸到（[9game 2025-11-15](https://www.9game.cn/yys/11523378.html)）。
- 节奏：每年本家 SP 4~6 个、SSR 5~9 个，加上联动与 UR，**每年 10~17 个新类别**。

也就是说，即使今天把缺口补齐，**三个月后又会缺 3~4 个**。所以真正要解决的不是"补一次模型"，而是"让新式神进来时不炸、并且能被收进训练闭环"。

---

## 五、怎么"识别到新式神"——三条路线

### 路线 C：先止血（1 天，零模型改动，建议立刻做）

不追求"认识新式神"，先做到"**遇到不认识的不要死**"：

1. `_rarity_score` 遇到未知返回**配置化的默认稀有度**（默认按 SSR 处理），而不是 -1。
2. `_select_best_boss` 保持"不中止"，但把兜底从"静默选第 1 个"改成"记录 `unknown` + 按默认值参与比较 + 日志告警"；严格模式做成开关（默认关闭）。
3. 撒豆侧：未知目标按"SSR 且无概率UP → 不砸"的保守策略处理，宁可少砸也不乱砸。
4. 修复童男/童女：`_rarity_score` 里把 `R_007/R_008` 按 `R` 计分（它们本来只是不该被优先砸，而不是"身份不明"）。

收益：新式神上线当天不会再整局中止；代价：新式神的价值仍可能被低估。这是**用精度换可用性**，符合"脚本要能一直跑"的目标。

### 路线 B：检测与身份解耦（2~4 周，推荐中期方案）

核心思想：**"这是谁"和"这是不是式神"分开，稀有度不靠模型猜，靠名字查表。**

```text
截图 → ① 类别无关检测（式神 / 纸人 / buff）
     → ② 跟踪（ByteTrack / OC-SORT，给每个目标稳定 ID 与速度）
     → ③ 身份判定：
            a. 旧分类头置信高 → 采信
            b. 否则用图库 embedding 最近邻（DINOv2 / CLIP / ResNet 特征）
            c. 都不过 → UNKNOWN
     → ④ 稀有度 = lookup(名字)；UNKNOWN 用默认策略
     → ⑤ 交给现有 agent/focus 决策（这部分完全不用改）
```

为什么可行：

- 检测层只学"一个飞行中的式神长什么样"，**不随新式神变化**，训练数据需求骤降。
- 身份层用"图库 + 最近邻"，**新式神只要加一张参考图**（wiki 立绘、或一局截图裁切），不需要重训。
- 稀有度来自查表，天然支持新增 UR 这种"新档位"。

配套：备份里的 `dev_tools/hyakki_video.py`（本次回滚时移入 `log/backup/20261025T130915-local-wip/untracked/`）已经有"标准 YOLO ONNX"分支和离线预览，恢复后扩成在线 detector（缺的只是跟踪）就能替换掉封闭的 `oashya.tracker`。

### 路线 A：重训同构模型（219 → 268+）

只在"必须精确到具体式神"（比如 `hya_priorities` 精确点名单个式神）时才值得做：

- 类别 id 必须**只在尾部追加**，旧的 219 个保持不动，否则历史标注全部错位。
- 训练走仓库里现成的 `data/shikigami_dataset/scripts/01_prepare → 02_train → 03_export`。
- 增量策略：新增类时冻结 backbone、只训分类头 + 少量微调，避免旧类灾难性遗忘。
- 但：**训练出来的 ONNX 无法喂给 `oashya.tracker`**（加密 + 闭源），必须配合路线 B 的自建推理。

**推荐节奏：C 立刻上 → B 作为主方向 → A 按需。**

---

## 六、落地改造点（文件级）

1. **新增 `tasks/Hyakkiyakou/shikigami_table.py`**：`id → (label, name, rarity, group)`。旧 219 个 id 原样保留，UR 段与新增 SP/SSR/SR 追加在尾部；`CLASSINDEX` 的 MIN/MAX 改成由表推导，避免再加一个稀有度就改一次魔法数字。
2. **`_rarity_score` 改查表**，未命中返回默认值；`_select_best_boss` 的"全失败选第一个"改为"记录 unknown + 默认分"，严格模式做成开关（[script_task.py:208](tasks/Hyakkiyakou/script_task.py#L208)）。
3. **补诊断信号**：top-1 置信度、top-1/top-2 间隔、目标宽高比与面积、是否命中已知类别——这四个量是"未知检测"最便宜的代理指标，先记录，用来定阈值。
4. **把 `selection_record.py` 从备份恢复回来，并给 `RoundCapture` 加 `identity_status`**（`known / low_conf / unknown`），这样采集数据能直接筛出"新式神候选帧"。
5. **`Agent.OBSERVE_THRESHOLD` 进配置**，把"权重"和"可见性"解耦。
6. **周期性缺口检查**：`dev_tools/check_shikigami_gap.py`（本次新增）每周跑一次，输出缺失清单；`--icons` 可顺带把官方立绘抓下来做参考图库。
7. **训练与验证**：沿用现有四阶段脚本；验证看三个指标——每局 `unknown` 率、鬼王候选 `unknown` 率、误选率（用 `record.json` 对比最终收益）。

### 兜底逻辑的骨架

```python
RARITY_SCORE = {'G': 0, 'N': 1, 'R': 2, 'SR': 3, 'SSR': 4, 'SP': 5, 'UR': 6}
UNKNOWN_SCORE = RARITY_SCORE[config.hya_unknown_rarity]  # 默认 'SSR'

def rarity_score(class_id) -> tuple[int, str]:
    row = SHIKIGAMI_TABLE.get(class_id)          # 表里没有 → 新式神或误检
    if row is None:
        logger.warning(f'unknown class {class_id}, treat as {config.hya_unknown_rarity}')
        return UNKNOWN_SCORE, 'unknown'
    return RARITY_SCORE[row.rarity], 'known'
```

在没有新模型之前，"未知"只能用代理指标判断（这几项现在都能从 `Tracker.detect()` 的原始输出里拿到，不需要改模型）：

- 置信度落在 0.6~0.75 的灰区；
- 同一目标连续多帧的类别在跳变（跟踪器 ID 不变、类别不稳）；
- 目标框的宽高比/面积明显偏离该类别历史分布；
- 目标与已知类别的图像特征距离偏大（路线 B 落地后可直接用）。

前三项先记录、统计，等有了一批真实样本再定阈值，不要一上来就拍数字。

### 新增的检查脚本

```bash
# 只列缺口
.venv/Scripts/python.exe dev_tools/check_shikigami_gap.py
# 输出 JSON（含 wiki 链接、上线日期、联动标记）
.venv/Scripts/python.exe dev_tools/check_shikigami_gap.py --json data/shikigami/gap.json
# 顺便把缺失式神的官方立绘抓到本地，建参考图库
.venv/Scripts/python.exe dev_tools/check_shikigami_gap.py --icons data/shikigami/new_icons
```

数据源是 BWIKI 的 Semantic MediaWiki ask 接口（`[[单元::式神]][[稀有度::SP]]`），比爬 HTML 稳定；脚本会额外列出"日期早但对不上名字"的项，用来发现 oashya 的拼写差异（例如 `烬天玉澡前`、`饭筒`＝饭笥）和从未建模过的联动限定（例如 SSR `玉取` 其实在覆盖期内就缺了）。

---

## 七、缺口清单（截至 2026-10-25，共 49 个）

完整可机读版本见 [shikigami_gap.json](tasks/Hyakkiyakou/docs/shikigami_gap.json)。汇总：**UR 2 / SP 16 / SSR 30 / SR 1**。

<details>
<summary>展开完整列表</summary>

| 上线 | 稀有度 | 式神 | 备注 |
|---|---|---|---|
| 2024-03-06 | SSR | 初音未来 | 联动 |
| 2024-03-06 | SSR | 镜音铃·连 | 联动 |
| 2024-04-03 | SP | 晨晖惠比寿 |  |
| 2024-04-24 | SSR | 申屠子夜 | 联动 |
| 2024-05-22 | SP | 龙吟铃鹿御前 |  |
| 2024-06-19 | SSR | 猫川 |  |
| 2024-07-17 | SSR | 祸津神 |  |
| 2024-08-14 | SP | 遥念烟烟罗 |  |
| 2024-09-11 | SP | 心友犬神 |  |
| 2024-09-11 | SSR | 龙珏 |  |
| 2024-10-09 | SSR | 封阳君 |  |
| 2024-11-06 | SSR | 鬼金羊 |  |
| 2024-12-04 | SP | 神酿星熊童子 |  |
| 2024-12-31 | SSR | 夏目&猫老师 | 联动 |
| 2025-01-22 | SP | 晴思日和坊 |  |
| 2025-01-22 | SP | 瑶音紧那罗 |  |
| 2025-02-26 | SSR | 歌留多 |  |
| 2025-03-26 | SSR | 巡音流歌 | 联动 |
| 2025-04-23 | SSR | 卑弥呼 |  |
| 2025-05-21 | SP | 时曜泷夜叉姬 |  |
| 2025-06-18 | SSR | 荒骷髅 |  |
| 2025-07-16 | SP | 云间不见岳 |  |
| 2025-07-30 | SR | 湍津姬 |  |
| 2025-08-13 | SSR | 木之本樱 | 联动 |
| 2025-08-13 | SSR | 李小狼 | 联动 |
| 2025-09-10 | SP | 妙主九命猫 |  |
| 2025-09-10 | SSR | 雪御前 |  |
| 2025-10-10 | SSR | 平将门 |  |
| 2025-10-29 | UR | 妖刀姬·绯夜猎刃 | 首个 UR |
| 2025-11-19 | SP | 梦引蝴蝶精 |  |
| 2025-12-17 | SP | 梦山白藏主 |  |
| 2026-01-01 | SSR | 刻世命 |  |
| 2026-01-01 | SSR | 月城雪兔&月 | 联动 |
| 2026-01-01 | SSR | 知世&小可 | 联动 |
| 2026-01-14 | SSR | 神无月 |  |
| 2026-02-11 | SP | 灼华桃花妖 |  |
| 2026-02-11 | SSR | 葛叶 |  |
| 2026-03-11 | SP | 蚀月吸血姬 |  |
| 2026-04-01 | SSR | 市加美 |  |
| 2026-04-29 | SSR | 思金神 |  |
| 2026-05-27 | SP | 天火命铃彦姬 |  |
| 2026-06-24 | SSR | 毗沙门天 |  |
| 2026-07-22 | SSR | 不相狐禅 |  |
| 2026-08-19 | SSR | 洛天依 | 联动 |
| 2026-08-19 | SSR | 言和 | 联动 |
| 2026-09-09 | SP | 百羽凤凰火 |  |
| 2026-09-09 | SSR | 石长姬 |  |
| 2026-10-10 | SSR | 咲耶 |  |
| 2026-10-28 | UR | 大天狗·高天铁羽 | 尚未上线（wiki 日期） |

</details>

另外还有 **23 个"日期在覆盖期内但对不上名字"** 的条目，主要是联动限定（犬夜叉、杀生丸、桔梗、鬼灯、黑崎一护、灶门炭治郎/祢豆子、我妻善逸、嘴平伊之助、坂田银时、神乐&定春、奴良陆生、卖药郎、夜刀神、闻人翊悬、玉取 等）。其中 **`玉取`（2023-12-14）是真正的缺口**，其余多为联动或 oashya 的错别字（`饭筒` = 饭笥）。

---

## 八、每次大版本后的维护清单

- [ ] 跑 `check_shikigami_gap.py`，看新增了哪些类别
- [ ] 更新 `shikigami_table.py`（**只追加，不改旧 id**）
- [ ] 开启 `hya_collect_round` 打 5~10 局，收集新式神出现帧
- [ ] 用 X-AnyLabeling 核对预标注（旧模型给的不一定是正确答案）
- [ ] `01_prepare → 02_train → 03_export`，对比 `unknown` 率是否下降
- [ ] 回归检查页面模板：`I_HACCESS / I_HSTART / I_HSELECTED / I_HEND / I_HFREEZE` 是否被 UI 改版影响（这是 OAS issue 里最高频的故障）

---

## 九、已落地的第一步（2026-10-25）

**类别空间已扩到 268 类**（219 + 49），这是"能标注新式神"的前提：

- [shikigami_table.py](tasks/Hyakkiyakou/shikigami_table.py)：`id → label/name/rarity/released`，前 219 类与 oashya **逐条一致**（已断言校验），新类别按上线日期追加；`rarity_score()` 对表外 id 返回默认稀有度而不是 -1。注意**同一稀有度的 id 不再连续**，判断稀有度一律查表，不要用 MIN/MAX 区间。
- [gen_shikigami_table.py](dev_tools/gen_shikigami_table.py)：读缺口清单、只追加不改名，重复执行新增 0（幂等）。
- `data/shikigami_dataset/label_space/`：`classes.txt`、`xany_classes.txt`（X-AnyLabeling 用）、`classes.json`、`data.yaml`。
- [check_shikigami_gap.py](dev_tools/check_shikigami_gap.py)：现在区分"**模型缺 49 个**"和"**本地表也缺 23 个**"（联动限定 + `玉取`）。

**但真正的瓶颈是样本，不是代码**：现有标注只有 6 张图、6 个框，`runs/shikigami/full_classes/weights/best.pt` 远不足以替代 oashya。所以"直接重训 268 类"现在做不出来。

**推荐两段式**，把标注量压到最低：

```text
oashya 检测器（认 219 个旧类，已练过）
        +
新式神专用小模型（49 类，负样本 = 任意局内帧）
        ↓
合并：新模型置信高 → 覆盖 oashya 的类别；否则用 oashya
        ↓
UNKNOWN → 默认稀有度（不中止）+ 自动存图
```

这样只需标注**新式神**，不必重标旧 219 类；也可以先只做 5~10 个高频式神验证闭环。

---

## 附：本次分析的取证方式

- 代码：直接阅读上述文件；`_rarity_score` 的区间来自 `oashya.labels.CLASSINDEX`（`len(CLASSIFY) == 219`）。
- 模型：`onnxruntime.InferenceSession` 加载 `oashya_fp32.onnx` 报 `INVALID_PROTOBUF`，文件头为 Fernet 风格密文；`oashya_int8.onnx` 实为 Python 源码。
- 实测（撒豆页）：在本仓库的 `log/hyakkiyakou/video_tests/.../preview.png` 上跑 `Tracker`，预热后约 **54 ms/帧**（CPU onnxruntime），检测到 2 个目标，类别 id 落在 R 区间，与 `labels.py` 一致。
- 实测（鬼王页）：5 张 `log/hyakkiyakou/boss_selection/*/candidate_*.png`，conf 依次试 0.6 / 0.45 / 0.3，其中 2 张**任何阈值都是 0 检出**；把左槽裁出、缩放到 640×640 仍为 0。`Tracker.detect()` 只接受 640×640 或 1280×720 输入，模型原生输入 640。
- 游戏数据：BWIKI `SP/SSR/UR/SR/R` 页面的 SMW 查询（`?式神名`、`?上线日期`），共 251 条有日期的记录。
- 规则：UR 碎片可经百鬼夜行获取、联动碎片概率掉落、每局 36 只、鬼王三选一三倍碎片。

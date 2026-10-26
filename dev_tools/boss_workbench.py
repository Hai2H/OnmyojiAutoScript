"""鬼王稀有度工作台：一个本地页面完成 标注 / 看数据 / 训练 / 评估模拟。

启动：

    .venv/Scripts/python.exe dev_tools/boss_workbench.py
    # 浏览器打开 http://127.0.0.1:8017

页签：

- 标注：键盘 1~4 打标签、0 跳过，标签直接写入 data/boss_rarity/labels.json
- 数据：每类数量、达标进度、重新裁图、构建 train/val
- 训练：调 epochs/folds/backbone/device，后台跑 GPU 训练，看实时日志
- 评估：调阈值，看单张准确率、回退率、每局三选一是否选对（模拟运行时）

只监听 127.0.0.1。训练用 `.venv-train`，评估用当前 `.venv` 的 onnxruntime。
"""

import sys as _sys


def _scrub_training_paths() -> None:
    """把训练环境（.venv-train）从 sys.path 里摘掉。

    训练环境装的是 numpy 2.x，而运行环境的 cv2 / onnxruntime 是按 numpy 1.x 编译的。
    只要 `.venv-train\Lib\site-packages` 因为 PYTHONPATH 之类进了 sys.path，
    import numpy 就会拿到 2.x，然后 cv2 报 `_ARRAY_API not found`、onnxruntime 直接被
    吞掉变成 SystemError。必须在 import numpy 之前清理。
    """
    removed = [path for path in list(_sys.path) if 'venv-train' in path.replace(chr(92), '/').lower()]
    for path in removed:
        try:
            _sys.path.remove(path)
        except ValueError:
            pass
    if removed:
        print(f'已从 sys.path 移除训练环境路径（避免 numpy 2.x 混入）：{removed}')


_scrub_training_paths()

import argparse
import json
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import boss_dataset as dataset_tool  # noqa: E402
import boss_eval  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PYTHON = ROOT/'.venv-train/Scripts/python.exe'
TRAIN_LOG = ROOT/'log/workbench/train.log'
TARGET_PER_CLASS = 20
TARGET_TOTAL = 60

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>鬼王稀有度工作台</title>
<style>
 :root{--bg:#14161a;--panel:#1d2026;--line:#2c333d;--text:#e6e6e6;--dim:#9aa4b2;--ok:#2f7d4f;--warn:#8a6d2f;--bad:#7d2f2f}
 *{box-sizing:border-box}
 body{margin:0;font-family:system-ui,Segoe UI,Microsoft YaHei;background:var(--bg);color:var(--text)}
 header{padding:10px 16px;background:var(--panel);display:flex;gap:14px;align-items:center;flex-wrap:wrap;border-bottom:1px solid var(--line)}
 h1{font-size:16px;margin:0;font-weight:600}
 nav{display:flex;gap:6px}
 nav button{padding:7px 14px;border-radius:8px;border:1px solid var(--line);background:#232830;color:var(--text);cursor:pointer}
 nav button.on{background:#31465c;border-color:#41617f}
 .pill{background:#232830;border:1px solid #333a44;border-radius:999px;padding:3px 10px;font-size:13px;color:var(--dim)}
 .pill b{color:#9fd}
 section{display:none;padding:16px;gap:16px}
 section.on{display:flex;flex-wrap:wrap;align-items:flex-start}
 #stage{background:#0b0c0f;border:1px solid var(--line);border-radius:8px;padding:8px}
 #stage img{display:block;max-height:60vh}
 .side{flex:1 1 320px;min-width:300px}
 .btn{display:block;width:100%;margin:8px 0;padding:11px;border-radius:8px;border:1px solid #3a4048;background:#232830;color:var(--text);font-size:15px;cursor:pointer;text-align:left}
 .btn:hover{background:#2d333d}
 .btn.cur{outline:2px solid #4c9}
 kbd{background:#111;border:1px solid #444;border-radius:4px;padding:1px 6px;margin-right:8px;font-family:monospace}
 .row{display:flex;gap:10px;margin-top:12px;flex-wrap:wrap}
 .act{padding:9px 14px;border-radius:8px;border:1px solid #3a4048;background:#2b313a;color:var(--text);cursor:pointer;font-size:14px}
 .act:hover{background:#343b46}
 .act.primary{background:#2f7d4f;border-color:#3a9a62}
 .act:disabled{opacity:.5;cursor:not-allowed}
 .small{color:var(--dim);font-size:13px;line-height:1.7}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:260px;flex:1 1 300px}
 .card h3{margin:0 0 10px;font-size:14px;color:#c9d4e0;font-weight:600}
 .bar{height:8px;border-radius:6px;background:#232830;overflow:hidden;margin:6px 0}
 .bar>i{display:block;height:100%;background:#2f7d4f;width:0}
 .badge{padding:2px 8px;border-radius:999px;font-size:12px;border:1px solid transparent}
 .badge.ok{background:#1f3b2b;border-color:#2f7d4f;color:#8fe0ad}
 .badge.no{background:#3b2f1f;border-color:#8a6d2f;color:#e8c98a}
 #msg{margin-top:10px;padding:10px;border-radius:8px;background:#1b2026;border:1px solid var(--line);font-size:13px;white-space:pre-wrap;min-height:20px}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
 th{color:var(--dim);font-weight:500}
 tr.hit td:last-child{color:#8fe0ad}
 tr.miss td:last-child{color:#e8a0a0}
 pre{background:#0b0c0f;border:1px solid var(--line);border-radius:8px;padding:10px;font-size:12px;line-height:1.5;max-height:320px;overflow:auto;white-space:pre-wrap;margin:0}
 label.field{display:block;margin:10px 0;font-size:13px;color:var(--dim)}
 label.field input,label.field select{display:block;width:100%;margin-top:4px;padding:8px;border-radius:8px;border:1px solid var(--line);background:#0f1216;color:var(--text)}
 .kpi{display:flex;gap:18px;flex-wrap:wrap}
 .kpi div{min-width:110px}
 .kpi b{display:block;font-size:22px;color:#9fd}
 input[type=range]{width:100%}
</style>
<header>
 <h1>鬼王稀有度工作台</h1>
 <nav>
  <button data-tab="label" class="on">标注</button>
  <button data-tab="data">数据</button>
  <button data-tab="train">训练</button>
  <button data-tab="eval">评估</button>
 </nav>
 <span class="pill">进度 <b id="p-count">0/0</b></span>
 <span class="pill">模型 <b id="p-model">无</b></span>
</header>

<section id="tab-label" class="on">
 <div id="stage"><img id="pic" alt="候选图"></div>
 <div class="side">
  <div id="buttons"></div>
  <div class="row">
   <button class="act" id="prev">← 上一张</button>
   <button class="act" id="next">下一张 →</button>
   <button class="act" id="next-new">跳到未标注</button>
   <label class="small"><input type="checkbox" id="only-new"> 只看未标注</label>
  </div>
  <div id="msg" class="small">用键盘 1~4 打标签，0 跳过，← → 翻页。标签立即保存。</div>
 </div>
</section>

<section id="tab-data">
 <div class="card">
  <h3>样本数量 <span id="data-badge" class="badge no">未达标</span></h3>
  <div id="data-bars"></div>
  <p class="small">目标：每类 ≥ <b id="target-class">20</b> 张，总数 ≥ <b id="target-total">60</b> 张。
    SR+ 类达不到时模型只会"永远猜 below"，等于没训练。</p>
 </div>
 <div class="card">
  <h3>数据集</h3>
  <p class="small" id="data-paths"></p>
  <div class="row">
   <button class="act" id="crop">重新裁图</button>
   <button class="act primary" id="build">构建 train/val</button>
  </div>
  <div id="build-msg" class="small" style="margin-top:10px"></div>
 </div>
 <div class="card">
  <h3>每局候选</h3>
  <div id="runs" class="small"></div>
 </div>
</section>

<section id="tab-train">
 <div class="card">
  <h3>训练参数</h3>
  <label class="field">epochs（线性头迭代次数，600 够用）
   <input id="t-epochs" type="number" value="600" min="50" step="50"></label>
  <label class="field">folds（交叉验证折数，样本少时 5）
   <input id="t-folds" type="number" value="5" min="2" max="10"></label>
  <label class="field">backbone（特征提取器）
   <select id="t-backbone">
    <option value="mobilenet">mobilenet_v3_small（预训练，推荐）</option>
    <option value="none">小型 CNN（从零训练，样本少时没用）</option>
   </select></label>
  <label class="field">device
   <select id="t-device">
    <option value="cuda">cuda（GPU）</option>
    <option value="cpu">cpu</option>
   </select></label>
  <div class="row">
   <button class="act primary" id="train-start">开始训练</button>
   <button class="act" id="train-refresh">刷新日志</button>
  </div>
  <p class="small" id="train-state" style="margin-top:10px">空闲</p>
 </div>
 <div class="card" style="flex:2 1 520px">
  <h3>训练日志</h3>
  <pre id="train-log">（还没有开始训练）</pre>
 </div>
</section>

<section id="tab-eval">
 <div class="card">
  <h3>评估参数</h3>
  <label class="field">回退阈值 <b id="thr-value">0.60</b>
   <input id="thr" type="range" min="0.30" max="0.99" step="0.01" value="0.60"></label>
  <div class="row">
   <button class="act primary" id="eval-run">运行评估</button>
   <button class="act" id="thr-save">保存阈值到模型</button>
  </div>
  <p class="small">阈值越高越保守：低于它的候选不采信，运行时会回退 oashya。
   模型本身不比"永远猜多数类"好时，任务里会自动禁用。</p>
  <div id="eval-kpi" class="kpi" style="margin-top:12px"></div>
  <div id="eval-warn" class="small" style="margin-top:10px"></div>
 </div>
 <div class="card" style="flex:2 1 620px">
  <h3>模拟：每局三选一</h3>
  <div id="eval-rounds" class="small">点「运行评估」后显示</div>
 </div>
 <div class="card" style="flex:1 1 100%">
  <h3>单张结果</h3>
  <div id="eval-rows" class="small"></div>
 </div>
</section>

<script>
let state = {};
let cursor = 0, onlyNew = false, lastEval = null;

const el = id => document.getElementById(id);
const msg = text => { el('msg').textContent = text; };
const pct = value => Math.round(100*value) + '%';

function switchTab(name) {
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('on', b.dataset.tab === name));
  document.querySelectorAll('section').forEach(s => s.classList.toggle('on', s.id === 'tab-' + name));
  if (name === 'data') renderData();
  if (name === 'train') refreshTrain();
  if (name === 'eval') refreshEval();
}
document.querySelectorAll('nav button').forEach(b => b.onclick = () => switchTab(b.dataset.tab));

/* ---------------- 标注 ---------------- */
function visible() { return onlyNew ? state.items.filter(i => !i.label) : state.items; }

function renderLabel() {
  const stats = state.stats || {};
  el('p-count').textContent = (stats.labeled || 0) + '/' + (stats.total || 0);
  el('p-model').textContent = state.model ? (state.model.backbone || '有') : '无';
  const items = visible();
  if (!items.length) {
    el('pic').removeAttribute('src');
    msg.textContent = state.items.length ? '当前筛选下都标完了。' : '还没有候选图：去「数据」页点重新裁图。';
    return;
  }
  cursor = Math.max(0, Math.min(cursor, items.length - 1));
  const item = items[cursor];
  el('pic').src = '/crops/' + encodeURIComponent(item.id);
  document.querySelectorAll('.btn[data-label]').forEach(b => b.classList.toggle('cur', item.label === b.dataset.label));
}

async function putLabel(label) {
  const items = visible();
  if (!items.length) return;
  const item = items[cursor];
  const response = await fetch('/api/label', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: item.id, label})});
  if (!response.ok) { msg('保存失败：' + response.status); return; }
  const payload = await response.json();
  state.stats = payload.stats; item.label = label;
  if (!onlyNew && cursor < items.length - 1) cursor += 1;
  renderLabel();
}

function move(step) {
  const items = visible();
  cursor = Math.max(0, Math.min(items.length - 1, cursor + step));
  renderLabel();
}

function buildButtons() {
  const box = el('buttons');
  box.innerHTML = '';
  (state.label_list || []).forEach((label, index) => {
    const button = document.createElement('button');
    button.className = 'btn'; button.dataset.label = label;
    button.innerHTML = '<kbd>' + (index + 1) + '</kbd>' + label + ' · ' + (state.hints[label] || '');
    button.onclick = () => putLabel(label);
    box.appendChild(button);
  });
  const skip = document.createElement('button');
  skip.className = 'btn'; skip.dataset.label = 'skip';
  skip.innerHTML = '<kbd>0</kbd>跳过（没加载完 / 看不清）';
  skip.onclick = () => putLabel('skip');
  box.appendChild(skip);
}

el('prev').onclick = () => move(-1);
el('next').onclick = () => move(1);
el('next-new').onclick = () => {
  const items = visible();
  const index = items.findIndex(item => !item.label);
  if (index < 0) { msg('没有未标注的图片了。'); return; }
  cursor = index; renderLabel();
};
el('only-new').onchange = event => { onlyNew = event.target.checked; cursor = 0; renderLabel(); };
document.addEventListener('keydown', event => {
  if (document.querySelector('#tab-label').classList.contains('on') === false) return;
  if (event.key >= '1' && event.key <= '4') putLabel(['UR_SP', 'SSR', 'SR', 'below'][Number(event.key) - 1]);
  else if (event.key === '0') putLabel('skip');
  else if (event.key === 'ArrowRight') move(1);
  else if (event.key === 'ArrowLeft') move(-1);
});

/* ---------------- 数据 ---------------- */
function renderData() {
  const stats = state.stats || {};
  const counts = stats.counts || {};
  const labels = state.label_list || [];
  const bars = labels.map(label => {
    const count = counts[label] || 0;
    const ratio = Math.min(1, count/TARGET_PER_CLASS);
    return '<div style="margin:8px 0"><div class="small">' + label + '：' + count + ' / ' + TARGET_PER_CLASS + '</div>'
      + '<div class="bar"><i style="width:' + pct(ratio) + '"></i></div></div>';
  }).join('');
  el('data-bars').innerHTML = bars;
  const total = stats.labeled || 0;
  const perClassOk = labels.every(label => (counts[label] || 0) >= TARGET_PER_CLASS);
  const totalOk = total >= TARGET_TOTAL;
  const badge = el('data-badge');
  badge.className = 'badge ' + (perClassOk && totalOk ? 'ok' : 'no');
  badge.textContent = perClassOk && totalOk ? '已达标' : ('未达标（总数 ' + total + '/' + TARGET_TOTAL + '）');
  el('build-msg').textContent = stats.built
    ? ('上次构建：' + stats.built.total + ' 张，' + JSON.stringify(stats.built.counts)) : '还没构建过';
  const grouped = {};
  (state.items || []).forEach(item => {
    const run = item.run || '?';
    grouped[run] = grouped[run] || [];
    grouped[run].push(item);
  });
  el('runs').innerHTML = Object.keys(grouped).sort().reverse().slice(0, 20).map(run =>
    '<div>' + run + '：' + grouped[run].map(i => i.label || '—').join(' / ') + '</div>').join('');
}

async function action(name) {
  el('build-msg').textContent = '执行中：' + name + ' ...';
  const response = await fetch('/api/' + name, {method: 'POST'});
  const payload = await response.json();
  if (payload.error) { el('build-msg').textContent = payload.error; return; }
  if (name === 'crop') {
    el('build-msg').textContent = '裁图完成：' + payload.runs + ' 局 → ' + payload.kept + ' 张（去重跳过 ' + payload.skipped + ' 张）';
  } else {
    el('build-msg').textContent = '构建完成：' + payload.total + ' 张，跳过 ' + payload.skipped + ' 张 ' + JSON.stringify(payload.counts);
  }
  await fetchState();
  renderData();
}
el('crop').onclick = () => action('crop');
el('build').onclick = () => action('build');

/* ---------------- 训练 ---------------- */
async function startTrain() {
  const body = {
    epochs: Number(el('t-epochs').value),
    folds: Number(el('t-folds').value),
    backbone: el('t-backbone').value,
    device: el('t-device').value
  };
  el('train-state').textContent = '启动中...';
  const response = await fetch('/api/train', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)});
  const payload = await response.json();
  if (payload.error) { el('train-state').textContent = payload.error; return; }
  refreshTrain();
  setTimeout(refreshTrain, 1500);
}
async function refreshTrain() {
  const response = await fetch('/api/train/status');
  const payload = await response.json();
  el('train-log').textContent = payload.log || '（还没有开始训练）';
  el('train-log').scrollTop = el('train-log').scrollHeight;
  el('train-state').textContent = payload.running
    ? ('训练中… 已运行 ' + payload.elapsed + ' s')
    : (payload.returncode === null ? '空闲' : ('上次退出码 ' + payload.returncode + '，耗时 ' + payload.elapsed + ' s'));
  if (payload.running) setTimeout(refreshTrain, 1500);
}
el('train-start').onclick = startTrain;
el('train-refresh').onclick = refreshTrain;

/* ---------------- 评估 ---------------- */
function renderEval() {
  if (!lastEval) { el('eval-rounds').textContent = '点「运行评估」后显示'; return; }
  if (lastEval.error) { el('eval-rounds').textContent = lastEval.error; return; }
  const threshold = Number(el('thr').value);
  el('thr-value').textContent = threshold.toFixed(2);
  const rows = lastEval.rows.map(row => ({...row, fallback: row.confidence < threshold}));
  const accuracy = rows.length ? rows.filter(r => r.truth === r.pred).length/rows.length : 0;
  const fallback = rows.filter(r => r.fallback).length;
  const cv = lastEval.cv_accuracy, baseline = lastEval.baseline;
  el('eval-kpi').innerHTML =
    kpi('训练集内准确率', pct(accuracy)) + kpi('CV（泛化）', cv === null || cv === undefined ? '—' : cv) +
    kpi('基线', pct(baseline || 0)) + kpi('回退率', pct(rows.length ? fallback/rows.length : 0)) +
    kpi('样本', rows.length);
  el('eval-warn').innerHTML = (cv !== null && cv !== undefined && baseline !== null && baseline !== undefined && cv <= baseline)
    ? '<span class="badge no">模型不如基线</span> 交叉验证准确率低于"永远猜多数类"，任务里会自动禁用；继续攒样本再训。'
    : '<span class="badge ok">可用于试跑</span> 泛化准确率高于基线，可以到任务里开阈值试跑。';

  const grouped = {};
  rows.forEach(row => { (grouped[row.run] = grouped[row.run] || []).push(row); });
  let hits = 0, total = 0;
  const table = ['<table><tr><th>局</th><th>候选（真值 / 预测 / 置信度）</th><th>模型选择</th><th>结果</th></tr>'];
  Object.keys(grouped).sort().forEach(run => {
    const candidates = grouped[run].sort((a, b) => a.index - b.index);
    const usable = candidates.filter(c => !c.fallback);
    const picked = usable.length ? usable.reduce((a, b) => (b.score > a.score ? b : a)) : null;
    const bestTrue = candidates.some(c => c.truth === 'SR_plus') ? 'SR_plus' : 'below';
    if (picked) { total += 1; if (picked.truth === bestTrue) hits += 1; }
    const cells = candidates.map(c =>
      (c.truth === c.pred ? '✓' : '✗') + ' ' + c.truth + '→' + c.pred + ' p=' + c.confidence.toFixed(2)
      + (c.fallback ? '（回退）' : '')).join('<br>');
    const pickedText = picked ? (picked.id.split('_').pop() + '（' + picked.pred + '）') : '全部回退';
    table.push('<tr class="' + (picked ? (picked.truth === bestTrue ? 'hit' : 'miss') : '') + '"><td>' + run
      + '</td><td>' + cells + '</td><td>' + pickedText + '</td><td>'
      + (picked ? (picked.truth === bestTrue ? '选对' : '选错') : '未采信') + '</td></tr>');
  });
  table.push('</table>');
  el('eval-rounds').innerHTML = '<p class="small">三选一命中：' + hits + '/' + total
    + '（只统计至少一个候选过阈值的局）。注意这里用的是训练时的同一批图，结果偏乐观，'
    + '真实效果看左侧 CV。</p>' + table.join('');
  el('eval-rows').innerHTML = '<table><tr><th>候选</th><th>真值</th><th>预测</th><th>置信度</th><th>分</th></tr>'
    + rows.map(r => '<tr><td>' + r.id + '</td><td>' + r.truth + '</td><td>' + r.pred + '</td><td>'
      + r.confidence.toFixed(2) + '</td><td>' + r.score + '</td></tr>').join('') + '</table>';
}
function kpi(name, value) { return '<div><span class="small">' + name + '</span><b>' + value + '</b></div>'; }

async function runEval() {
  el('eval-rounds').textContent = '评估中...';
  const response = await fetch('/api/eval', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({threshold: Number(el('thr').value)})});
  lastEval = await response.json();
  renderEval();
}
async function refreshEval() {
  await fetchState();
  if (!state.model) { el('eval-rounds').textContent = '还没有模型：先去「训练」训一个。'; return; }
  if (state.model.threshold !== undefined) {
    el('thr').value = state.model.threshold;
    el('thr-value').textContent = Number(state.model.threshold).toFixed(2);
  }
  el('eval-kpi').innerHTML = kpi('CV 准确率', (state.model.cv_accuracy ?? '—'))
    + kpi('基线', (state.model.baseline ?? '—'))
    + kpi('样本', (state.model.samples ?? '—'))
    + kpi('阈值', (state.model.threshold ?? 0.6));
  el('eval-warn').innerHTML = (state.model.cv_accuracy !== undefined && state.model.cv_accuracy <= (state.model.baseline || 0))
    ? '<span class="badge no">模型不如基线</span> 先不要期待它在局里选对，继续攒样本。'
    : '';
}
el('thr').oninput = () => { el('thr-value').textContent = Number(el('thr').value).toFixed(2); renderEval(); };
el('eval-run').onclick = runEval;
el('thr-save').onclick = async () => {
  const response = await fetch('/api/threshold', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({threshold: Number(el('thr').value)})});
  const payload = await response.json();
  el('train-state').textContent = payload.error || ('阈值已保存：' + payload.threshold);
  await fetchState();
};

/* ---------------- 初始化 ---------------- */
async function fetchState() {
  const response = await fetch('/api/state');
  state = await response.json();
  renderLabel(); renderData();
}
(async function start() {
  await fetchState();
  buildButtons();
  const index = (state.items || []).findIndex(item => !item.label);
  cursor = index < 0 ? 0 : index;
  renderLabel();
})();
</script>
"""


class TrainingRunner:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.process = None
        self.started_at = None

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, payload: dict) -> dict:
        if self.running():
            return {'error': '已有训练在跑，等它结束'}
        if not TRAIN_PYTHON.is_file():
            return {'error': f'找不到训练环境：{TRAIN_PYTHON}（先创建 .venv-train 并装 torch）'}
        command = [str(TRAIN_PYTHON), str(ROOT/'dev_tools/boss_train.py'),
                   '--device', str(payload.get('device', 'cuda')),
                   '--epochs', str(int(payload.get('epochs', 600))),
                   '--folds', str(int(payload.get('folds', 5))),
                   '--backbone', str(payload.get('backbone', 'mobilenet'))]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(f"$ {' '.join(command)}\n", encoding='utf-8')
        with self.log_path.open('a', encoding='utf-8') as log:
            self.process = subprocess.Popen(command, cwd=str(ROOT), stdout=log,
                                            stderr=subprocess.STDOUT)
        self.started_at = time.time()
        return {'started': True, 'command': command}

    def status(self) -> dict:
        log = ''
        if self.log_path.is_file():
            log = self.log_path.read_text(encoding='utf-8', errors='replace')[-8000:]
        elapsed = 0 if self.started_at is None else round(time.time() - self.started_at, 1)
        return {'running': self.running(), 'elapsed': elapsed, 'log': log,
                'returncode': None if self.process is None else self.process.poll()}


def model_meta(model_path: Path) -> dict | None:
    meta_path = model_path.parent/'model.json'
    if not model_path.is_file() or not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    meta['path'] = str(model_path)
    meta['size_mb'] = round(model_path.stat().st_size/1024/1024, 2)
    return meta


def create_app(dataset_dir: Path, capture_dir: Path, model_path: Path) -> FastAPI:
    app = FastAPI(title='Boss rarity workbench', docs_url=None, redoc_url=None)
    runner = TrainingRunner(TRAIN_LOG)

    @app.get('/', response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get('/api/state')
    def state():
        items = dataset_tool.load_manifest(dataset_dir).get('items', [])
        labels = dataset_tool.load_labels(dataset_dir)
        for item in items:
            item['label'] = labels.get(item['id'])
        stats = dataset_tool.dataset_stats(dataset_dir)
        return {
            'items': items,
            'stats': stats,
            'label_list': dataset_tool.LABELS,
            'hints': dataset_tool.LABEL_HINT,
            'model': model_meta(model_path),
            'targets': {'per_class': TARGET_PER_CLASS, 'total': TARGET_TOTAL},
            'training': {'running': runner.running()},
            'dataset': str(dataset_dir),
        }

    @app.post('/api/label')
    def put_label(payload: dict = Body(...)):
        item_id, label = payload.get('id'), payload.get('label')
        if label not in dataset_tool.LABELS + [dataset_tool.SKIP]:
            raise HTTPException(status_code=400, detail=f'unknown label: {label}')
        known = {item['id'] for item in dataset_tool.load_manifest(dataset_dir).get('items', [])}
        if item_id not in known:
            raise HTTPException(status_code=404, detail=f'unknown item: {item_id}')
        labels = dataset_tool.load_labels(dataset_dir)
        labels[item_id] = label
        dataset_tool.save_labels(labels, dataset_dir)
        return {'ok': True, 'stats': dataset_tool.dataset_stats(dataset_dir)}

    @app.post('/api/crop')
    def crop():
        return dataset_tool.crop_dataset(capture_dir, dataset_dir, quiet=True)

    @app.post('/api/build')
    def build():
        return dataset_tool.build_dataset(dataset_dir)

    @app.post('/api/train')
    def train(payload: dict = Body(default={})):
        return runner.start(payload or {})

    @app.get('/api/train/status')
    def train_status():
        return runner.status()

    @app.post('/api/eval')
    def evaluate(payload: dict = Body(default={})):
        if not model_path.is_file():
            return {'error': '还没有模型，先去「训练」'}
        payload = payload or {}
        threshold = payload.get('threshold')
        from tasks.Hyakkiyakou.boss_rarity import BossRarityModel
        model = BossRarityModel(model_path, threshold=float(threshold) if threshold is not None
                                else float(model_meta(model_path).get('threshold', 0.6)))
        result = boss_eval.evaluate(model, dataset_dir, threshold=threshold)
        result['baseline'] = model_meta(model_path).get('baseline')
        result['cv_accuracy'] = model_meta(model_path).get('cv_accuracy')
        return result

    @app.post('/api/threshold')
    def threshold(payload: dict = Body(...)):
        meta_path = model_path.parent/'model.json'
        if not meta_path.is_file():
            return {'error': '还没有模型'}
        value = float(payload.get('threshold', 0.6))
        if not 0.05 <= value <= 0.99:
            return {'error': '阈值要在 0.05~0.99 之间'}
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        meta['threshold'] = round(value, 2)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
        return {'threshold': meta['threshold']}

    @app.get('/crops/{name}')
    def crops(name: str):
        root = (dataset_dir/'crops').resolve()
        target = (root/name).resolve()
        if root not in target.parents or not target.is_file():
            raise HTTPException(status_code=404, detail='not found')
        return FileResponse(target)

    return app


def pick_port(host: str, port: int, tries: int = 20) -> int:
    """端口被占用时自动往后找一个空闲的，避免"通常每个套接字地址只允许使用一次"。"""
    import socket
    for candidate in range(port, port + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, candidate))
                return candidate
            except OSError:
                continue
    raise SystemExit(f'{host}:{port} 起 20 个端口都被占用，用 --port 换一个范围')


def environment_report() -> None:
    """先打印 numpy 的实际来源。加载到 numpy 2.x 时，cv2/onnxruntime 这类按 1.x
    编译的扩展会直接崩，而且报错信息往往指向 cv2 而不是 numpy，很难查。"""
    try:
        import numpy
    except Exception as exception:
        print(f'!! numpy 导入失败：{exception}')
        return
    print(f'numpy {numpy.__version__}  ←  {numpy.__file__}')
    if numpy.__version__.split('.')[0] != '1':
        print('!! 当前进程加载的是 numpy 2.x，与 .venv 里的 cv2 / onnxruntime（按 numpy 1.x 编译）不兼容。')
        print('   最常见原因：PYTHONPATH 指到了 .venv-train（训练环境才有 numpy 2.x）。')
        print('   PowerShell 清掉后重开：Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue')
        print('   或强制忽略环境变量：.venv\Scripts\python.exe -E dev_tools\boss_workbench.py')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8017)
    parser.add_argument('--out', default=str(dataset_tool.DATASET_ROOT))
    parser.add_argument('--captures', default=str(dataset_tool.CAPTURE_ROOT))
    parser.add_argument('--model', default=str(dataset_tool.DATASET_ROOT/'model/boss_rarity.onnx'))
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    environment_report()
    dataset_dir = Path(args.out)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(dataset_dir, Path(args.captures), Path(args.model))

    port = pick_port(args.host, args.port)
    if port != args.port:
        print(f'端口 {args.port} 已被占用（可能是上次没关的工作台），改用 {port}')
    url = f'http://{args.host}:{port}'
    print(f'工作台：{url}')
    print(f'数据集：{dataset_dir.resolve()}')
    if not args.no_browser:
        webbrowser.open(url)

    import uvicorn
    uvicorn.run(app, host=args.host, port=port, log_level='warning')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

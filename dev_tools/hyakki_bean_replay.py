"""Replay a captured Hyakki bean round through detection and decision logic without clicking.

Usage:
    .venv/Scripts/python.exe dev_tools/hyakki_bean_replay.py data/bean_capture/<round>
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_html_report(output_path: Path, round_path: Path, rows: list[dict]) -> Path:
    frames = []
    for row in rows:
        image_path = round_path / 'raw' / f"{row['frame']}.jpg"
        image_url = os.path.relpath(image_path.resolve(), output_path.parent.resolve()).replace('\\', '/')
        frames.append({**row, 'image': image_url})
    payload = json.dumps(frames, ensure_ascii=False).replace('<', '\\u003c')
    document = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>百鬼夜行撒豆回放</title>
<style>
:root{color-scheme:dark;font:14px "Segoe UI","Microsoft YaHei UI",sans-serif;background:#151a1e;color:#e8eeee}
*{box-sizing:border-box}body{margin:0}header{height:54px;display:flex;align-items:center;gap:18px;padding:0 18px;border-bottom:1px solid #394348}header strong{font-size:16px;font-weight:600}#summary{color:#a6b2b4;margin-left:auto;font-variant-numeric:tabular-nums}
main{max-width:1440px;margin:auto;padding:16px;display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:16px}.stage{min-width:0}canvas{display:block;width:100%;aspect-ratio:16/9;background:#080a0b;object-fit:contain}.controls{display:grid;grid-template-columns:auto auto auto minmax(90px,1fr) auto;align-items:center;gap:10px;padding-top:12px}button{height:34px;min-width:38px;padding:0 10px;border:1px solid #465358;background:#222a2e;color:#edf3f3;border-radius:4px;cursor:pointer}button:hover{background:#303a3e}input[type=range]{width:100%;accent-color:#f0c44e}.info{border-left:1px solid #394348;padding-left:16px}.info h2{font-size:15px;font-weight:600;margin:2px 0 14px}.info dl{display:grid;grid-template-columns:105px 1fr;gap:11px 8px;margin:0}.info dt{color:#a6b2b4}.info dd{margin:0;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}.badge{color:#f1cb55}.legend{margin-top:20px;color:#a6b2b4;line-height:1.8}.legend b{color:#55d6c2}.legend i{color:#f1cb55;font-style:normal}@media(max-width:800px){main{grid-template-columns:1fr}.info{border:0;border-top:1px solid #394348;padding:14px 0}.controls{grid-template-columns:auto auto auto 1fr}#speed{grid-column:4}header{gap:10px}header strong{font-size:14px}}
</style>
<header><strong>百鬼夜行撒豆回放</strong><span id="summary"></span></header>
<main><section class="stage"><canvas id="screen" width="1280" height="720" aria-label="检测与撒豆决策叠加画面"></canvas><div class="controls"><button id="prev" aria-label="上一帧">上一帧</button><button id="play">播放</button><button id="next" aria-label="下一帧">下一帧</button><input id="seek" type="range" min="0" value="0"><select id="speed" aria-label="播放速度"><option value="1000">1 fps</option><option value="500">2 fps</option><option value="250" selected>4 fps</option><option value="100">10 fps</option></select></div></section>
<aside class="info"><h2 id="frame-title"></h2><dl><dt>目标式神</dt><dd id="target"></dd><dt>稀有度</dt><dd id="rarity"></dd><dt>概率 UP</dt><dd id="prob"></dd><dt>剩余豆子</dt><dd id="beans"></dd><dt>剩余式神</dt><dd id="shikigami"></dd><dt>本帧决策</dt><dd id="decision"></dd><dt>模式</dt><dd id="mode"></dd></dl><div class="legend"><b>青框</b> 检测结果<br><i>黄框 / 十字</i> 当前瞄准位置</div></aside></main>
<script id="frame-data" type="application/json">__DATA__</script>
<script>
const frames=JSON.parse(document.getElementById('frame-data').textContent),canvas=document.getElementById('screen'),ctx=canvas.getContext('2d'),img=new Image();let index=0,timer=null;
const rarity={};const $=id=>document.getElementById(id);$('seek').max=Math.max(0,frames.length-1);$('summary').textContent=`${frames.length} 帧`;
function render(){if(!frames.length)return;const f=frames[index];img.onload=()=>{ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0,canvas.width,canvas.height);for(const t of (f.tracks||[])){const [trackId,classId,conf,cx,cy,w,h]=t,x=cx-w/2,y=cy-h/2;ctx.strokeStyle='#55d6c2';ctx.lineWidth=2;ctx.strokeRect(x,y,w,h);ctx.fillStyle='#55d6c2';ctx.font='16px Segoe UI';ctx.fillText(`${classId} ${Number(conf).toFixed(2)}`,x,Math.max(18,y-4))}if(f.throw){ctx.strokeStyle='#f1cb55';ctx.lineWidth=3;ctx.strokeRect(Number(f.x)-18,Number(f.y)-18,36,36);ctx.beginPath();ctx.moveTo(Number(f.x)-10,Number(f.y));ctx.lineTo(Number(f.x)+10,Number(f.y));ctx.moveTo(Number(f.x),Number(f.y)-10);ctx.lineTo(Number(f.x),Number(f.y)+10);ctx.stroke()}};img.src=f.image;$('frame-title').textContent=`${index+1} / ${frames.length} · ${f.frame}`;$('target').textContent=f.target_name||`class ${f.target_class}`;$('rarity').textContent=f.rarity||'—';$('prob').textContent=f.prob_up?'有':'无';$('beans').textContent=f.bean_count;$('shikigami').textContent=f.shikigami_count;$('decision').textContent=f.throw?'撒豆 @ '+f.x+', '+f.y:'未撒豆';$('mode').textContent=f.mode;$('mode').className=f.mode==='rapid'?'badge':'';$('seek').value=index}
function step(n){index=Math.max(0,Math.min(frames.length-1,index+n));render()}$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('seek').oninput=e=>{index=Number(e.target.value);render()};$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放'}else{$('play').textContent='暂停';timer=setInterval(()=>{if(index>=frames.length-1){clearInterval(timer);timer=null;$('play').textContent='播放';return}step(1)},Number($('speed').value))}};$('speed').onchange=()=>{if(timer){clearInterval(timer);timer=setInterval(()=>step(index>=frames.length-1?-index:1),Number($('speed').value))}};document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1);if(e.code==='Space'){e.preventDefault();$('play').click()}});render();
</script></html>'''.replace('__DATA__', payload)
    page_path = output_path.with_suffix('.html')
    page_path.write_text(document, encoding='utf-8')
    return page_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round', type=Path, nargs='?', help='Captured round directory (defaults to latest)')
    parser.add_argument('--list', action='store_true', help='List captured rounds and exit')
    parser.add_argument('--config', default='oas1', help='OAS config profile')
    parser.add_argument('--recorded-tracks', action='store_true',
                        help='Use saved detections instead of running the configured detector')
    parser.add_argument('--delay-ms', type=int, default=0,
                        help='Simulated interval between frames, used for throw cooldown (default: 0)')
    parser.add_argument('--bean-count', type=int, default=250,
                        help='Fallback bean count when replaying captures without saved state')
    parser.add_argument('--shikigami-count', type=int, default=36,
                        help='Fallback remaining shikigami count when state was not saved')
    parser.add_argument('--prob-up', action='store_true',
                        help='Assume probability-up buff when replaying captures without saved state')
    parser.add_argument('--output', type=Path,
                        help='CSV output path (default: log/hyakkiyakou/replay/<round>.csv)')
    args = parser.parse_args()

    capture_root = ROOT / 'data' / 'bean_capture'
    rounds = sorted(
        (path for path in capture_root.iterdir() if path.is_dir() and (path / 'frames.jsonl').is_file()),
        key=lambda path: path.name,
        reverse=True,
    ) if capture_root.is_dir() else []
    if args.list:
        if not rounds:
            print(f'No captured rounds found under {capture_root}')
            return 1
        for index, path in enumerate(rounds, start=1):
            print(f'{index:>2}. {path.name}')
        return 0
    if args.round is None:
        if not rounds:
            parser.error(f'No captured rounds found under {capture_root}; enable hya_bean_capture and play one round first')
        args.round = rounds[0]

    log_path = args.round / 'frames.jsonl'
    if not log_path.is_file():
        parser.error(f'frames.jsonl not found: {log_path}')

    from module.config.config import Config
    from oashya.labels import label2id
    from oashya.tracker import Tracker
    from tasks.Hyakkiyakou.agent.agent import Agent
    from tasks.Hyakkiyakou.new_detector import DetectionRouter
    from tasks.Hyakkiyakou.shikigami_table import SHIKIGAMI_TABLE
    from tasks.Hyakkiyakou.slave.hya_slave import HyaBuff

    config = Config(args.config)
    settings = config.hyakkiyakou.hyakkiyakou_models
    strategy_config = config.hyakkiyakou.hyakkiyakou_config
    weights = [strategy_config.hya_sp, strategy_config.hya_ssr, strategy_config.hya_sr,
               strategy_config.hya_r, strategy_config.hya_n, strategy_config.hya_g,
               strategy_config.hya_ur]
    labels = {row['label']: class_id for class_id, row in SHIKIGAMI_TABLE.items()}
    priorities = []
    for label in strategy_config.hya_priorities.replace(' ', '').replace('，', ',').split(','):
        if label:
            priorities.append(labels[label] if label in labels else label2id(label))
    agent = Agent({'weights': weights, 'priorities': priorities,
                   'invite_friend': strategy_config.hya_invite_friend,
                   'auto_bean': strategy_config.hya_auto_bean})
    precision = 'fp32' if settings.model_precision.value == 'FP32' else 'int8'
    engine = 'onnxruntime' if settings.inference_engine.value == 'Onnxruntime' else 'tensorrt'
    tracker = Tracker(args={'conf_threshold': settings.conf_threshold,
                            'iou_threshold': settings.iou_threshold,
                            'precision': precision, 'inference_engine': engine, 'debug': False})
    detector = DetectionRouter(
        tracker, project_root=ROOT, mode=settings.detector_mode.value,
        manifest_path=settings.detector_manifest, timeout_ms=settings.detector_timeout_ms,
        observation_log=False,
    )
    rows = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        print(f'No frame records found in {log_path}')
        return 1
    state_saved = any(row.get('state') for row in rows)
    if not state_saved:
        print('This older round has no saved bean/buff state. Using fallback values; use --prob-up to test that case.')
    last_action = [0, 0, False, 10]
    previous_throw = None
    frames = 0
    throws = 0
    output_path = args.output or ROOT / 'log' / 'hyakkiyakou' / 'replay' / f'{args.round.name}.csv'
    output_rows = []

    for row in rows:
        name = row.get('frame', '')
        image_path = args.round / 'raw' / f'{name}.jpg'
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        saved_state = row.get('state')
        if saved_state:
            state = saved_state
        else:
            buff = HyaBuff.BUFF_STATE6 if args.prob_up else HyaBuff.BUFF_STATE0
            state = [args.bean_count, args.shikigami_count, 10,
                     HyaBuff.BUFF_STATE0, HyaBuff.BUFF_STATE0, HyaBuff.BUFF_STATE0, buff]
        freeze = row.get('phase') == 'frozen'
        if args.recorded_tracks:
            tracks = [tuple(track) for track in row.get('tracks', [])]
        else:
            tracks = detector(image=rgb_image, response=last_action)
        if previous_throw is not None:
            agent.last_throw_time = previous_throw
        action = agent.decision(tracks=tracks, state=state, freeze=freeze)
        last_action = action
        threw = bool(action[2] and state[0] > 0 and not freeze)
        if threw:
            throws += 1
            previous_throw = datetime.now()
        prob_up = any(int(value) == 3 for value in state[3:7])
        focus = agent.focus
        target = focus._class if focus is not None else -1
        timestamp = row.get('captured_at', '')
        from tasks.Hyakkiyakou.shikigami_table import SHIKIGAMI_TABLE
        rapid = bool(SHIKIGAMI_TABLE.get(int(target), {}).get('rarity') in {'SSR', 'SP'}
                     and prob_up and threw)
        rarity = SHIKIGAMI_TABLE.get(int(target), {}).get('rarity')
        output_rows.append({
            'frame': name, 'time': timestamp, 'bean_count': state[0],
            'shikigami_count': state[1], 'prob_up': prob_up, 'freeze': freeze,
            'target_class': target, 'throw': threw, 'x': action[0], 'y': action[1],
            'mode': 'rapid' if rapid else 'normal', 'rarity': rarity,
            'target_name': SHIKIGAMI_TABLE.get(int(target), {}).get('name'),
            'tracks': [list(track) for track in tracks],
        })
        frames += 1
        if args.delay_ms > 0:
            time.sleep(args.delay_ms / 1000)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8-sig') as output:
        writer = csv.DictWriter(output, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key != 'tracks'}
                          for row in output_rows])
    page_path = write_html_report(output_path, args.round, output_rows)
    print(f'Replayed {frames} frames; throw decisions: {throws}. No device clicks were sent.')
    print(f'Visual replay: {page_path}')
    print(f'CSV results: {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""对比 BWIKI 式神档册与已安装 oashya 的类别表，列出模型没有的式神。

用途：oashya 的类别表固定在 219 类（最新到 2024-02），之后上线的新式神模型无法
表示，只会被漏检或误判成旧式神。本脚本只做只读查询与对比，不修改任何模型或标签。

用法：
    .venv/Scripts/python.exe dev_tools/check_shikigami_gap.py
    .venv/Scripts/python.exe dev_tools/check_shikigami_gap.py --json data/shikigami/gap.json
    .venv/Scripts/python.exe dev_tools/check_shikigami_gap.py --icons data/shikigami/new_icons

数据来源：https://wiki.biligame.com/yys 的 Semantic MediaWiki ask 接口
（[[单元::式神]][[稀有度::SP]]|?式神名|?上线日期），与游戏内图鉴一致，
但属于第三方整理，上线日期偶尔有错（例如标成 2286 年），异常值会被标出。
"""

import argparse
import importlib.util
import json
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

WIKI_API = 'https://wiki.biligame.com/yys/api.php'
# BWIKI 前置有 WAF，只带 UA 会被判定为脚本并返回验证页；这里补齐浏览器常用头。
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/126.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': 'https://wiki.biligame.com/yys/',
}
# oashya 0.0.7 中最新式神的上线日期：SP 鲸汐千姬 / 福悦座敷童子 2024-02-07，SSR 泷 2024-01-10
DEFAULT_SINCE = '2024-02-07'
RARITIES = ['UR', 'SP', 'SSR', 'SR', 'R']
# oashya 标签里的历史拼写差异，避免把已有式神当成缺失
ALIASES = {
    '烬天玉澡前': '烬天玉藻前', '玉澡前': '玉藻前', '御馒津呱': '御馔津呱',
    '镰铀': '镰鼬', '档鸩': '鸩', '怡细工': '饴细工', '免丸': '兔丸', '巫师': '巫蛊师',
}


def fetch_json(params: dict, retry: int = 2) -> dict:
    url = WIKI_API + '?' + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retry + 1):
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                body = response.read().decode('utf-8')
            if body.startswith('{'):
                return json.loads(body)
            raise ValueError('BWIKI 返回了非 JSON 内容（可能是验证页），请稍后重试')
        except Exception:
            if attempt == retry:
                raise
            time.sleep(2 * (attempt + 1))


def query_units(pause: float = 1.0) -> list[dict]:
    units = []
    for index, rarity in enumerate(RARITIES):
        query = unicodedata.normalize(
            'NFC', f'[[单元::式神]][[稀有度::{rarity}]]|?式神名|?上线日期|limit=500')
        if index:
            time.sleep(pause)
        data = fetch_json({'action': 'ask', 'query': query, 'format': 'json'})
        for page, info in data['query']['results'].items():
            printouts = info['printouts']
            name = (printouts.get('式神名') or [page])[0]
            dates = printouts.get('上线日期') or []
            timestamp = dates[0]['timestamp'] if dates else None
            released = None
            if timestamp not in (None, ''):
                released = datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d')
            units.append({
                'rarity': rarity,
                'name': name,
                'released': released,
                'wiki': info['fullurl'],
            })
    return units


def installed_names() -> tuple[set[str], set[str]]:
    """返回 (类别表里的名称, 加上拼写别名后的比对集合)。别名只用于比对，不进入类别表。

    优先读本地扩展表 `tasks/Hyakkiyakou/shikigami_table.py`（含已追加的新式神）；
    没有该文件时退回 oashya 的 219 类。这样脚本在"扩表之后"仍然有意义：
    它反映的是**模型**还缺哪些，而不是本地表缺哪些。
    """
    from oashya.labels import CLASSIFY
    model_names = {item['name'] for item in CLASSIFY}
    table_path = Path('tasks/Hyakkiyakou/shikigami_table.py')
    if table_path.is_file():
        spec = importlib.util.spec_from_file_location('shikigami_table', table_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        table_names = {row[1] for row in module.ROWS}
    else:
        table_names = set(model_names)
    alias = set(ALIASES.values()) | set(ALIASES)
    return table_names, table_names | alias, model_names | alias


def _fetch_all(params: dict, key: str) -> list[dict]:
    """按 continue 协议把分页结果取全；BWIKI 单次请求有结果数上限。"""
    pages = []
    while True:
        data = fetch_json(params)
        pages.extend(data.get('query', {}).get(key, []))
        cont = data.get('continue')
        if not cont:
            return pages
        params = {**params, **cont}


def pick_icon_file(unit: str, files: list[str]) -> str | None:
    """从页面图片里挑一张最像头像的文件页。"""
    def stem(title: str) -> str:
        return title.split(':', 1)[-1].rsplit('.', 1)[0]

    for keyword in (unit, '头像', '觉醒前'):
        for title in files:
            if title.startswith('文件:') and keyword in stem(title):
                return title
    for title in files:  # 退而求其次：避开觉醒后与绘卷
        name = stem(title)
        if '觉醒后' not in name and '绘卷' not in name:
            return title
    return files[0] if files else None


def page_thumbnails(titles: list[str]) -> dict[str, str]:
    """批量取页面主图缩略图，用于人工建样本库；每批 8 个标题。

    BWIKI 没有开启 pageimages 扩展，只能先取页面里的文件页，再取缩略图地址。
    优先选择与式神同名、带“头像”或“觉醒前”的文件。
    """
    thumbs = {}
    for start in range(0, len(titles), 8):
        batch = titles[start:start + 8]
        pages = _fetch_all({
            'action': 'query', 'format': 'json', 'formatversion': '2',
            'prop': 'images', 'imlimit': '50', 'titles': '|'.join(batch),
        }, 'pages')
        wanted = []
        for page in pages:
            unit = page.get('title')
            files = [image['title'] for image in page.get('images', [])]
            pick = pick_icon_file(unit, files) if unit else None
            if unit and pick:
                wanted.append((unit, pick))
        if not wanted:
            continue
        info = _fetch_all({
            'action': 'query', 'format': 'json', 'formatversion': '2',
            'prop': 'imageinfo', 'iiprop': 'url', 'iiurlwidth': '240',
            'titles': '|'.join(file for _, file in wanted),
        }, 'pages')
        urls = {}
        for page in info:
            image_info = (page.get('imageinfo') or [{}])[0]
            url = image_info.get('thumburl') or image_info.get('url')
            if url:
                urls[page.get('title')] = url
        for unit, file in wanted:
            if file in urls:
                thumbs[unit] = urls[file]
        time.sleep(0.5)
    return thumbs


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=40) as response:
        target.write_bytes(response.read())


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sleep', type=float, default=1.0,
                        help='每次查询之间的间隔秒数，避免触发 BWIKI 的访问限制')
    parser.add_argument('--since', default=DEFAULT_SINCE,
                        help=f'视为“模型已覆盖”的上线日期上限，默认 {DEFAULT_SINCE}')
    parser.add_argument('--json', type=Path, help='把结果写成 JSON')
    parser.add_argument('--icons', type=Path, help='下载缺失式神页面主图到该目录，供人工建样本库')
    args = parser.parse_args()

    table_names, known, model_known = installed_names()
    units = query_units(pause=args.sleep)
    anomalies = [u for u in units if u['released'] and u['released'] > datetime.now().strftime('%Y-%m-%d')]
    # 模型缺 = 本地表里有、但 oashya 的 219 类里没有 → 需要训练新模型才能认出来
    missing = [u for u in units
               if u['released'] and u['released'] > args.since and u['name'] not in model_known]
    missing.sort(key=lambda u: (u['released'], u['rarity']))
    # 表也缺 = 连类别空间都还没加，先跑 gen_shikigami_table.py
    not_in_table = [u for u in units
                    if u['released'] and u['name'] not in known]
    # 日期在覆盖范围内、但名字对不上的，多半是 oashya 的拼写差异，单独提示
    unmatched = [u for u in units
                 if u['released'] and u['released'] <= args.since and u['name'] not in known
                 and u not in anomalies]

    counts = {}
    for unit in missing:
        counts[unit['rarity']] = counts.get(unit['rarity'], 0) + 1
    print(f'本地类别表 {len(table_names)} 个名称（含 {len(ALIASES)} 条拼写别名）；'
          f'其中 oashya 模型只有 {"219 类" if len(table_names) > 219 else "同表"}')
    print(f'模型缺失（上线日期晚于 {args.since}）：{len(missing)} 个  {counts}')
    for unit in missing:
        print(f"  {unit['released']}  {unit['rarity']:<3} {unit['name']:<10} {unit['wiki']}")
    if not_in_table:
        print(f'\n⚠ 本地类别表也没有：{len(not_in_table)} 个（需要先 gen_shikigami_table.py 追加）')
        for unit in not_in_table:
            print(f"  {unit['released'] or '????-??-??'}  {unit['rarity']:<3} {unit['name']:<10} {unit['wiki']}")
    if unmatched:
        print(f'\n需人工核对（日期较早但对不上名字，可能是拼写差异或联动限定）：{len(unmatched)} 个')
        for unit in unmatched:
            print(f"  {unit['released']}  {unit['rarity']:<3} {unit['name']:<10} {unit['wiki']}")
    if anomalies:
        print(f'\n日期异常，请手工确认：{len(anomalies)} 个')
        for unit in anomalies:
            print(f"  {unit['released']}  {unit['rarity']:<3} {unit['name']}")

    if args.icons and missing:
        args.icons.mkdir(parents=True, exist_ok=True)
        thumbs = page_thumbnails([u['name'] for u in missing])
        saved = 0
        for unit in missing:
            url = thumbs.get(unit['name'])
            if not url:
                print(f'  跳过（无缩略图）：{unit["name"]}')
                continue
            target = args.icons/f"{unit['rarity']}_{unit['name']}.png"
            download(url, target)
            unit['icon'] = str(target)
            saved += 1
        print(f'已下载 {saved} 张页面主图到 {args.icons.resolve()}')

    if args.json:
        payload = {
            'generated_at': datetime.now().astimezone().isoformat(),
            'source': 'https://wiki.biligame.com/yys (Semantic MediaWiki ask API)',
            'since': args.since,
            'counts': counts,
            'total_missing': len(missing),
            'units': missing,
            'not_in_table': not_in_table,
            'unmatched': unmatched,
            'anomalies': anomalies,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'已写入 {args.json.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

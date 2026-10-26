const state = {
  summary: null,
  events: [],
  scanning: false,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function toast(message) {
  const element = $('#toast');
  element.textContent = message;
  element.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.remove('show'), 2800);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || payload.error || detail;
    } catch (_) {
      // Keep the HTTP status when the body is not JSON.
    }
    throw new Error(detail);
  }
  return response.json();
}

function switchPage(name) {
  $$('.step').forEach((button) => button.classList.toggle('active', button.dataset.page === name));
  $$('.page').forEach((page) => page.classList.toggle('active', page.id === `page-${name}`));
}

function setNextAction(title, detail, label) {
  $('#next-title').textContent = title;
  $('#next-detail').textContent = detail;
  $('#next-action').textContent = label;
}

function renderSummary(summary) {
  state.summary = summary;
  $('#metric-events').textContent = summary.events ?? 0;
  $('#metric-high').textContent = summary.pending_high_rarity ?? 0;
  $('#metric-bean').textContent = summary.rounds?.bean ?? 0;
  $('#metric-boss').textContent = summary.rounds?.boss ?? 0;
  $('#path-bean').textContent = summary.bean_capture || '—';
  $('#path-boss').textContent = summary.boss_capture || '—';
  $('#path-state').textContent = summary.database || '—';
  $('#scan-state').textContent = summary.last_scan
    ? `上次扫描 ${summary.last_scan}`
    : '尚未扫描';

  if (!summary.events) {
    setNextAction(
      '先扫描现有采集数据',
      '把撒豆轨迹和鬼王候选整理成可审核的目标出现事件。',
      '开始扫描',
    );
  } else if (summary.pending_high_rarity > 0) {
    setNextAction(
      `有 ${summary.pending_high_rarity} 个高稀有度事件待审核`,
      '进入审核页确认旧模型是否认对，再决定是否送入 X-AnyLabeling。',
      '查看审核队列',
    );
  } else if ((summary.status?.pending ?? 0) > 0) {
    setNextAction(
      `有 ${summary.status.pending} 个事件待审核`,
      '高稀有度队列已处理完，可以继续处理可疑事件。',
      '继续审核',
    );
  } else {
    setNextAction(
      '当前审核队列已清空',
      '继续运行百鬼采集，或进入数据集页生成下一版本。',
      '查看数据集',
    );
  }
}

function renderEvents(payload) {
  state.events = payload.items || [];
  const container = $('#event-list');
  if (!state.events.length) {
    container.className = 'event-list empty';
    container.textContent = '暂无待审核的高稀有度或可疑事件。';
    return;
  }
  container.className = 'event-list';
  container.innerHTML = state.events.slice(0, 8).map((event) => {
    const rarity = event.rarity || 'other';
    const name = event.old_name || event.old_label || '身份未知';
    const source = event.source_kind === 'boss' ? '鬼王候选' : '撒豆轨迹';
    const confidence = Number.isFinite(event.confidence)
      ? ` · ${Math.round(event.confidence * 100)}%`
      : '';
    return `
      <div class="event">
        <img src="${escapeHtml(event.image_url)}" alt="${escapeHtml(name)}" loading="lazy">
        <div>
          <strong>${escapeHtml(name)}</strong>
          <small>${escapeHtml(source)} · ${escapeHtml(event.round_id)}${confidence}</small>
        </div>
        <span class="rarity ${escapeHtml(rarity)}">${escapeHtml(rarity)}</span>
      </div>`;
  }).join('');
}

async function refreshEvents() {
  try {
    const payload = await api('/api/events?pending_high_only=true&limit=40');
    renderEvents(payload);
  } catch (error) {
    toast(`读取审核队列失败：${error.message}`);
  }
}

async function refreshStatus() {
  try {
    const summary = await api('/api/status');
    renderSummary(summary);
    await refreshEvents();
  } catch (error) {
    toast(`工作台状态读取失败：${error.message}`);
  }
}

async function scan() {
  if (state.scanning) return;
  state.scanning = true;
  const buttons = [$('#scan-button'), $('#next-action')];
  buttons.forEach((button) => { button.disabled = true; });
  $('#scan-state').textContent = '正在扫描…';
  try {
    const result = await api('/api/scan', { method: 'POST' });
    toast(`扫描完成：${result.events_upserted} 个事件，${result.high_rarity_events} 个高稀有度`);
    await refreshStatus();
  } catch (error) {
    toast(`扫描失败：${error.message}`);
    await refreshStatus();
  } finally {
    state.scanning = false;
    buttons.forEach((button) => { button.disabled = false; });
  }
}

$$('.step').forEach((button) => {
  button.addEventListener('click', () => switchPage(button.dataset.page));
});
$('#scan-button').addEventListener('click', scan);
$('#refresh-events').addEventListener('click', refreshEvents);
$('#next-action').addEventListener('click', () => {
  const summary = state.summary;
  if (!summary?.events) {
    scan();
  } else if (summary.pending_high_rarity > 0 || (summary.status?.pending ?? 0) > 0) {
    switchPage('review');
  } else {
    switchPage('dataset');
  }
});

refreshStatus();

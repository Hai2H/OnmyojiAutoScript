const state = {
  summary: null,
  events: [],
  scanning: false,
  reviewFilter: 'high',
  reviewEvent: null,
  reviewImageIndex: 0,
  classes: null,
  labelPackage: null,
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
      <button class="event" data-event-id="${escapeHtml(event.id)}">
        <img src="${escapeHtml(event.image_url)}" alt="${escapeHtml(name)}" loading="lazy">
        <div>
          <strong>${escapeHtml(name)}</strong>
          <small>${escapeHtml(source)} · ${escapeHtml(event.round_id)}${confidence}</small>
        </div>
        <span class="rarity ${escapeHtml(rarity)}">${escapeHtml(rarity)}</span>
      </button>`;
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

function reviewQuery(filter) {
  if (filter === 'high') return '/api/events?pending_high_only=true&limit=200';
  if (filter === 'pending') return '/api/events?status=pending,needs_label&limit=200';
  if (filter === 'unknown') return '/api/events?status=unknown,needs_label&limit=200';
  return '/api/events?status=correct,changed,unknown&limit=200';
}

async function loadReviewQueue(selectFirst = false) {
  const list = $('#review-list');
  list.className = 'review-list empty';
  list.textContent = '正在读取…';
  try {
    const payload = await api(reviewQuery(state.reviewFilter));
    $('#review-total').textContent = payload.total ?? 0;
    if (!payload.items?.length) {
      list.textContent = '当前筛选下没有事件。';
      return;
    }
    list.className = 'review-list';
    list.innerHTML = payload.items.map((event) => {
      const name = event.old_name || event.old_label || '身份未知';
      const active = state.reviewEvent?.id === event.id ? ' active' : '';
      return `
        <button class="review-item${active}" data-review-id="${escapeHtml(event.id)}">
          <img src="${escapeHtml(event.image_url)}" alt="" loading="lazy">
          <div>
            <strong>${escapeHtml(name)}</strong>
            <span>${escapeHtml(event.rarity || '?')} · ${escapeHtml(event.status)} · ${escapeHtml(event.round_id)}</span>
          </div>
        </button>`;
    }).join('');
    if (selectFirst && (!state.reviewEvent || !payload.items.some((item) => item.id === state.reviewEvent.id))) {
      await selectReviewEvent(payload.items[0].id);
    }
  } catch (error) {
    list.textContent = `读取失败：${error.message}`;
  }
}

const suspicionLabels = {
  class_flip: '类别跳变',
  low_confidence: '低置信度',
  high_rarity: '高稀有度',
  unsupported_by_old_model: '旧模型类别表不支持',
  unknown_identity: '旧模型未识别身份',
};

function updateReviewBox() {
  const image = $('#review-image');
  const overlay = $('#review-box');
  const box = state.reviewEvent?.representatives?.[state.reviewImageIndex]?.box;
  if (!box || !image.naturalWidth || !image.naturalHeight) {
    overlay.hidden = true;
    return;
  }
  const imageRect = image.getBoundingClientRect();
  const shellRect = image.parentElement.getBoundingClientRect();
  const widthRatio = imageRect.width / image.naturalWidth;
  const heightRatio = imageRect.height / image.naturalHeight;
  overlay.style.left = `${imageRect.left - shellRect.left + box.x1 * widthRatio}px`;
  overlay.style.top = `${imageRect.top - shellRect.top + box.y1 * heightRatio}px`;
  overlay.style.width = `${(box.x2 - box.x1) * widthRatio}px`;
  overlay.style.height = `${(box.y2 - box.y1) * heightRatio}px`;
  overlay.hidden = false;
}

function renderReviewFrame(index) {
  if (!state.reviewEvent) return;
  const frames = state.reviewEvent.representatives || [];
  if (!frames.length) return;
  state.reviewImageIndex = Math.max(0, Math.min(index, frames.length - 1));
  $('#frame-tabs').innerHTML = frames.map((frame, frameIndex) => `
    <button data-frame-index="${frameIndex}" class="${frameIndex === state.reviewImageIndex ? 'active' : ''}">
      代表帧 ${frameIndex + 1}
    </button>`).join('');
  const image = $('#review-image');
  image.src = `/api/events/${encodeURIComponent(state.reviewEvent.id)}/image/${state.reviewImageIndex}`;
  $('#review-box').hidden = true;
}

async function ensureClasses() {
  if (state.classes) return;
  const payload = await api('/api/classes?limit=500');
  state.classes = payload.items || [];
  renderClassOptions();
}

function renderClassOptions() {
  const query = $('#class-search').value.trim().toLowerCase();
  const options = (state.classes || []).filter((item) => {
    if (!query) return true;
    return `${item.label} ${item.name} ${item.rarity}`.toLowerCase().includes(query);
  });
  $('#class-select').innerHTML = options.map((item) => `
    <option value="${item.class_id}">${escapeHtml(item.rarity)} · ${escapeHtml(item.name)} · ${escapeHtml(item.label)}</option>
  `).join('');
}

async function selectReviewEvent(eventId) {
  try {
    await ensureClasses();
    const event = await api(`/api/events/${encodeURIComponent(eventId)}`);
    state.reviewEvent = event;
    state.reviewImageIndex = 0;
    state.labelPackage = null;
    $$('.review-item').forEach((item) => item.classList.toggle('active', item.dataset.reviewId === eventId));
    $('#review-empty').hidden = true;
    $('#review-content').hidden = false;
    $('#old-identity').textContent = event.old_name || event.old_label || '身份未知';
    $('#old-rarity').textContent = event.rarity || 'UNKNOWN';
    $('#old-confidence').textContent = Number.isFinite(event.confidence)
      ? `${Math.round(event.confidence * 100)}%`
      : '分类模型';
    $('#frame-count').textContent = event.frame_count;
    $('#suspicion-list').innerHTML = (event.suspicion || [])
      .map((item) => `<span class="tag">${escapeHtml(suspicionLabels[item] || item)}</span>`).join('');
    $('#review-correct').disabled = event.old_class_id === null;
    $('#review-note').value = event.review_note || '';
    $('#label-folder').textContent = '尚未生成';
    $('#copy-label-folder').disabled = true;
    renderReviewFrame(0);
  } catch (error) {
    toast(`读取事件失败：${error.message}`);
  }
}

async function reviewAction(action, classId = null) {
  if (!state.reviewEvent) return;
  try {
    const result = await api(`/api/events/${encodeURIComponent(state.reviewEvent.id)}/review`, {
      method: 'POST',
      body: JSON.stringify({
        action,
        class_id: classId,
        note: $('#review-note').value,
      }),
    });
    toast(`已保存：${result.status}`);
    state.reviewEvent = result;
    await refreshStatus();
    await loadReviewQueue(false);
    if (state.reviewFilter === 'reviewed') {
      await selectReviewEvent(result.id);
    } else {
      const first = document.querySelector('.review-item');
      if (first) await selectReviewEvent(first.dataset.reviewId);
    }
  } catch (error) {
    toast(`保存失败：${error.message}`);
  }
}

async function prepareLabelPackage() {
  if (!state.reviewEvent) return;
  try {
    const result = await api(`/api/events/${encodeURIComponent(state.reviewEvent.id)}/prepare-label`, {
      method: 'POST',
    });
    state.labelPackage = result;
    $('#label-folder').textContent = result.folder;
    $('#copy-label-folder').disabled = false;
    toast('标注目录已生成。目录中已有的人工 JSON 不会被覆盖。');
  } catch (error) {
    toast(`生成失败：${error.message}`);
  }
}

async function loadXanyConfig() {
  try {
    const result = await api('/api/xany');
    $('#xany-executable').value = result.executable || '';
  } catch (_) {
    // Settings are optional until the first configuration attempt.
  }
}

$$('.step').forEach((button) => {
  button.addEventListener('click', () => {
    switchPage(button.dataset.page);
    if (button.dataset.page === 'review') loadReviewQueue(true);
  });
});
$('#scan-button').addEventListener('click', scan);
$('#refresh-events').addEventListener('click', refreshEvents);
$('#next-action').addEventListener('click', () => {
  const summary = state.summary;
  if (!summary?.events) {
    scan();
  } else if (summary.pending_high_rarity > 0 || (summary.status?.pending ?? 0) > 0) {
    switchPage('review');
    loadReviewQueue(true);
  } else {
    switchPage('dataset');
  }
});

$('#event-list').addEventListener('click', (event) => {
  const item = event.target.closest('[data-event-id]');
  if (!item) return;
  switchPage('review');
  loadReviewQueue(false).then(() => selectReviewEvent(item.dataset.eventId));
});
$('#review-filters').addEventListener('click', (event) => {
  const button = event.target.closest('[data-filter]');
  if (!button) return;
  state.reviewFilter = button.dataset.filter;
  $$('#review-filters button').forEach((item) => item.classList.toggle('active', item === button));
  loadReviewQueue(true);
});
$('#review-list').addEventListener('click', (event) => {
  const item = event.target.closest('[data-review-id]');
  if (item) selectReviewEvent(item.dataset.reviewId);
});
$('#frame-tabs').addEventListener('click', (event) => {
  const button = event.target.closest('[data-frame-index]');
  if (button) renderReviewFrame(Number(button.dataset.frameIndex));
});
$('#review-image').addEventListener('load', updateReviewBox);
window.addEventListener('resize', updateReviewBox);
$('#class-search').addEventListener('input', renderClassOptions);
$('#review-correct').addEventListener('click', () => reviewAction('correct'));
$('#review-unknown').addEventListener('click', () => reviewAction('unknown'));
$('#review-needs-label').addEventListener('click', () => reviewAction('needs_label'));
$('#review-skip').addEventListener('click', () => reviewAction('skip'));
$('#review-reopen').addEventListener('click', () => reviewAction('reopen'));
$('#review-change').addEventListener('click', () => {
  const value = $('#class-select').value;
  if (!value) {
    toast('请先选择一个具体式神类别。');
    return;
  }
  reviewAction('change', Number(value));
});
$('#prepare-label').addEventListener('click', prepareLabelPackage);
$('#copy-label-folder').addEventListener('click', async () => {
  if (!state.labelPackage?.folder) return;
  try {
    await navigator.clipboard.writeText(state.labelPackage.folder);
    toast('标注目录路径已复制。');
  } catch (_) {
    toast(`请手动复制：${state.labelPackage.folder}`);
  }
});
$('#save-xany').addEventListener('click', async () => {
  try {
    await api('/api/xany/config', {
      method: 'POST',
      body: JSON.stringify({ executable: $('#xany-executable').value }),
    });
    toast('X-AnyLabeling 路径已保存。');
  } catch (error) {
    toast(`保存失败：${error.message}`);
  }
});
$('#launch-xany').addEventListener('click', async () => {
  try {
    await api('/api/xany/launch', { method: 'POST' });
    toast('已启动 X-AnyLabeling。生成目录后，在程序中选择该目录。');
  } catch (error) {
    toast(`启动失败：${error.message}`);
  }
});

loadXanyConfig();
refreshStatus();

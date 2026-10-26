const state = {
  summary: null,
  events: [],
  scanning: false,
  reviewFilter: 'high',
  reviewEvent: null,
  reviewImageIndex: 0,
  reviewItems: [],
  reviewSelectedClassId: null,
  classes: null,
  labelPackage: null,
  datasetReadiness: null,
  datasets: [],
  trainingJobs: [],
  currentJob: null,
  trainingPoll: null,
  deployment: null,
  scanFilter: 'unreviewed',
  scanRound: '',
  scanOffset: 0,
  scanLimit: 48,
  scanCurrent: null,
  scanPackage: null,
  scanItems: [],
  scanTotal: 0,
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

function formatBytes(value) {
  const amount = Number(value || 0);
  if (amount >= 1024 ** 3) return `${(amount / 1024 ** 3).toFixed(2)} GB`;
  if (amount >= 1024 ** 2) return `${(amount / 1024 ** 2).toFixed(1)} MB`;
  if (amount >= 1024) return `${(amount / 1024).toFixed(1)} KB`;
  return `${amount} B`;
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
    if ($('#page-capture').classList.contains('active')) await loadCapturePage();
  } catch (error) {
    toast(`扫描失败：${error.message}`);
    await refreshStatus();
  } finally {
    state.scanning = false;
    buttons.forEach((button) => { button.disabled = false; });
  }
}

function renderCaptureItems(container, items, kind) {
  if (!items?.length) {
    container.className = 'capture-list empty';
    container.textContent = '没有采集数据';
    return;
  }
  container.className = 'capture-list';
  container.innerHTML = items.map((item) => `
    <div class="capture-row">
      <strong title="${escapeHtml(item.path)}">${escapeHtml(item.round)}</strong>
      <span>${kind === 'bean' ? `${item.frames_saved ?? '—'} 帧` : `${item.candidates ?? '—'} 候选`}</span>
      <span>${formatBytes(item.bytes)}</span>
    </div>`).join('');
}

async function loadCapturePage() {
  try {
    const payload = await api('/api/capture');
    $('#capture-bean').textContent = payload.bean.rounds;
    $('#capture-boss').textContent = payload.boss.rounds;
    $('#capture-size').textContent = formatBytes(payload.total_bytes);
    $('#capture-age').textContent = `${payload.retention.oldest_age_days} 天`;
    const notices = [];
    if (payload.retention.days_exceeded) notices.push('最早原始采集已超过 90 天。请人工决定是否清理。');
    if (payload.retention.bytes_exceeded) notices.push('原始采集已超过 30GB。请人工决定是否清理。');
    if (!notices.length) notices.push('当前未超过保留提示。工作台不会自动删除原始采集。');
    $('#capture-retention').innerHTML = notices
      .map((note) => `<div class="notice">${escapeHtml(note)}</div>`).join('');
    renderCaptureItems($('#capture-bean-list'), payload.bean.items, 'bean');
    renderCaptureItems($('#capture-boss-list'), payload.boss.items, 'boss');
  } catch (error) {
    toast(`读取采集统计失败：${error.message}`);
  }
}

const scanStatusLabels = {
  unreviewed: '未处理',
  candidate: '需要补框',
  no_action: '无需补框',
  not_new: '不是新式神',
  skipped: '已跳过',
  annotated: '已标注',
};

async function loadScanRounds() {
  const payload = await api('/api/raw-frames/rounds');
  const current = $('#scan-round').value;
  $('#scan-round').innerHTML = '<option value="">全部局</option>' + (payload.items || []).map((item) => `
    <option value="${escapeHtml(item.round_id)}">${escapeHtml(item.round_id)} · ${item.frames} 帧 · 空检测 ${item.empty_frames}</option>
  `).join('');
  if (current && [...$('#scan-round').options].some((option) => option.value === current)) {
    $('#scan-round').value = current;
  }
}

function renderScanGallery(payload) {
  const grid = $('#scan-grid');
  state.scanItems = payload.items || [];
  state.scanTotal = payload.total || 0;
  $('#scan-total').textContent = `${payload.total || 0} 帧`;
  const page = Math.floor((payload.offset || 0) / state.scanLimit) + 1;
  const pages = Math.max(1, Math.ceil((payload.total || 0) / state.scanLimit));
  $('#scan-page').textContent = `第 ${page} / ${pages} 页`;
  $('#scan-prev').disabled = state.scanOffset <= 0;
  $('#scan-next').disabled = state.scanOffset + state.scanLimit >= (payload.total || 0);
  if (!payload.items?.length) {
    grid.className = 'scan-grid empty';
    grid.textContent = '当前筛选下没有原始帧';
    return;
  }
  grid.className = 'scan-grid';
  grid.innerHTML = payload.items.map((frame) => {
    const active = state.scanCurrent?.id === frame.id ? ' active' : '';
    const status = frame.review_status || 'unreviewed';
    const suspicionCount = (frame.suspicion || []).length;
    return `
      <button class="scan-card${active}" data-frame-id="${escapeHtml(frame.id)}">
        <img src="${escapeHtml(frame.image_url)}" loading="lazy" alt="">
        <span class="scan-card-status ${escapeHtml(status)}">${escapeHtml(scanStatusLabels[status] || status)}</span>
        ${suspicionCount ? `<span class="scan-card-suspect">${suspicionCount} 个疑点</span>` : ''}
      </button>`;
  }).join('');
}

async function loadScanFrames() {
  const params = new URLSearchParams({
    filter: state.scanFilter,
    limit: String(state.scanLimit),
    offset: String(state.scanOffset),
  });
  if (state.scanRound) params.set('round_id', state.scanRound);
  try {
    const payload = await api(`/api/raw-frames?${params}`);
    if (!payload.items?.length && payload.total > 0 && state.scanOffset > 0) {
      state.scanOffset = Math.max(0, Math.floor((payload.total - 1) / state.scanLimit) * state.scanLimit);
      return loadScanFrames();
    }
    renderScanGallery(payload);
    return payload;
  } catch (error) {
    toast(`读取原始帧失败：${error.message}`);
    return null;
  }
}

function clearScanSelection() {
  state.scanCurrent = null;
  state.scanPackage = null;
  $('#scan-empty').hidden = false;
  $('#scan-content').hidden = true;
  $('#scan-image').removeAttribute('src');
  $('#scan-boxes').innerHTML = '';
}

function renderScanBoxes() {
  if (!state.scanCurrent) return;
  const image = $('#scan-image');
  const container = $('#scan-boxes');
  const shell = image.parentElement;
  if (!image.naturalWidth || !image.naturalHeight) {
    container.innerHTML = '';
    return;
  }
  const imageRect = image.getBoundingClientRect();
  const shellRect = shell.getBoundingClientRect();
  const widthRatio = imageRect.width / image.naturalWidth;
  const heightRatio = imageRect.height / image.naturalHeight;
  const shapes = [
    ...(state.scanCurrent.tracks || []).map((item) => ({
      box: item.box,
      label: item.name || item.label || '旧模型',
      manual: false,
    })),
    ...(state.scanCurrent.manual_shapes || []).map((item) => ({
      box: item.box,
      label: item.label,
      manual: true,
    })),
  ];
  container.innerHTML = shapes.map((item) => {
    const box = item.box;
    if (!box) return '';
    const left = imageRect.left - shellRect.left + box.x1 * widthRatio;
    const top = imageRect.top - shellRect.top + box.y1 * heightRatio;
    const width = (box.x2 - box.x1) * widthRatio;
    const height = (box.y2 - box.y1) * heightRatio;
    return `<span class="scan-box ${item.manual ? 'manual' : ''}" style="left:${left}px;top:${top}px;width:${width}px;height:${height}px"><span>${escapeHtml(item.label)}</span></span>`;
  }).join('');
}

function renderScanDetail(frame) {
  state.scanCurrent = frame;
  state.scanPackage = null;
  $('#scan-empty').hidden = true;
  $('#scan-content').hidden = false;
  $('#scan-round-label').textContent = frame.round_id;
  $('#scan-frame-label').textContent = frame.frame_key;
  $('#scan-track-count').textContent = frame.track_count;
  $('#scan-status-label').textContent = scanStatusLabels[frame.review_status] || frame.review_status;
  $('#scan-note').value = frame.note || '';
  $('#scan-tags').innerHTML = (frame.suspicion || [])
    .map((item) => `<span class="tag">${escapeHtml(suspicionLabels[item] || item)}</span>`).join('');
  $('#scan-package').textContent = '尚未生成标注目录';
  $('#scan-open-folder').disabled = true;
  const shapes = frame.manual_shapes || [];
  $('#scan-manual-shapes').innerHTML = shapes.length
    ? shapes.map((shape) => `<span class="manual-shape">${escapeHtml(shape.label)}</span>`).join('')
    : '尚无人工框';
  $('#scan-image').src = frame.image_url;
}

async function selectScanFrame(frameId) {
  try {
    const frame = await api(`/api/raw-frames/${encodeURIComponent(frameId)}`);
    renderScanDetail(frame);
    $$('.scan-card').forEach((card) => card.classList.toggle('active', card.dataset.frameId === frameId));
  } catch (error) {
    toast(`读取原始帧失败：${error.message}`);
  }
}

async function markScanFrame(action, { advance = false, prepare = false } = {}) {
  if (!state.scanCurrent) return;
  const currentIndex = state.scanItems.findIndex((item) => item.id === state.scanCurrent.id);
  const nextId = currentIndex >= 0 ? state.scanItems[currentIndex + 1]?.id : null;
  try {
    const frame = await api(`/api/raw-frames/${encodeURIComponent(state.scanCurrent.id)}/status`, {
      method: 'POST',
      body: JSON.stringify({ action, note: $('#scan-note').value }),
    });
    if (prepare && action === 'candidate') {
      const packageInfo = await api(`/api/raw-frames/${encodeURIComponent(frame.id)}/prepare-label`, {
        method: 'POST',
      });
      state.scanPackage = packageInfo;
    }
    await loadScanFrames();
    toast(`已标记为：${scanStatusLabels[frame.review_status]}`);
    if (advance) {
      const target = state.scanItems.find((item) => item.id === nextId) || state.scanItems[0];
      if (target) {
        await selectScanFrame(target.id);
      } else {
        clearScanSelection();
      }
    } else {
      renderScanDetail(frame);
    }
  } catch (error) {
    toast(`保存盲扫状态失败：${error.message}`);
  }
}

async function moveScan(direction) {
  if (!state.scanItems.length) return;
  const currentIndex = state.scanCurrent
    ? state.scanItems.findIndex((item) => item.id === state.scanCurrent.id)
    : -1;
  const targetIndex = currentIndex + direction;
  if (targetIndex >= 0 && targetIndex < state.scanItems.length) {
    await selectScanFrame(state.scanItems[targetIndex].id);
    return;
  }
  if (direction > 0 && state.scanOffset + state.scanLimit < state.scanTotal) {
    state.scanOffset += state.scanLimit;
    const payload = await loadScanFrames();
    if (payload?.items?.length) await selectScanFrame(payload.items[0].id);
    return;
  }
  if (direction < 0 && state.scanOffset > 0) {
    state.scanOffset = Math.max(0, state.scanOffset - state.scanLimit);
    const payload = await loadScanFrames();
    if (payload?.items?.length) await selectScanFrame(payload.items.at(-1).id);
    return;
  }
  toast(direction > 0 ? '已经是最后一帧。' : '已经是第一帧。');
}

async function loadScanPage() {
  try {
    await loadScanRounds();
    await loadScanFrames();
  } catch (error) {
    toast(`读取盲扫页失败：${error.message}`);
  }
}

function reviewQuery(filter) {
  if (filter === 'high') return '/api/events?rarity=UR,SP,SSR&limit=200';
  if (filter === 'pending') return '/api/events?status=pending,needs_label&limit=200';
  if (filter === 'unknown') return '/api/events?status=unknown,needs_label&limit=200';
  return '/api/events?status=correct,changed,unknown&limit=200';
}

function clearReviewSelection(message = '当前筛选下没有事件。') {
  state.reviewEvent = null;
  state.reviewImageIndex = 0;
  state.labelPackage = null;
  $('#review-empty').hidden = false;
  $('#review-empty').querySelector('strong').textContent = message;
  $('#review-empty').querySelector('span').textContent = '可以切换筛选条件，或先扫描新的采集数据。';
  $('#review-content').hidden = true;
  $('#review-image').removeAttribute('src');
  $('#review-box').hidden = true;
}

async function loadReviewQueue(selectFirst = false) {
  const list = $('#review-list');
  list.className = 'review-list empty';
  list.textContent = '正在读取…';
  try {
    const payload = await api(reviewQuery(state.reviewFilter));
    state.reviewItems = payload.items || [];
    $('#review-total').textContent = payload.total ?? 0;
    if (!payload.items?.length) {
      list.textContent = '当前筛选下没有事件。';
      clearReviewSelection();
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
  empty_detection: '旧模型没有检测到目标',
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
  $('#class-select').innerHTML = '<option value="">选择具体式神（可先搜索）</option>' + options.map((item) => `
    <option value="${item.class_id}">${escapeHtml(item.rarity)} · ${escapeHtml(item.name)} · ${escapeHtml(item.label)}</option>
  `).join('');
  $('#class-select').value = '';
  state.reviewSelectedClassId = null;
}

async function selectReviewEvent(eventId) {
  try {
    await ensureClasses();
    const event = await api(`/api/events/${encodeURIComponent(eventId)}`);
    state.reviewEvent = event;
    state.reviewImageIndex = 0;
    state.reviewSelectedClassId = null;
    state.labelPackage = null;
    $('#class-search').value = '';
    renderClassOptions();
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

async function reviewAction(action, classId = null, { advance = true, prepare = false } = {}) {
  if (!state.reviewEvent) return;
  const currentIndex = state.reviewItems.findIndex((item) => item.id === state.reviewEvent.id);
  const nextId = currentIndex >= 0 ? state.reviewItems[currentIndex + 1]?.id : null;
  try {
    const result = await api(`/api/events/${encodeURIComponent(state.reviewEvent.id)}/review`, {
      method: 'POST',
      body: JSON.stringify({
        action,
        class_id: classId,
        note: $('#review-note').value,
      }),
    });
    if (prepare && action === 'needs_label') {
      state.labelPackage = await api(
        `/api/events/${encodeURIComponent(result.id)}/prepare-label`,
        { method: 'POST' },
      );
    }
    toast(`已保存：${result.status}`);
    await refreshStatus();
    await loadReviewQueue(false);
    if (!advance) {
      await selectReviewEvent(result.id);
      return;
    }
    const target = state.reviewItems.find((item) => item.id === nextId) || state.reviewItems[0];
    if (target) {
      await selectReviewEvent(target.id);
    } else {
      clearReviewSelection('当前筛选下没有更多事件。');
    }
  } catch (error) {
    toast(`保存失败：${error.message}`);
  }
}

async function moveReview(direction) {
  if (!state.reviewItems.length) return;
  const currentIndex = state.reviewEvent
    ? state.reviewItems.findIndex((item) => item.id === state.reviewEvent.id)
    : -1;
  const targetIndex = currentIndex + direction;
  if (targetIndex < 0 || targetIndex >= state.reviewItems.length) {
    toast(direction > 0 ? '已经是队列最后一事件。' : '已经是队列第一事件。');
    return;
  }
  await selectReviewEvent(state.reviewItems[targetIndex].id);
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
    $('#scan-xany-executable').value = result.executable || '';
  } catch (_) {
    // Settings are optional until the first configuration attempt.
  }
}

function selectedDatasetClasses() {
  return $$('#dataset-classes input[type="checkbox"]:checked')
    .map((input) => Number(input.value));
}

function updateDatasetSelection() {
  const selected = selectedDatasetClasses();
  $('#dataset-selection').textContent = selected.length
    ? `已选择 ${selected.length} 个类别`
    : '未选择类别';
  $('#build-dataset').disabled = !state.datasetReadiness?.can_build || !selected.length;
  $$('#dataset-classes tr').forEach((row) => {
    row.classList.toggle('selected', row.querySelector('input')?.checked === true);
  });
}

function renderDatasetReadiness(readiness) {
  state.datasetReadiness = readiness;
  $('#dataset-reviewed').textContent = (readiness.reviewed_events ?? 0) + (readiness.manual_frames ?? 0);
  $('#dataset-groups').textContent = readiness.round_groups ?? 0;
  $('#dataset-eligible').textContent = readiness.default_classes?.length ?? 0;
  $('#dataset-warnings').innerHTML = (readiness.warnings || [])
    .map((warning) => `<div class="notice">${escapeHtml(warning)}</div>`).join('');
  const body = $('#dataset-classes');
  if (!readiness.classes?.length) {
    body.innerHTML = '<tr><td colspan="6" class="muted">还没有已审核并且带可用矩形框的类别。</td></tr>';
  } else {
    body.innerHTML = readiness.classes.map((row) => {
      const checked = readiness.default_classes.includes(row.class_id) ? ' checked' : '';
      const disabled = row.box_ready_events < readiness.gates.train ? ' disabled' : '';
      const className = row.gate === 'insufficient' ? 'insufficient' : row.gate;
      return `
        <tr>
          <td><input type="checkbox" value="${row.class_id}"${checked}${disabled} aria-label="选择 ${escapeHtml(row.name || row.label)}"></td>
          <td><strong>${escapeHtml(row.name || row.label)}</strong><br><span class="muted">${escapeHtml(row.label)}</span></td>
          <td>${escapeHtml(row.rarity)}</td>
          <td>${row.events}</td>
          <td>${row.box_ready_events}</td>
          <td><span class="gate ${className}">${className}</span></td>
        </tr>`;
    }).join('');
  }
  updateDatasetSelection();
}

function renderDatasetHistory(items) {
  state.datasets = items || [];
  $('#dataset-versions').textContent = state.datasets.length;
  const container = $('#dataset-history');
  if (!state.datasets.length) {
    container.className = 'dataset-history empty';
    container.textContent = '尚未构建数据集';
    return;
  }
  container.className = 'dataset-history';
  container.innerHTML = state.datasets.map((item) => {
    const stats = item.stats || {};
    const classes = (item.classes || []).map((row) => row.label).join(', ');
    return `
      <div class="version-card">
        <strong>${escapeHtml(item.version)}</strong>
        <span>${stats.images || 0} 图 · ${stats.boxes || 0} 框 · ${escapeHtml(JSON.stringify(stats.per_split || {}))}</span><br>
        <span>${escapeHtml(classes)}</span>
        <code>${escapeHtml(item.path)}</code>
      </div>`;
  }).join('');
}

async function loadDatasetPage() {
  try {
    const [readiness, history] = await Promise.all([
      api('/api/dataset/readiness'),
      api('/api/datasets'),
    ]);
    renderDatasetReadiness(readiness);
    renderDatasetHistory(history.items);
  } catch (error) {
    toast(`读取数据集状态失败：${error.message}`);
  }
}

function renderGpu(gpu) {
  const element = $('#gpu-status');
  if (gpu?.available) {
    element.textContent = `${gpu.name} · ${gpu.memory_used_mb}/${gpu.memory_total_mb} MB`;
  } else {
    element.textContent = 'GPU 不可用';
  }
}

function renderTrainingJobs(payload) {
  state.trainingJobs = payload.items || [];
  renderGpu(payload.gpu);
  const container = $('#training-jobs');
  if (!state.trainingJobs.length) {
    container.className = 'job-list empty';
    container.textContent = '还没有训练任务';
  } else {
    container.className = 'job-list';
    container.innerHTML = state.trainingJobs.map((job) => `
      <button class="job-item${state.currentJob?.id === job.id ? ' active' : ''}" data-job-id="${escapeHtml(job.id)}">
        <strong><span>${escapeHtml(job.kind)} · ${escapeHtml(job.id)}</span><em class="job-state ${escapeHtml(job.status)}">${escapeHtml(job.status)}</em></strong>
        <span>${escapeHtml(job.dataset_version)} · ${escapeHtml(job.model)} · ${escapeHtml(job.created_at)}</span>
      </button>`).join('');
  }
  const active = state.trainingJobs.some((job) => ['queued', 'running'].includes(job.status));
  $('#start-training').disabled = Boolean(payload.active);
  if (active) {
    clearTimeout(state.trainingPoll);
    state.trainingPoll = setTimeout(refreshTraining, 2500);
  }
}

function classRowsForJob(job) {
  const dataset = state.datasets.find((item) => item.version === job.dataset_version);
  return new Map((dataset?.classes || []).map((item) => [item.class_index, item]));
}

function renderJobMetrics(job) {
  const result = job.result;
  if (!result) {
    $('#job-metrics').innerHTML = job.error
      ? `<div class="notice">${escapeHtml(job.error)}</div>`
      : '<span class="muted">训练完成后显示每类 Precision / Recall 和校准阈值。</span>';
    return;
  }
  const metricSet = result.test || result.val;
  if (!metricSet) {
    $('#job-metrics').textContent = '任务已完成，但没有指标数据。';
    return;
  }
  const classes = classRowsForJob(job);
  const thresholds = result.thresholds || {};
  const rows = (metricSet.classes || []).map((row) => {
    const meta = classes.get(row.class_index) || {};
    return `<tr>
      <td>${escapeHtml(meta.name || row.class_index)}</td>
      <td>${Math.round(row.precision * 100)}%</td>
      <td>${Math.round(row.recall * 100)}%</td>
      <td>${thresholds[String(row.class_index)] ?? '—'}</td>
      <td>${row.tp}/${row.fp}/${row.fn}</td>
    </tr>`;
  }).join('');
  $('#job-metrics').innerHTML = `
    <div class="metrics">
      <article><span>${escapeHtml(metricSet.split)} 图</span><strong>${metricSet.images}</strong></article>
      <article><span>Precision</span><strong>${Math.round(metricSet.precision * 100)}%</strong></article>
      <article><span>Recall</span><strong>${Math.round(metricSet.recall * 100)}%</strong></article>
      <article><span>P95 延迟</span><strong>${metricSet.latency_ms_p95 ?? '—'} ms</strong></article>
      <article><span>best.pt</span><strong>${result.best_weight ? '已生成' : '—'}</strong></article>
    </div>
    <table class="metrics-table"><thead><tr><th>类别</th><th>P</th><th>R</th><th>阈值</th><th>TP/FP/FN</th></tr></thead><tbody>${rows}</tbody></table>`;
}

async function selectTrainingJob(jobId, includeLog = true) {
  try {
    const suffix = includeLog ? '?log_tail=60000' : '';
    const job = await api(`/api/training/${encodeURIComponent(jobId)}${suffix}`);
    state.currentJob = job;
    $$('.job-item').forEach((item) => item.classList.toggle('active', item.dataset.jobId === jobId));
    $('#job-title').textContent = `${job.kind} · ${job.status}`;
    $('#job-log').textContent = job.log || '（暂无日志）';
    $('#job-log').scrollTop = $('#job-log').scrollHeight;
    renderJobMetrics(job);
    $('#run-test').disabled = !(job.kind === 'train' && job.status === 'completed' && job.result?.best_weight);
    $('#cancel-job').disabled = !['queued', 'running'].includes(job.status);
  } catch (error) {
    toast(`读取任务失败：${error.message}`);
  }
}

function datasetOptions(items) {
  const current = $('#train-dataset').value;
  $('#train-dataset').innerHTML = (items || []).map((item) => `
    <option value="${escapeHtml(item.version)}">${escapeHtml(item.version)} · ${item.stats?.images || 0} 图 · ${item.stats?.boxes || 0} 框</option>
  `).join('');
  if (current && [...$('#train-dataset').options].some((item) => item.value === current)) {
    $('#train-dataset').value = current;
  }
}

async function refreshTraining(includeDetails = true) {
  try {
    const [payload, history] = await Promise.all([
      api('/api/training?limit=30'),
      api('/api/datasets'),
    ]);
    state.datasets = history.items || [];
    datasetOptions(state.datasets);
    renderTrainingJobs(payload);
    if (state.currentJob) {
      await selectTrainingJob(state.currentJob.id, includeDetails);
    }
  } catch (error) {
    toast(`读取训练状态失败：${error.message}`);
  }
}

async function loadTrainingPage() {
  await refreshTraining(true);
  if (!state.currentJob && state.trainingJobs.length) {
    await selectTrainingJob(state.trainingJobs[0].id);
  }
}

function renderService(service, active) {
  $('#service-state').textContent = service?.running ? '运行中' : '未运行';
  $('#service-detail').textContent = service?.running
    ? `${service.model_version || active?.model_version || '?'} · ${service.gpu?.name || 'GPU'} · PID ${service.pid}`
    : active?.model_version
      ? `已激活 ${active.model_version}，模式 ${active.mode}`
      : '尚未激活模型。';
  $('#start-inference').disabled = !active?.model_version || Boolean(service?.running);
  $('#stop-inference').disabled = !service?.running;
}

function renderModels(models, active) {
  const container = $('#model-versions');
  if (!models?.length) {
    container.className = 'model-list empty';
    container.textContent = '还没有通过验收并发布的模型';
    return;
  }
  container.className = 'model-list';
  container.innerHTML = models.map((model) => {
    const test = model.metrics?.test || {};
    return `
      <div class="model-card${model.active ? ' active' : ''}">
        <strong>${escapeHtml(model.version)}</strong>
        <p>数据集 ${escapeHtml(model.dataset_version)} · ${(model.classes || []).length} 类</p>
        <p>test P ${Math.round((test.precision || 0) * 100)}% · R ${Math.round((test.recall || 0) * 100)}% · P95 ${test.latency_ms_p95 ?? '—'}ms</p>
        <div class="inline-actions">
          <button class="text-button" data-activate-model="${escapeHtml(model.version)}">
            ${model.active ? `已激活 · ${escapeHtml(model.mode)}` : '按当前模式激活'}
          </button>
        </div>
      </div>`;
  }).join('');
}

function renderCandidates(candidates) {
  const container = $('#model-candidates');
  if (!candidates.length) {
    container.className = 'model-list empty';
    container.textContent = '暂无训练候选';
    return;
  }
  container.className = 'model-list';
  container.innerHTML = candidates.map(({ job, candidate }) => {
    if (!candidate) {
      return `<div class="model-card"><strong>${escapeHtml(job.id)}</strong><p>无法读取候选状态</p></div>`;
    }
    const failures = (candidate.failures || []).map((item) => `<li>${escapeHtml(item)}</li>`).join('');
    return `
      <div class="model-card">
        <strong>${escapeHtml(job.id)}</strong>
        <p>${escapeHtml(job.dataset_version)} · ${escapeHtml(job.model)}</p>
        ${candidate.ready
          ? '<p>已通过冻结 test，可发布。</p><div class="inline-actions"><button class="text-button" data-publish-job="${escapeHtml(job.id)}">发布并激活</button></div>'
          : `<p>${escapeHtml(candidate.reason || '尚未达到发布条件')}</p>${failures ? `<ul class="candidate-failures">${failures}</ul>` : ''}`}
      </div>`;
  }).join('');
}

async function loadDeploymentPage() {
  try {
    const [deployment, training] = await Promise.all([
      api('/api/deployment'),
      api('/api/training?limit=100'),
    ]);
    state.deployment = deployment;
    state.trainingJobs = training.items || [];
    const active = deployment.active || {};
    if (active.mode) $('#deploy-mode').value = active.mode;
    renderService(deployment.service, active);
    renderModels(deployment.models, active);
    const trainJobs = state.trainingJobs.filter((job) => job.kind === 'train' && job.status === 'completed');
    const candidates = await Promise.all(trainJobs.map(async (job) => {
      try {
        return { job, candidate: await api(`/api/models/${encodeURIComponent(job.id)}/candidate`) };
      } catch (_) {
        return { job, candidate: null };
      }
    }));
    renderCandidates(candidates);
  } catch (error) {
    toast(`读取部署状态失败：${error.message}`);
  }
}

$$('.step').forEach((button) => {
  button.addEventListener('click', () => {
    switchPage(button.dataset.page);
    if (button.dataset.page === 'review') loadReviewQueue(true);
    if (button.dataset.page === 'capture') loadCapturePage();
    if (button.dataset.page === 'scan') loadScanPage();
    if (button.dataset.page === 'dataset') loadDatasetPage();
    if (button.dataset.page === 'training') loadTrainingPage();
    if (button.dataset.page === 'deploy') loadDeploymentPage();
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
    loadDatasetPage();
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
$('#class-select').addEventListener('change', (event) => {
  state.reviewSelectedClassId = event.target.value ? Number(event.target.value) : null;
});
$('#review-correct').addEventListener('click', () => reviewAction('correct'));
$('#review-unknown').addEventListener('click', () => reviewAction('unknown'));
$('#review-needs-label').addEventListener('click', () => reviewAction('needs_label', null, { prepare: true }));
$('#review-skip').addEventListener('click', () => reviewAction('skip'));
$('#review-reopen').addEventListener('click', () => reviewAction('reopen', null, { advance: false }));
$('#review-change').addEventListener('click', () => {
  if (!state.reviewSelectedClassId) {
    toast('请先选择一个具体式神类别。');
    return;
  }
  reviewAction('change', state.reviewSelectedClassId);
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
$('#import-label').addEventListener('click', async () => {
  if (!state.reviewEvent) return;
  try {
    const result = await api(`/api/events/${encodeURIComponent(state.reviewEvent.id)}/import-label`, {
      method: 'POST',
    });
    state.reviewEvent = result.event;
    toast(`已导入 ${result.imported} 张人工标注。`);
    await refreshStatus();
    await loadReviewQueue(false);
    await selectReviewEvent(result.event.id);
  } catch (error) {
    toast(`导入失败：${error.message}`);
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
$('#refresh-dataset').addEventListener('click', loadDatasetPage);
$('#dataset-classes').addEventListener('change', updateDatasetSelection);
$('#select-eligible').addEventListener('click', () => {
  const ready = new Set(state.datasetReadiness?.default_classes || []);
  $$('#dataset-classes input[type="checkbox"]').forEach((input) => {
    input.checked = ready.has(Number(input.value));
  });
  updateDatasetSelection();
});
$('#build-dataset').addEventListener('click', async () => {
  const classIds = selectedDatasetClasses();
  if (!classIds.length) {
    toast('请至少选择一个类别。');
    return;
  }
  $('#build-dataset').disabled = true;
  try {
    const result = await api('/api/datasets/build', {
      method: 'POST',
      body: JSON.stringify({ class_ids: classIds }),
    });
    toast(`数据集 ${result.version} 已生成。`);
    await loadDatasetPage();
  } catch (error) {
    toast(`构建失败：${error.message}`);
    updateDatasetSelection();
  }
});
$('#refresh-training').addEventListener('click', () => refreshTraining(true));
$('#training-jobs').addEventListener('click', (event) => {
  const item = event.target.closest('[data-job-id]');
  if (item) selectTrainingJob(item.dataset.jobId);
});
$('#start-training').addEventListener('click', async () => {
  const datasetVersion = $('#train-dataset').value;
  if (!datasetVersion) {
    toast('请先构建数据集版本。');
    return;
  }
  try {
    const job = await api('/api/training/start', {
      method: 'POST',
      body: JSON.stringify({
        dataset_version: datasetVersion,
        model: $('#train-model').value,
        epochs: Number($('#train-epochs').value),
        imgsz: Number($('#train-imgsz').value),
        batch: Number($('#train-batch').value),
        patience: Number($('#train-patience').value),
      }),
    });
    state.currentJob = job;
    toast(`训练任务已启动：${job.id}`);
    await refreshTraining(true);
    await selectTrainingJob(job.id);
  } catch (error) {
    toast(`启动失败：${error.message}`);
  }
});
$('#run-test').addEventListener('click', async () => {
  if (!state.currentJob) return;
  try {
    const job = await api(`/api/training/${encodeURIComponent(state.currentJob.id)}/test`, {
      method: 'POST',
      body: JSON.stringify({
        imgsz: Number($('#train-imgsz').value),
        batch: Number($('#train-batch').value),
      }),
    });
    toast(`最终测试已启动：${job.id}`);
    await refreshTraining(true);
    await selectTrainingJob(job.id);
  } catch (error) {
    toast(`无法运行测试：${error.message}`);
  }
});
$('#cancel-job').addEventListener('click', async () => {
  if (!state.currentJob) return;
  try {
    await api(`/api/training/${encodeURIComponent(state.currentJob.id)}/cancel`, { method: 'POST' });
    toast('任务已取消。');
    await refreshTraining(true);
  } catch (error) {
    toast(`取消失败：${error.message}`);
  }
});
$('#refresh-deployment').addEventListener('click', loadDeploymentPage);
$('#refresh-capture').addEventListener('click', loadCapturePage);
$('#scan-filters').addEventListener('click', (event) => {
  const button = event.target.closest('[data-scan-filter]');
  if (!button) return;
  state.scanFilter = button.dataset.scanFilter;
  state.scanOffset = 0;
  $$('#scan-filters button').forEach((item) => item.classList.toggle('active', item === button));
  loadScanFrames();
});
$('#scan-round').addEventListener('change', (event) => {
  state.scanRound = event.target.value;
  state.scanOffset = 0;
  loadScanFrames();
});
$('#scan-grid').addEventListener('click', (event) => {
  const card = event.target.closest('[data-frame-id]');
  if (card) selectScanFrame(card.dataset.frameId);
});
$('#scan-prev').addEventListener('click', () => {
  state.scanOffset = Math.max(0, state.scanOffset - state.scanLimit);
  loadScanFrames();
});
$('#scan-next').addEventListener('click', () => {
  state.scanOffset += state.scanLimit;
  loadScanFrames();
});
$('#scan-candidate').addEventListener('click', () => markScanFrame('candidate', { advance: true, prepare: true }));
$('#scan-not-new').addEventListener('click', () => markScanFrame('no_action', { advance: true }));
$('#scan-reset').addEventListener('click', () => markScanFrame('reset'));
$('#scan-prepare').addEventListener('click', async () => {
  if (!state.scanCurrent) return;
  try {
    const packageInfo = await api(`/api/raw-frames/${encodeURIComponent(state.scanCurrent.id)}/prepare-label`, {
      method: 'POST',
    });
    state.scanPackage = packageInfo;
    $('#scan-package').textContent = packageInfo.folder;
    $('#scan-open-folder').disabled = false;
    toast('标注目录已生成。');
  } catch (error) {
    toast(`生成失败：${error.message}`);
  }
});
$('#scan-open-folder').addEventListener('click', async () => {
  if (!state.scanCurrent) return;
  try {
    await api(`/api/raw-frames/${encodeURIComponent(state.scanCurrent.id)}/open-label-folder`, {
      method: 'POST',
    });
    toast('已在文件资源管理器中打开标注目录。');
  } catch (error) {
    toast(`打开目录失败：${error.message}`);
  }
});
$('#scan-launch-xany').addEventListener('click', async () => {
  try {
    await api('/api/xany/launch', { method: 'POST' });
    toast('已启动 X-AnyLabeling。请选择刚才打开的标注目录。');
  } catch (error) {
    toast(`启动失败：${error.message}`);
  }
});
$('#scan-save-xany').addEventListener('click', async () => {
  try {
    await api('/api/xany/config', {
      method: 'POST',
      body: JSON.stringify({ executable: $('#scan-xany-executable').value }),
    });
    $('#xany-executable').value = $('#scan-xany-executable').value;
    toast('X-AnyLabeling 路径已保存。');
  } catch (error) {
    toast(`保存失败：${error.message}`);
  }
});
$('#scan-import').addEventListener('click', async () => {
  if (!state.scanCurrent) return;
  try {
    const result = await api(`/api/raw-frames/${encodeURIComponent(state.scanCurrent.id)}/import-label`, {
      method: 'POST',
    });
    renderScanDetail(result.frame);
    toast(`已导入 ${result.imported} 个人工框。`);
  } catch (error) {
    toast(`导入失败：${error.message}`);
  }
});
$('#scan-image').addEventListener('load', renderScanBoxes);
window.addEventListener('resize', renderScanBoxes);
document.addEventListener('keydown', (event) => {
  if (!$('#page-scan').classList.contains('active')) return;
  if (event.target.closest('input, textarea, select, [contenteditable="true"]')) return;
  if (event.key === '1') {
    event.preventDefault();
    markScanFrame('no_action', { advance: true });
  } else if (event.key === '2') {
    event.preventDefault();
    markScanFrame('candidate', { advance: true, prepare: true });
  } else if (event.key === '0') {
    event.preventDefault();
    markScanFrame('skip', { advance: true });
  } else if (event.key === 'ArrowRight') {
    event.preventDefault();
    moveScan(1);
  } else if (event.key === 'ArrowLeft') {
    event.preventDefault();
    moveScan(-1);
  }
});
document.addEventListener('keydown', (event) => {
  if (!$('#page-review').classList.contains('active')) return;
  if (event.target.closest('input, textarea, select, [contenteditable="true"]')) return;
  if (event.key === '1') {
    event.preventDefault();
    if (state.reviewEvent?.old_class_id == null) {
      toast('旧模型没有给出类别，请选择真实类别或标记 UNKNOWN。');
      return;
    }
    reviewAction('correct');
  } else if (event.key === '2') {
    event.preventDefault();
    if (!state.reviewSelectedClassId) {
      $('#class-search').focus();
      toast('搜索并选择一个具体式神，再按 2 保存。');
      return;
    }
    reviewAction('change', state.reviewSelectedClassId);
  } else if (event.key === '3') {
    event.preventDefault();
    reviewAction('needs_label', null, { prepare: true });
  } else if (event.key === '4') {
    event.preventDefault();
    reviewAction('unknown');
  } else if (event.key === '0') {
    event.preventDefault();
    reviewAction('skip');
  } else if (event.key === 'ArrowRight') {
    event.preventDefault();
    moveReview(1);
  } else if (event.key === 'ArrowLeft') {
    event.preventDefault();
    moveReview(-1);
  }
});
$('#model-versions').addEventListener('click', async (event) => {
  const button = event.target.closest('[data-activate-model]');
  if (!button) return;
  try {
    await api(`/api/models/${encodeURIComponent(button.dataset.activateModel)}/activate`, {
      method: 'POST',
      body: JSON.stringify({ mode: $('#deploy-mode').value }),
    });
    toast(`已激活 ${button.dataset.activateModel}（${$('#deploy-mode').value}）`);
    await loadDeploymentPage();
  } catch (error) {
    toast(`激活失败：${error.message}`);
  }
});
$('#model-candidates').addEventListener('click', async (event) => {
  const button = event.target.closest('[data-publish-job]');
  if (!button) return;
  button.disabled = true;
  try {
    const result = await api(`/api/models/${encodeURIComponent(button.dataset.publishJob)}/publish`, {
      method: 'POST',
      body: JSON.stringify({ mode: $('#deploy-mode').value }),
    });
    toast(`已发布并激活 ${result.version}`);
    await loadDeploymentPage();
  } catch (error) {
    toast(`发布失败：${error.message}`);
    button.disabled = false;
  }
});
$('#start-inference').addEventListener('click', async () => {
  try {
    await api('/api/inference/start', { method: 'POST' });
    toast('GPU 推理服务已启动。');
    await loadDeploymentPage();
  } catch (error) {
    toast(`启动失败：${error.message}`);
  }
});
$('#stop-inference').addEventListener('click', async () => {
  try {
    await api('/api/inference/stop', { method: 'POST' });
    toast('GPU 推理服务已停止，OAS 将自动回退旧模型。');
    await loadDeploymentPage();
  } catch (error) {
    toast(`停止失败：${error.message}`);
  }
});

loadXanyConfig();
refreshStatus();

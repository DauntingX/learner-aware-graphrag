'use strict';

/* =========================================================================
 * AI 知识图谱 · 学习者感知可视化
 *
 * 数据来源：build_graph.py 导出的 graph_data.json
 * 核心交互：力导向探索 + 学习路径逐步播放动画
 *
 * 学习路径在浏览器端计算（图很小，88 节点），
 * 算法与 Python 端 src/kg/paths.py 保持一致：
 *   祖先收集 -> 教学序拓扑排序 -> 缺口标注
 * ========================================================================= */

const MASTERED_THRESHOLD = 0.85;
const GAP_THRESHOLD = 0.35;
const AUTOPLAY_MS = 1500;

const REASON = {
  known: '已掌握，可快速跳过',
  learning: '已有基础，复习巩固即可',
  gap: '前置缺口，建议优先补上',
};

const STATUS_META = {
  known: { label: '已掌握', color: '#1D9E75' },
  learning: { label: '学习中', color: '#EF9F27' },
  gap: { label: '缺口', color: '#E24B4A' },
};

const BAND_LABEL = {
  untouched: '未接触',
  weak: '薄弱',
  learning: '学习中',
  solid: '较扎实',
  mastered: '已掌握',
};

const REL_LABEL = {
  prerequisite_of: '前置',
  part_of: '属于',
  related_to: '相关',
  evolved_from: '演进自',
  applied_in: '应用于',
  implemented_by: '实现于',
  described_in: '出自',
  confusable_with: '易混淆',
};

const state = {
  nodes: [],
  links: [],
  rawLinks: [],
  byId: new Map(),
  meta: {},
  degree: new Map(),
  preds: new Map(),
  succs: new Map(),
  activeDomains: new Set(),
  activeRels: new Set(),
  selectedId: null,
  showAllLabels: false,
  path: null,
  sim: null,
  focusId: null,
};

let svg, gRoot, gLink, gNode, linkSel, nodeSel, zoom, W, H;

/* ---------------------------------------------------------------- 启动 */

init().catch(showFatal);

async function init() {
  // 优先读内联数据（standalone 版本），否则从服务器拉取
  let data = readInlineData();
  if (!data) {
    try {
      data = await d3.json('graph_data.json');
    } catch (err) {
      showFatal(new Error(
        '无法加载 graph_data.json。请通过本地服务器打开本页面（在项目根目录运行 python serve.py），' +
        '或改用双击即可打开的 web/standalone.html。'
      ));
      return;
    }
  }

  state.nodes = data.nodes;
  state.links = data.links.map((l) => Object.assign({}, l));
  state.rawLinks = data.links.map((l) => Object.assign({}, l));
  state.meta = data.meta || {};
  state.byId = new Map(state.nodes.map((n) => [n.id, n]));
  if (state.meta.demo) {
    document.querySelector('.brand p').textContent = '公开演示 · 完全合成的学习画像';
  }

  buildIndices();
  buildLegend();
  buildPanel();
  setupGraph();
  render();
  window.addEventListener('resize', onResize);

  // 支持用链接直接播放某条路径： standalone.html#path=graph_rag
  // 既方便分享，也方便自动化截图验证
  const m = /^#path=(.+)$/.exec(location.hash || '');
  if (m) {
    const targetId = decodeURIComponent(m[1]);
    if (state.byId.has(targetId)) {
      d3.select('#targetSelect').property('value', targetId);
      setTimeout(() => startPath(targetId), 1600);
    }
  }
}

function readInlineData() {
  const el = document.getElementById('kg-data');
  if (!el) return null;
  const txt = (el.textContent || '').trim();
  if (!txt) return null;
  try { return JSON.parse(txt); } catch (e) { return null; }
}

function showFatal(err) {
  const el = document.createElement('div');
  el.style.cssText =
    'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;' +
    'padding:40px;text-align:center;color:#A32D2D;font-size:13px;line-height:1.8;';
  el.textContent = err.message;
  document.querySelector('.stage').appendChild(el);
  console.error(err);
}

/* ---------------------------------------------------------------- 索引 */

function buildIndices() {
  state.degree = new Map();
  state.preds = new Map();
  state.succs = new Map();
  state.nodes.forEach((n) => {
    state.degree.set(n.id, 0);
    state.preds.set(n.id, []);
    state.succs.set(n.id, []);
  });

  state.rawLinks.forEach((l) => {
    state.degree.set(l.source, (state.degree.get(l.source) || 0) + 1);
    state.degree.set(l.target, (state.degree.get(l.target) || 0) + 1);
    if (l.type === 'prerequisite_of') {
      state.succs.get(l.source).push(l.target);
      state.preds.get(l.target).push(l.source);
    }
  });

  state.domains = [...new Set(state.nodes.map((n) => n.domain))].sort();
  state.relTypes = [...new Set(state.rawLinks.map((l) => l.type))];
  state.activeDomains = new Set(state.domains);
  state.activeRels = new Set(state.relTypes);
}

function radiusOf(d) {
  const k = state.degree.get(d.id) || 0;
  return 5 + Math.min(k, 10) * 0.78;
}

/* ------------------------------------------------------- 学习路径算法 */

function ancestorsOf(target) {
  const seen = new Set();
  const stack = [...(state.preds.get(target) || [])];
  while (stack.length) {
    const n = stack.pop();
    if (seen.has(n)) continue;
    seen.add(n);
    for (const p of state.preds.get(n) || []) stack.push(p);
  }
  return seen;
}

function depthsFrom(target, scope) {
  const depth = new Map([[target, 0]]);
  let frontier = [target];
  while (frontier.length) {
    const next = [];
    for (const n of frontier) {
      for (const p of state.preds.get(n) || []) {
        if (scope.has(p) && !depth.has(p)) {
          depth.set(p, depth.get(n) + 1);
          next.push(p);
        }
      }
    }
    frontier = next;
  }
  return depth;
}

/** 教学序拓扑排序：depth 大（更基础）优先，同 depth 难度低优先。 */
function pedagogicalOrder(scope, depth) {
  const indeg = new Map();
  for (const n of scope) {
    indeg.set(n, (state.preds.get(n) || []).filter((p) => scope.has(p)).length);
  }

  const remaining = new Set(scope);
  const out = [];
  while (remaining.size) {
    let best = null;
    for (const n of remaining) {
      if (indeg.get(n) !== 0) continue;
      if (best === null) { best = n; continue; }
      const dn = depth.get(n) ?? -1;
      const db = depth.get(best) ?? -1;
      if (dn > db) best = n;
      else if (dn === db) {
        const fn = (state.byId.get(n) || {}).difficulty ?? 3;
        const fb = (state.byId.get(best) || {}).difficulty ?? 3;
        if (fn < fb) best = n;
      }
    }
    if (best === null) break; // 理论上不会发生：DAG 已在构建期校验
    out.push(best);
    remaining.delete(best);
    for (const s of state.succs.get(best) || []) {
      if (indeg.has(s)) indeg.set(s, indeg.get(s) - 1);
    }
  }
  return out;
}

function statusOf(mastery) {
  if (mastery >= MASTERED_THRESHOLD) return 'known';
  if (mastery >= GAP_THRESHOLD) return 'learning';
  return 'gap';
}

function computePath(targetId) {
  const scope = ancestorsOf(targetId);
  scope.add(targetId);
  const depth = depthsFrom(targetId, scope);

  return pedagogicalOrder(scope, depth).map((id, i) => {
    const n = state.byId.get(id);
    const mastery = n ? n.mastery : 0;
    const status = statusOf(mastery);
    return {
      order: i,
      id,
      name: n ? n.name : id,
      domain: n ? n.domain : '',
      difficulty: n ? n.difficulty : 3,
      desc: n ? n.desc : '',
      mastery,
      color: n ? n.color : '#B4B2A9',
      depth: depth.get(id) ?? -1,
      status,
      reason: REASON[status],
    };
  });
}

/* ---------------------------------------------------------------- 图 */

function setupGraph() {
  const stage = document.querySelector('.stage');
  W = stage.clientWidth;
  H = stage.clientHeight;

  svg = d3.select('#graph').attr('viewBox', [0, 0, W, H]);

  gRoot = svg.append('g');
  gLink = gRoot.append('g');
  gNode = gRoot.append('g');

  zoom = d3.zoom().scaleExtent([0.15, 5]).on('zoom', (e) => {
    gRoot.attr('transform', e.transform);
  });
  svg.call(zoom).on('dblclick.zoom', null);
  svg.on('click', () => { if (!state.path) hideDetail(); });

  const sim = d3.forceSimulation(state.nodes)
    .force('link', d3.forceLink(state.links).id((d) => d.id)
      .distance((d) => (d.type === 'prerequisite_of' ? 56 : 72))
      .strength((d) => (d.type === 'prerequisite_of' ? 0.62 : 0.2)))
    .force('charge', d3.forceManyBody().strength(-190).distanceMax(330))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collide', d3.forceCollide().radius((d) => radiusOf(d) + 9))
    .force('x', d3.forceX(W / 2).strength(0.02))
    .force('y', d3.forceY(H / 2).strength(0.02));

  // 同向多条边做曲线偏移，避免完全重叠
  const pairCount = new Map();
  state.links.forEach((l) => {
    const k = l.source.id ? `${l.source.id}|${l.target.id}` : `${l.source}|${l.target}`;
    pairCount.set(k, (pairCount.get(k) || 0) + 1);
  });
  const pairSeen = new Map();
  state.links.forEach((l) => {
    const s = l.source.id || l.source;
    const t = l.target.id || l.target;
    const k = `${s}|${t}`;
    const total = pairCount.get(k) || 1;
    const i = pairSeen.get(k) || 0;
    pairSeen.set(k, i + 1);
    l._offset = total > 1 ? (i - (total - 1) / 2) * 30 : 0;
  });

  linkSel = gLink.selectAll('path').data(state.links).join('path')
    .attr('class', 'link')
    .attr('stroke', (d) => d.color)
    .attr('stroke-width', (d) => d.width)
    .attr('stroke-dasharray', (d) => d.dash || null)
    .attr('opacity', 0.55);

  nodeSel = gNode.selectAll('g').data(state.nodes).join('g')
    .attr('class', 'node');

  nodeSel.append('circle')
    .attr('class', 'halo')
    .attr('display', 'none');

  nodeSel.append('circle')
    .attr('class', 'node-circle')
    .attr('r', radiusOf)
    .attr('fill', (d) => d.color);

  nodeSel.append('text')
    .attr('class', 'badge')
    .attr('text-anchor', 'middle')
    .attr('dominant-baseline', 'central')
    .attr('display', 'none');

  nodeSel.append('text')
    .attr('class', 'node-label')
    .attr('text-anchor', 'middle')
    .text((d) => d.name);

  nodeSel
    .on('mouseenter', onHover)
    .on('mousemove', onHoverMove)
    .on('mouseleave', () => { d3.select('#tooltip').attr('hidden', true); })
    .on('click', (event, d) => { event.stopPropagation(); showDetail(d.id); })
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.25).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));

  sim.on('tick', () => {
    linkSel.attr('d', linkPath);
    nodeSel.attr('transform', (d) => `translate(${d.x},${d.y})`);
    nodeSel.select('text.node-label').attr('y', (d) => radiusOf(d) + 12);
    nodeSel.select('text.badge').attr('x', (d) => radiusOf(d) * 1.9 + 6).attr('y', (d) => -radiusOf(d) * 1.5);
    nodeSel.select('circle.halo').attr('r', (d) => (d._haloR || radiusOf(d) + 5));
  });

  state.sim = sim;
  updateLabelVisibility();
}

function linkPath(d) {
  const sx = d.source.x, sy = d.source.y, tx = d.target.x, ty = d.target.y;
  const off = d._offset || 0;
  if (!off) return `M${sx},${sy}L${tx},${ty}`;
  const dx = tx - sx, dy = ty - sy;
  const len = Math.hypot(dx, dy) || 1;
  const mx = (sx + tx) / 2 - (dy / len) * off;
  const my = (sy + ty) / 2 + (dx / len) * off;
  return `M${sx},${sy}Q${mx},${my} ${tx},${ty}`;
}

function onHover(event, d) {
  const tip = d3.select('#tooltip');
  tip.attr('hidden', null).html(
    `<b>${esc(d.name)}</b><br>${esc(d.domain)} · 难度 ${d.difficulty}<br>` +
    `掌握度 ${(d.mastery * 100).toFixed(0)}% · ${BAND_LABEL[d.band]}`
  );
  onHoverMove(event);
}

function onHoverMove(event) {
  const stage = document.querySelector('.stage');
  const rect = stage.getBoundingClientRect();
  d3.select('#tooltip')
    .style('left', event.clientX - rect.left + 14 + 'px')
    .style('top', event.clientY - rect.top + 14 + 'px');
}

function onResize() {
  const stage = document.querySelector('.stage');
  W = stage.clientWidth;
  H = stage.clientHeight;
  svg.attr('viewBox', [0, 0, W, H]);
  state.sim.force('center', d3.forceCenter(W / 2, H / 2));
  state.sim.force('x', d3.forceX(W / 2).strength(0.02));
  state.sim.force('y', d3.forceY(H / 2).strength(0.02));
  state.sim.alpha(0.3).restart();
  if (state.path) focusPathBounds(state.path.steps);
}

function render() {
  applyFilters();
  updateLabelVisibility();
  renderStats();
}

/* ---------------------------------------------------------------- 筛选 */

function applyFilters() {
  const visible = (d) => state.activeDomains.has(d.domain);

  nodeSel.classed('off', (d) => !visible(d));
  linkSel.classed('off', (l) => {
    const s = state.byId.get(l.source.id || l.source);
    const t = state.byId.get(l.target.id || l.target);
    return !state.activeRels.has(l.type) || !s || !t || !visible(s) || !visible(t);
  });

  // 只让可见节点参与力布局，否则被隐藏的节点会继续占位、把可见节点挤成空洞
  if (state.sim && !state.path) {
    state.sim.nodes(state.nodes.filter(visible));
    state.sim.force('link').links(
      state.links.filter((l) => {
        const s = state.byId.get(l.source.id || l.source);
        const t = state.byId.get(l.target.id || l.target);
        return state.activeRels.has(l.type) && s && t && visible(s) && visible(t);
      })
    );
    state.sim.alpha(0.5).restart();
  }
}

function updateLabelVisibility() {
  nodeSel.select('text.node-label').attr('display', (d) => {
    if (state.showAllLabels) return null;
    if (state.focusId === d.id) return null;
    return (state.degree.get(d.id) || 0) >= 5 ? null : 'none';
  });
}

/* ---------------------------------------------------------------- 面板 */

function buildLegend() {
  const order = ['untouched', 'weak', 'learning', 'solid', 'mastered'];
  const counts = {};
  state.nodes.forEach((n) => { counts[n.band] = (counts[n.band] || 0) + 1; });

  d3.select('#legend').selectAll('li').data(order).join('li')
    .html((b) => {
      const color = state.nodes.find((n) => n.band === b)?.color || '#B4B2A9';
      return `<i style="background:${color}"></i>${BAND_LABEL[b]}<em>${counts[b] || 0}</em>`;
    });
}

function buildPanel() {
  // 领域 chips
  d3.select('#domainChips').selectAll('button').data(state.domains).join('button')
    .attr('class', 'chip on')
    .text((d) => d)
    .on('click', function (event, d) {
      toggleSet(state.activeDomains, d);
      d3.select(this).classed('on', state.activeDomains.has(d));
      render();
    });

  d3.select('[data-domain-all]').on('click', () => {
    state.activeDomains = new Set(state.domains);
    d3.select('#domainChips').selectAll('button').classed('on', true);
    render();
  });
  d3.select('[data-domain-none]').on('click', () => {
    state.activeDomains = new Set();
    d3.select('#domainChips').selectAll('button').classed('on', false);
    render();
  });

  // 关系 chips
  d3.select('#relChips').selectAll('button').data(state.relTypes).join('button')
    .attr('class', 'chip on rel')
    .html((d) => {
      const dash = (state.rawLinks.find((l) => l.type === d) || {}).dash ? ' dash' : '';
      return `<i class="${dash.trim()}"></i>${esc(REL_LABEL[d] || d)}`;
    })
    .on('click', function (event, d) {
      toggleSet(state.activeRels, d);
      d3.select(this).classed('on', state.activeRels.has(d));
      render();
    });

  // 目标下拉
  const groups = d3.group(
    state.nodes.slice().sort((a, b) => a.name.localeCompare(b.name, 'zh')),
    (n) => n.domain
  );
  const sel = d3.select('#targetSelect');
  sel.append('option').attr('value', '').text('— 选择一个目标 —');
  [...groups.keys()].sort().forEach((domain) => {
    const og = sel.append('optgroup').attr('label', domain);
    groups.get(domain).forEach((n) => {
      og.append('option').attr('value', n.id).text(n.name);
    });
  });

  sel.on('change', function () {
    const v = this.value;
    if (v) {
      const steps = computePath(v);
      const gaps = steps.filter((s) => s.status === 'gap').length;
      d3.select('#pathHint').html(
        `到 <b>${esc(state.byId.get(v).name)}</b> 共 <b>${steps.length}</b> 步，其中 <b>${gaps}</b> 个缺口。`
      );
    } else {
      d3.select('#pathHint').text('选一个目标，看从当前水平到它要走哪些步。');
    }
  });

  d3.select('#playBtn').on('click', () => {
    const v = d3.select('#targetSelect').property('value');
    if (!v) { d3.select('#pathHint').text('请先选择一个目标知识点。'); return; }
    startPath(v);
  });

  // 详情关闭 / 播放器控制
  d3.select('#prevBtn').on('click', () => { pause(); goTo(state.path.idx - 1); });
  d3.select('#nextBtn').on('click', () => { pause(); goTo(state.path.idx + 1); });
  d3.select('#autoBtn').on('click', () => { state.path.playing ? pause() : play(); });
  d3.select('#exitBtn').on('click', exitPath);

  // 搜索
  d3.select('#search').on('keydown', function (event) {
    if (event.key !== 'Enter') return;
    const kw = this.value.trim().toLowerCase();
    if (!kw) return;
    const hit = state.nodes.find(
      (n) => n.name.toLowerCase().includes(kw) || n.id.includes(kw)
    );
    if (!hit) { d3.select('#pathHint').text(`没找到「${kw}」`); return; }
    focusOn(hit.id);
  });

  d3.select('#allLabels').on('change', function () {
    state.showAllLabels = this.checked;
    updateLabelVisibility();
  });
}

function toggleSet(set, v) {
  if (set.has(v)) set.delete(v); else set.add(v);
}

function renderStats() {
  const m = state.meta.stats || {};
  const avg = d3.mean(state.nodes, (n) => n.mastery) || 0;
  d3.select('#stats').html(
    `<div class="stat"><b>${m.nodes || state.nodes.length}</b><span>知识点</span></div>` +
    `<div class="stat"><b>${m.edges || state.links.length}</b><span>关系</span></div>` +
    `<div class="stat"><b>${state.domains.length}</b><span>领域</span></div>` +
    `<div class="stat"><b>${(avg * 100).toFixed(0)}%</b><span>平均掌握度</span></div>`
  );
}

/* ---------------------------------------------------------------- 焦点 */

function focusOn(id) {
  const d = state.byId.get(id);
  if (!d || d.x == null) return;
  state.focusId = id;
  updateLabelVisibility();
  const k = 1.5;
  svg.transition().duration(650).call(
    zoom.transform,
    d3.zoomIdentity.translate(W / 2 - k * d.x, H / 2 - k * d.y).scale(k)
  );
  showDetail(id);
}

/* ---------------------------------------------------------------- 详情 */

function showDetail(id) {
  const d = state.byId.get(id);
  if (!d) return;
  state.selectedId = id;

  const prereqNames = (state.preds.get(id) || []).map((p) => state.byId.get(p)).filter(Boolean);
  const succNames = (state.succs.get(id) || []).map((p) => state.byId.get(p)).filter(Boolean);
  const confusable = state.rawLinks
    .filter((l) => l.type === 'confusable_with' && (l.source === id || l.target === id))
    .map((l) => (l.source === id ? l.target : l.source))
    .map((x) => state.byId.get(x))
    .filter(Boolean);

  const pct = (d.mastery * 100).toFixed(0);

  const detail = d3.select('#detail').attr('hidden', null).html(
    `<button class="close" type="button" data-close-detail aria-label="关闭详情">&times;</button>` +
    `<h3>${esc(d.name)}</h3>` +
    `<div class="sub">${esc(d.domain)} · ${esc(d.type)} · 难度 ${d.difficulty}/5</div>` +
    `<p>${esc(d.desc || '（暂无描述）')}</p>` +
    `<div class="kv"><span>掌握度</span><span>${pct}% · ${BAND_LABEL[d.band]}</span></div>` +
    `<div class="mbar"><i style="width:${pct}%;background:${d.color}"></i></div>` +
    (d.aliases && d.aliases.length
      ? `<div class="sect">别名</div><div>${d.aliases.map((a) => `<span class="tag">${esc(a)}</span>`).join('')}</div>`
      : '') +
    (prereqNames.length
      ? `<div class="sect">前置知识（${prereqNames.length}）</div>` +
        prereqNames.map((n) => `<button class="tag" type="button" data-node-id="${esc(n.id)}">${esc(n.name)}</button>`).join('')
      : '') +
    (succNames.length
      ? `<div class="sect">学会后可解锁（${succNames.length}）</div>` +
        succNames.map((n) => `<button class="tag" type="button" data-node-id="${esc(n.id)}">${esc(n.name)}</button>`).join('')
      : '') +
    (confusable.length
      ? `<div class="sect">易混淆</div>` +
        confusable.map((n) => `<button class="tag" type="button" data-node-id="${esc(n.id)}">${esc(n.name)}</button>`).join('')
      : '')
  );
  detail.select('[data-close-detail]').on('click', hideDetail);
  detail.selectAll('[data-node-id]').on('click', function () {
    focusOn(this.dataset.nodeId);
  });
}

function hideDetail() {
  state.selectedId = null;
  d3.select('#detail').attr('hidden', true);
}

/* ------------------------------------------------------- 学习路径动画 */

function startPath(targetId) {
  const steps = computePath(targetId);
  if (!steps.length) return;

  state.path = { targetId, steps, idx: -1, playing: false, timer: null };
  hideDetail();

  d3.select('#player').attr('hidden', null);
  d3.select('#stepCard').html('');
  d3.select('#bar').style('width', '0%');

  focusPathBounds(steps);

  goTo(0);
  play();
}

function play() {
  const p = state.path;
  if (!p) return;
  p.playing = true;
  d3.select('#autoBtn').text('暂停').classed('solid', true);
  clearInterval(p.timer);
  p.timer = setInterval(() => {
    if (p.idx >= p.steps.length - 1) { pause(); return; }
    goTo(p.idx + 1);
  }, AUTOPLAY_MS);
}

function pause() {
  const p = state.path;
  if (!p) return;
  p.playing = false;
  clearInterval(p.timer);
  p.timer = null;
  d3.select('#autoBtn').text('继续').classed('solid', true);
}

function goTo(i) {
  const p = state.path;
  if (!p) return;
  const idx = Math.max(0, Math.min(i, p.steps.length - 1));
  p.idx = idx;

  const step = p.steps[idx];
  paintPath(idx);
  renderStepCard(step, idx, p.steps.length);

  d3.select('#bar').style('width', `${((idx + 1) / p.steps.length) * 100}%`);
  // 播放期间镜头不动：整条路径已经框进视野，
  // 靠节点放大 + 光环 + 序号来指示"走到哪了"，比镜头乱晃更清楚。
}

function paintPath(idx) {
  const p = state.path;
  const inPath = new Set(p.steps.map((s) => s.id));
  const visited = new Set(p.steps.slice(0, idx + 1).map((s) => s.id));
  const current = p.steps[idx].id;

  nodeSel.classed('dimmed', (d) => !inPath.has(d.id));
  linkSel.classed('dimmed', (l) => {
    const s = l.source.id || l.source;
    const t = l.target.id || l.target;
    return !(l.type === 'prerequisite_of' && inPath.has(s) && inPath.has(t));
  });
  linkSel.filter((l) => {
    const s = l.source.id || l.source;
    const t = l.target.id || l.target;
    return l.type === 'prerequisite_of' && inPath.has(s) && inPath.has(t);
  }).attr('opacity', 0.85);

  nodeSel.each(function (d) {
    const g = d3.select(this);
    const r = radiusOf(d);
    const isVisited = visited.has(d.id);
    const isCurrent = d.id === current;

    g.select('circle.node-circle')
      .transition().duration(260)
      .attr('r', isCurrent ? r * 1.85 : r);

    const haloR = isCurrent ? r * 2.9 : r * 1.9;
    d._haloR = haloR;
    g.select('circle.halo')
      .attr('display', isVisited ? null : 'none')
      .attr('stroke', d.color)
      .attr('stroke-opacity', isCurrent ? 1 : 0.4)
      .attr('r', haloR);

    const order = p.steps.findIndex((s) => s.id === d.id);
    g.select('text.badge')
      .attr('display', isVisited ? null : 'none')
      .text(isVisited ? String(order + 1) : '')
      .attr('fill', d.color)
      .attr('stroke', '#fff')
      .attr('stroke-width', 3)
      .attr('paint-order', 'stroke');

    g.select('text.node-label')
      .attr('display', inPath.has(d.id) ? null : 'none')
      .style('font-size', isCurrent ? '12.5px' : '11px')
      .style('font-weight', isCurrent ? 500 : 400);
  });
}

function renderStepCard(step, idx, total) {
  const meta = STATUS_META[step.status];
  d3.select('#stepCard').html(
    `<div class="step-num" style="background:${meta.color}">${idx + 1}</div>` +
    `<div class="step-body">` +
      `<h4>${esc(step.name)}</h4>` +
      `<div class="meta">第 ${idx + 1}/${total} 步 · ${esc(step.domain)} · 难度 ${step.difficulty} · ` +
        `当前掌握度 ${(step.mastery * 100).toFixed(0)}% · <b style="color:${meta.color}">${meta.label}</b></div>` +
      `<div class="reason">${esc(step.reason)}${step.desc ? '　' + esc(step.desc) : ''}</div>` +
    `</div>`
  );
}

function centerOn(id) {
  const d = state.byId.get(id);
  if (!d || d.x == null) return;
  const k = 1.25;
  svg.transition().duration(620).call(
    zoom.transform,
    d3.zoomIdentity.translate(W / 2 - k * d.x, H / 2 - k * d.y).scale(k)
  );
}

function focusPathBounds(steps) {
  const pts = steps.map((s) => state.byId.get(s.id)).filter((d) => d && d.x != null);
  if (!pts.length) return;
  const x0 = d3.min(pts, (p) => p.x), x1 = d3.max(pts, (p) => p.x);
  const y0 = d3.min(pts, (p) => p.y), y1 = d3.max(pts, (p) => p.y);
  const pad = Math.min(110, Math.max(28, W * 0.08));
  const overlayHeight = document.querySelector('#player').offsetHeight || 130;
  const graphHeight = Math.max(H - overlayHeight, 100);
  const k = Math.min(
    (W - pad * 2) / Math.max(x1 - x0, 1),
    (graphHeight - pad * 2) / Math.max(y1 - y0, 1),
    1.35
  );
  const kk = Math.min(Math.max(k, 0.15), 1.3);
  svg.transition().duration(750).call(
    zoom.transform,
    d3.zoomIdentity
      .translate(W / 2 - kk * (x0 + x1) / 2, graphHeight / 2 - kk * (y0 + y1) / 2)
      .scale(kk)
  );
}

function exitPath() {
  const p = state.path;
  if (p) clearInterval(p.timer);
  state.path = null;

  d3.select('#player').attr('hidden', true);
  nodeSel.classed('dimmed', false);
  linkSel.classed('dimmed', false).attr('opacity', 0.55);
  nodeSel.each(function (d) {
    const g = d3.select(this);
    g.select('circle.node-circle').transition().duration(300).attr('r', radiusOf(d));
    g.select('circle.halo').attr('display', 'none');
    g.select('text.badge').attr('display', 'none');
  });
  nodeSel.select('text.node-label').style('font-size', '11px').style('font-weight', 400);
  updateLabelVisibility();

  svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity.translate(0, 0).scale(1));
}

/* ---------------------------------------------------------------- 工具 */

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
  ));
}

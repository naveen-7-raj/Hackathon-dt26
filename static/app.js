/* ================================================================
   app.js – FairLens AI  v2.1  (Fixed: auto-init + status polling)
   ================================================================ */
'use strict';

Chart.defaults.color = '#64748b';
Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
Chart.defaults.font.size = 12;

const S = { ready: false, overview: null, bias: null, shap: null, charts: {} };

/* ══════════════════════════════════════════════════════════════
   PARTICLES
══════════════════════════════════════════════════════════════ */
(function initParticles() {
  const canvas = document.getElementById('particles');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, dots = [];

  function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
  resize();
  window.addEventListener('resize', resize);

  const COLORS = ['rgba(124,58,237,.6)','rgba(6,182,212,.5)','rgba(16,185,129,.4)','rgba(236,72,153,.4)'];
  for (let i = 0; i < 55; i++) {
    dots.push({
      x: Math.random() * 1000, y: Math.random() * 800,
      r: Math.random() * 1.5 + 0.3,
      vx: (Math.random() - .5) * .3, vy: (Math.random() - .5) * .25,
      c: COLORS[Math.floor(Math.random() * COLORS.length)],
    });
  }
  function tick() {
    ctx.clearRect(0, 0, W, H);
    dots.forEach(d => {
      d.x += d.vx; d.y += d.vy;
      if (d.x < 0) d.x = W; if (d.x > W) d.x = 0;
      if (d.y < 0) d.y = H; if (d.y > H) d.y = 0;
      ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
      ctx.fillStyle = d.c; ctx.fill();
    });
    for (let i = 0; i < dots.length; i++) {
      for (let j = i + 1; j < dots.length; j++) {
        const dx = dots[i].x - dots[j].x, dy = dots[i].y - dots[j].y;
        const dist = Math.hypot(dx, dy);
        if (dist < 130) {
          ctx.beginPath();
          ctx.moveTo(dots[i].x, dots[i].y); ctx.lineTo(dots[j].x, dots[j].y);
          ctx.strokeStyle = `rgba(124,58,237,${(1 - dist / 130) * 0.12})`; ctx.lineWidth = 0.6; ctx.stroke();
        }
      }
    }
    requestAnimationFrame(tick);
  }
  tick();
})();

/* ══════════════════════════════════════════════════════════════
   NAVIGATION
══════════════════════════════════════════════════════════════ */
document.querySelectorAll('.nav-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.section).classList.add('active');
  });
});

function gotoTab(id) {
  const tab = document.querySelector(`[data-section="${id}"]`);
  if (tab) tab.click();
}

/* ══════════════════════════════════════════════════════════════
   STATUS POLLING  (replaces old manual-init flow)
══════════════════════════════════════════════════════════════ */
const overlay   = document.getElementById('loading-overlay');
const stepEl    = document.getElementById('loading-step');
const barEl     = document.getElementById('load-bar');
const initBtn   = document.getElementById('btn-init');
const statusDot = document.getElementById('status-dot');
const statusTxt = document.getElementById('status-text');

const STEP_MSGS = [
  'Generating 100,000 synthetic records…',
  'Embedding intentional bias into approval logic…',
  'Splitting 80/20 train / test sets…',
  'Training biased Logistic Regression…',
  'Computing fairness sample weights…',
  'Training fairness-reweighted model…',
  'Running SHAP LinearExplainer…',
  'Computing bias metrics…',
  'Finalising dashboard…',
];
let msgIdx = 0, msgTimer = null, pollTimer = null;

function rotateMsgs() {
  stepEl.textContent = STEP_MSGS[msgIdx % STEP_MSGS.length];
  msgIdx++;
  const pct = Math.min(10 + msgIdx * 10, 90);
  barEl.style.width = pct + '%';
}

function showOverlay() {
  overlay.classList.remove('hidden');
  msgIdx = 0;
  rotateMsgs();
  msgTimer = setInterval(rotateMsgs, 3000);
}

function hideOverlay() {
  clearInterval(msgTimer);
  barEl.style.width = '100%';
  stepEl.textContent = 'Complete!';
  setTimeout(() => overlay.classList.add('hidden'), 600);
}

async function pollStatus() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();

    if (data.status === 'done') {
      clearInterval(pollTimer);
      hideOverlay();
      await loadDashboardData();
      markReady();

    } else if (data.status === 'error') {
      clearInterval(pollTimer);
      clearInterval(msgTimer);
      overlay.classList.add('hidden');
      barEl.style.width = '0%';
      alert('Initialization failed:\n' + data.error);
      if (initBtn) { initBtn.disabled = false; initBtn.textContent = '🔄 Retry Initialize'; }

    } else if (data.status === 'running') {
      // still going – keep polling, update step text from server
      if (data.step) stepEl.textContent = data.step;
    }
    // idle = not started yet → do nothing (happens only briefly at startup)

  } catch (e) {
    // server probably restarting – keep polling
    console.warn('Poll error:', e);
  }
}

async function loadDashboardData() {
  const [ov, bias, shap] = await Promise.all([
    fetch('/api/overview').then(r => r.json()),
    fetch('/api/bias').then(r => r.json()),
    fetch('/api/shap/global').then(r => r.json()),
  ]);
  S.overview = ov; S.bias = bias; S.shap = shap; S.ready = true;
  renderAll();
}

function markReady() {
  statusDot.classList.add('ready');
  statusTxt.textContent = 'System Ready';
  if (initBtn) { initBtn.textContent = '✅ System Ready'; initBtn.disabled = true; }
  gotoTab('section-overview');
}

// ── Initialize on page load ───────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  // Check current status immediately
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();

    if (data.status === 'done') {
      // Already initialized (server was running before page load)
      hideOverlay();
      await loadDashboardData();
      markReady();

    } else if (data.status === 'running') {
      // Still computing – show overlay and poll
      showOverlay();
      pollTimer = setInterval(pollStatus, 2000);

    } else {
      // idle or error – start it
      showOverlay();
      await fetch('/api/initialize', { method: 'POST' });
      pollTimer = setInterval(pollStatus, 2000);
    }
  } catch (e) {
    console.error('Failed to get status:', e);
  }
});

// Manual init button – used when auto-init failed or user wants to retry
if (initBtn) {
  initBtn.addEventListener('click', async () => {
    if (S.ready) { gotoTab('section-overview'); return; }
    initBtn.disabled = true;
    showOverlay();
    try {
      await fetch('/api/initialize', { method: 'POST' });
    } catch (_) {}
    clearInterval(pollTimer);
    pollTimer = setInterval(pollStatus, 2000);
  });
}

/* ══════════════════════════════════════════════════════════════
   RENDER ALL
══════════════════════════════════════════════════════════════ */
function renderAll() {
  renderOverview();
  renderBias();
  renderSHAP();
  renderComparison();
  initWhatIf();
}

/* ── OVERVIEW ─────────────────────────────────────────────── */
function renderOverview() {
  const ov = S.overview;
  animCount('stat-total',    ov.n_samples,        0, '');
  animCount('stat-approval', ov.approval_rate*100, 1, '%');
  animCount('stat-income',   ov.income_median,     0, '', '$');
  animCount('stat-age',      ov.age_mean,          1, ' yrs');

  barChart('chart-approval-gender', {
    labels: Object.keys(ov.approval_by_gender).map(cap),
    values: Object.values(ov.approval_by_gender).map(v => +(v*100).toFixed(2)),
    colors: ['rgba(139,92,246,.75)','rgba(236,72,153,.75)'], label: 'Approval Rate %',
  });
  barChart('chart-approval-region', {
    labels: Object.keys(ov.approval_by_region).map(cap),
    values: Object.values(ov.approval_by_region).map(v => +(v*100).toFixed(2)),
    colors: ['rgba(59,130,246,.75)','rgba(16,185,129,.75)'], label: 'Approval Rate %',
  });
  const eduLabels = ['No Degree','High School','Some College',"Bachelor's",'Graduate'];
  barChart('chart-approval-education', {
    labels: eduLabels,
    values: Object.values(ov.approval_by_education).map(v => +(v*100).toFixed(2)),
    colors: ['rgba(245,158,11,.6)','rgba(251,146,60,.65)','rgba(139,92,246,.65)','rgba(99,102,241,.7)','rgba(16,185,129,.75)'],
    label: 'Approval Rate %',
  });
}

/* ── BIAS ─────────────────────────────────────────────────── */
function renderBias() {
  const { before, after, accuracy_biased, accuracy_fair } = S.bias;
  animCount('acc-biased', accuracy_biased*100, 2, '%');
  animCount('acc-fair',   accuracy_fair*100,   2, '%');
  const diff = ((accuracy_fair - accuracy_biased)*100).toFixed(2);
  el('acc-tradeoff').innerHTML = `<span style="color:${diff>=0?'var(--g1)':'var(--r1)'}">${diff>=0?'+':''}${diff}%</span>`;

  function mrow(id, bv, av, label, desc, lb=true) {
    const imp = (Math.abs(bv-av)/Math.max(bv,.001)*100).toFixed(1);
    const ok  = lb ? av<bv : av>bv;
    const icon= lb ? (ok?'↓':'↑') : (ok?'↑':'↓');
    el(id).innerHTML = `
      <div><div class="metric-name">${label}</div><div class="metric-desc">${desc}</div></div>
      <div class="metric-vals">
        <span class="before-val">${bv.toFixed(4)}</span>
        <span class="arrow-icon">→</span>
        <span class="after-val">${av.toFixed(4)}</span>
        <span class="imp-badge">${icon} ${imp}%</span>
      </div>`;
  }

  mrow('row-gender-dpd', before.gender_dpd, after.gender_dpd, 'Demographic Parity Diff.','Gender: |P(Ŷ=1|Male) − P(Ŷ=1|Female)|');
  mrow('row-gender-eod', before.gender_eod, after.gender_eod, 'Equal Opportunity Diff.','Gender: |TPR(Male) − TPR(Female)|');
  mrow('row-gender-dir', before.gender_dir, after.gender_dir, 'Disparate Impact Ratio','Gender: min(P+)/max(P+) closer to 1 = fairer',false);
  mrow('row-region-dpd', before.region_dpd, after.region_dpd, 'Demographic Parity Diff.','Region: |P(Ŷ=1|Urban) − P(Ŷ=1|Rural)|');
  mrow('row-region-eod', before.region_eod, after.region_eod, 'Equal Opportunity Diff.','Region: |TPR(Urban) − TPR(Rural)|');
  mrow('row-region-dir', before.region_dir, after.region_dir, 'Disparate Impact Ratio','Region: min(P+)/max(P+) — 80% rule: DIR ≥ 0.80',false);

  groupedBarChart('chart-approval-group', {
    labels: ['Male','Female','Urban','Rural'],
    datasets: [
      { label:'Biased Model', borderRadius:5,
        data: [+(before.gender_approval_rates?.male*100).toFixed(2), +(before.gender_approval_rates?.female*100).toFixed(2),
               +(before.region_approval_rates?.urban*100).toFixed(2), +(before.region_approval_rates?.rural*100).toFixed(2)],
        backgroundColor:'rgba(239,68,68,.55)', borderColor:'rgba(239,68,68,.9)', borderWidth:1.5 },
      { label:'Fair Model', borderRadius:5,
        data: [+(after.gender_approval_rates?.male*100).toFixed(2), +(after.gender_approval_rates?.female*100).toFixed(2),
               +(after.region_approval_rates?.urban*100).toFixed(2), +(after.region_approval_rates?.rural*100).toFixed(2)],
        backgroundColor:'rgba(16,185,129,.55)', borderColor:'rgba(16,185,129,.9)', borderWidth:1.5 },
    ],
  });
}

/* ── SHAP ─────────────────────────────────────────────────── */
function renderSHAP() {
  const { biased, fair } = S.shap;
  horizBarChart('chart-shap-biased', biased.feature_names, biased.mean_abs_shap, 'rgba(239,68,68,.7)');
  horizBarChart('chart-shap-fair',   fair.feature_names,   fair.mean_abs_shap,   'rgba(16,185,129,.7)');
}

/* ── COMPARISON ──────────────────────────────────────────── */
function renderComparison() {
  const { before, after, accuracy_biased, accuracy_fair } = S.bias;
  const rows = [
    ['Gender DPD (↓ fairer)', before.gender_dpd, after.gender_dpd, true],
    ['Gender EOD (↓ fairer)', before.gender_eod, after.gender_eod, true],
    ['Gender DIR (↑ fairer)', before.gender_dir, after.gender_dir, false],
    ['Region DPD (↓ fairer)', before.region_dpd, after.region_dpd, true],
    ['Region EOD (↓ fairer)', before.region_eod, after.region_eod, true],
    ['Region DIR (↑ fairer)', before.region_dir, after.region_dir, false],
    ['Model Accuracy',        accuracy_biased,   accuracy_fair,    false],
  ];
  el('compare-tbody').innerHTML = rows.map(([name,bv,av,lb]) => {
    const imp = ((Math.abs(av-bv)/Math.max(bv,.001))*100).toFixed(1);
    const ok  = lb ? av<bv : av>bv;
    const icon = ok ? (lb?'↓ ':'↑ ') : (lb?'↑ ':'↓ ');
    return `<tr>
      <td style="text-align:left;font-weight:500">${name}</td>
      <td style="color:var(--r1);font-weight:700">${bv.toFixed(4)}</td>
      <td class="${ok?'improved':'worse'}">${av.toFixed(4)}</td>
      <td class="${ok?'improved':'worse'}">${ok?'✓ ':'✗ '}${icon}${imp}%</td>
    </tr>`;
  }).join('');

  groupedBarChart('chart-compare-full', {
    labels: ['Gender DPD','Gender EOD','Gender DIR','Region DPD','Region EOD','Region DIR'],
    datasets: [
      { label:'Biased Model', borderRadius:5,
        data:[before.gender_dpd,before.gender_eod,before.gender_dir,before.region_dpd,before.region_eod,before.region_dir],
        backgroundColor:'rgba(239,68,68,.5)', borderColor:'rgba(239,68,68,.9)', borderWidth:2 },
      { label:'Fair Model', borderRadius:5,
        data:[after.gender_dpd,after.gender_eod,after.gender_dir,after.region_dpd,after.region_eod,after.region_dir],
        backgroundColor:'rgba(16,185,129,.5)', borderColor:'rgba(16,185,129,.9)', borderWidth:2 },
    ],
  });
}

/* ══════════════════════════════════════════════════════════════
   WHAT-IF
══════════════════════════════════════════════════════════════ */
function initWhatIf() {
  const EDU = ['No Degree','High School','Some College',"Bachelor's",'Graduate'];
  function updateLabels() {
    el('wi-income-val').textContent    = '$' + fmt(+get('wi-income').value);
    el('wi-age-val').textContent       = get('wi-age').value + ' yrs';
    el('wi-education-val').textContent = EDU[+get('wi-education').value];
  }
  let deb;
  function onChange() { updateLabels(); clearTimeout(deb); deb = setTimeout(doWhatIf, 220); }
  ['wi-income','wi-age','wi-education'].forEach(id => get(id)?.addEventListener('input', onChange));
  ['wi-gender','wi-region'].forEach(id => get(id)?.addEventListener('change', onChange));
  updateLabels();
  doWhatIf();
}

async function doWhatIf() {
  if (!S.ready) return;
  try {
    const data = await fetch('/api/whatif', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(getWiInputs()),
    }).then(r => r.json());
    if (data.error) return;
    showVerdict(data);
  } catch (_) {}
}

window.runDetailedPredict = async function() {
  if (!S.ready) { alert('Please wait — system is still initializing.'); return; }
  const btn = get('btn-detailed');
  btn.disabled = true; btn.textContent = '⏳ Computing SHAP…';
  try {
    const data = await fetch('/api/predict', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(getWiInputs()),
    }).then(r => r.json());
    if (data.error) throw new Error(data.error);
    renderLocalSHAP(data);
  } catch (e) { alert('Error: ' + e.message); }
  finally { btn.disabled = false; btn.textContent = '🔬 Explain This Decision (SHAP)'; }
};

function getWiInputs() {
  return {
    income:    +get('wi-income').value,
    age:       +get('wi-age').value,
    education: +get('wi-education').value,
    gender:    get('wi-gender').value,
    region:    get('wi-region').value,
  };
}

function showVerdict(data) {
  const { biased_prob, fair_prob, biased_decision, fair_decision, bias_flag } = data;
  const bA = biased_decision === 'Approved', fA = fair_decision === 'Approved';

  el('wi-biased-text').textContent = (bA?'✓ ':'✗ ') + biased_decision;
  el('wi-biased-text').className   = 'verdict-text ' + (bA?'verdict-approved':'verdict-rejected');
  el('wi-biased-prob').textContent = 'Confidence: ' + pct(biased_prob);
  el('wi-biased-bar').style.width  = (biased_prob*100).toFixed(1) + '%';
  el('wi-biased').className        = 'verdict-card ' + (bA?'approved-card':'rejected-card');

  el('wi-fair-text').textContent = (fA?'✓ ':'✗ ') + fair_decision;
  el('wi-fair-text').className   = 'verdict-text ' + (fA?'verdict-approved':'verdict-rejected');
  el('wi-fair-prob').textContent = 'Confidence: ' + pct(fair_prob);
  el('wi-fair-bar').style.width  = (fair_prob*100).toFixed(1) + '%';
  el('wi-fair').className        = 'verdict-card ' + (fA?'approved-card':'rejected-card');

  const alertEl = el('wi-bias-flag');
  alertEl.className = 'bias-alert';
  if (bias_flag) {
    alertEl.classList.add('warn');
    el('wi-bias-text').innerHTML = '<strong>⚠️ Bias Detected!</strong> The biased model rejects this applicant, but the fair model would approve. Protected attributes (gender/region) are unfairly influencing the outcome.';
  } else if (bA && fA) {
    alertEl.classList.add('ok');
    el('wi-bias-text').innerHTML = '✅ <strong>No Bias Detected</strong> — Both models agree this applicant should be <strong>Approved</strong>.';
  } else if (!bA && !fA) {
    alertEl.classList.add('ok');
    el('wi-bias-text').innerHTML = '📊 <strong>Consistent Decision</strong> — Both models reject. Try increasing income or education level.';
  } else {
    alertEl.classList.add('warn');
    el('wi-bias-text').innerHTML = '⚠️ <strong>Models Diverge</strong> — Fair model rejects but biased model approves.';
  }
}

function renderLocalSHAP(data) {
  el('shap-local-container').style.display = 'block';
  function drawWaterfall(contribs, waterId) {
    const maxAbs = Math.max(...contribs.map(c => Math.abs(c.shap_value)), .001);
    el(waterId).innerHTML = contribs.map(c => {
      const pctW = (Math.abs(c.shap_value)/maxAbs*45).toFixed(1);
      const pos  = c.shap_value > 0;
      return `<div class="shap-row">
        <div class="shap-feat">${c.feature}</div>
        <div class="shap-track">
          <div class="shap-track-mid"></div>
          <div class="shap-fill ${pos?'shap-fill-pos':'shap-fill-neg'}" style="width:${pctW}%"></div>
        </div>
        <div class="shap-num ${pos?'shap-pos-num':'shap-neg-num'}">${pos?'+':''}${c.shap_value.toFixed(4)}</div>
      </div>`;
    }).join('');
  }
  const bA = data.biased.approved, fA = data.fair.approved;
  el('shap-biased-verdict').textContent = bA ? '✓ Approved' : '✗ Rejected';
  el('shap-biased-verdict').className   = 'shap-verdict ' + (bA?'verdict-approved':'verdict-rejected');
  el('shap-biased-prob').textContent    = 'Approval probability: ' + pct(data.biased.prediction_prob);
  drawWaterfall(data.biased.contributions, 'shap-biased-waterfall');

  el('shap-fair-verdict').textContent = fA ? '✓ Approved' : '✗ Rejected';
  el('shap-fair-verdict').className   = 'shap-verdict ' + (fA?'verdict-approved':'verdict-rejected');
  el('shap-fair-prob').textContent    = 'Approval probability: ' + pct(data.fair.prediction_prob);
  drawWaterfall(data.fair.contributions, 'shap-fair-waterfall');

  el('shap-local-container').scrollIntoView({behavior:'smooth', block:'start'});
}

/* ══════════════════════════════════════════════════════════════
   CHART HELPERS
══════════════════════════════════════════════════════════════ */
const GRID = 'rgba(255,255,255,.05)';
function killChart(id) { if (S.charts[id]) { S.charts[id].destroy(); delete S.charts[id]; } }

function barChart(id, { labels, values, colors, label }) {
  killChart(id);
  const ctx = get(id)?.getContext('2d'); if (!ctx) return;
  S.charts[id] = new Chart(ctx, {
    type:'bar',
    data:{ labels, datasets:[{ label, data:values, backgroundColor:colors, borderRadius:7, borderSkipped:false }] },
    options:{ responsive:true, plugins:{legend:{display:false}},
      scales:{ x:{grid:{color:GRID},ticks:{color:'#64748b'}}, y:{grid:{color:GRID},ticks:{color:'#64748b'},beginAtZero:true} } },
  });
}
function horizBarChart(id, labels, values, color) {
  killChart(id);
  const ctx = get(id)?.getContext('2d'); if (!ctx) return;
  const paired = labels.map((l,i)=>({l,v:values[i]})).sort((a,b)=>b.v-a.v);
  S.charts[id] = new Chart(ctx, {
    type:'bar',
    data:{ labels:paired.map(p=>p.l), datasets:[{label:'Mean |SHAP|', data:paired.map(p=>p.v), backgroundColor:color, borderRadius:5}] },
    options:{ indexAxis:'y', responsive:true, plugins:{legend:{display:false}},
      scales:{ x:{grid:{color:GRID},ticks:{color:'#64748b'}}, y:{grid:{color:GRID},ticks:{color:'#64748b'}} } },
  });
}
function groupedBarChart(id, { labels, datasets }) {
  killChart(id);
  const ctx = get(id)?.getContext('2d'); if (!ctx) return;
  S.charts[id] = new Chart(ctx, {
    type:'bar', data:{labels, datasets},
    options:{ responsive:true, plugins:{legend:{labels:{color:'#94a3b8',boxWidth:12}}},
      scales:{ x:{grid:{color:GRID},ticks:{color:'#64748b'}}, y:{grid:{color:GRID},ticks:{color:'#64748b'}} } },
  });
}

/* ══════════════════════════════════════════════════════════════
   ANIMATED COUNTER
══════════════════════════════════════════════════════════════ */
function animCount(id, target, decimals, suffix='', prefix='') {
  const node = el(id); if (!node) return;
  const t0 = performance.now(), dur = 1200;
  function step(t) {
    const p = Math.min((t-t0)/dur, 1), e = 1-Math.pow(1-p,3), v = target*e;
    node.textContent = prefix + (decimals===0 ? fmt(Math.round(v)) : v.toFixed(decimals)) + suffix;
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ── Utility ────────────────────────────────────────────────── */
function el(id)  { return document.getElementById(id); }
function get(id) { return document.getElementById(id); }
function fmt(n)  { return (+n).toLocaleString(); }
function pct(v)  { return (v*100).toFixed(2)+'%'; }
function cap(s)  { return s.charAt(0).toUpperCase()+s.slice(1); }

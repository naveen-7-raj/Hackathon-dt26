/* ============================================================
   app.js – FairLens AI Dashboard Logic
   Chart.js visualizations + API calls + What-If real-time
   ============================================================ */

'use strict';

// ── Chart defaults ────────────────────────────────────────────
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
Chart.defaults.font.size = 12;

// ── State ─────────────────────────────────────────────────────
const state = {
  initialized: false,
  overview: null,
  bias: null,
  shapGlobal: null,
  charts: {},
};

// ── Nav ───────────────────────────────────────────────────────
document.querySelectorAll('.nav-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const target = tab.dataset.section;
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(target).classList.add('active');
  });
});

// ── Initialize system ─────────────────────────────────────────
const initBtn = document.getElementById('btn-init');
const overlay = document.getElementById('loading-overlay');
const loadingStep = document.getElementById('loading-step');

const LOADING_STEPS = [
  'Generating 100,000 synthetic records…',
  'Applying intentional bias patterns…',
  'Training biased Logistic Regression…',
  'Training fairness-reweighted model…',
  'Computing SHAP explanations…',
  'Calculating bias metrics…',
  'Finalizing dashboard…',
];

async function initializeSystem() {
  initBtn.disabled = true;
  overlay.classList.remove('hidden');

  let i = 0;
  const stepInterval = setInterval(() => {
    loadingStep.textContent = LOADING_STEPS[Math.min(i++, LOADING_STEPS.length - 1)];
  }, 3200);

  try {
    const res = await fetch('/api/initialize', { method: 'POST' });
    const data = await res.json();
    clearInterval(stepInterval);

    if (data.status !== 'ok') throw new Error(data.message);

    // Fetch all data in parallel
    const [overview, bias, shapGlobal] = await Promise.all([
      fetch('/api/overview').then(r => r.json()),
      fetch('/api/bias').then(r => r.json()),
      fetch('/api/shap/global').then(r => r.json()),
    ]);

    state.overview   = overview;
    state.bias       = bias;
    state.shapGlobal = shapGlobal;
    state.initialized = true;

    overlay.classList.add('hidden');
    renderAll();
    // Switch to overview tab
    document.querySelector('[data-section="section-overview"]').click();
    document.getElementById('status-badge').textContent = '✦ System Ready';
    document.getElementById('status-badge').style.background = 'rgba(16,185,129,0.2)';
    document.getElementById('status-badge').style.color = '#10b981';
    document.getElementById('status-badge').style.borderColor = 'rgba(16,185,129,0.4)';
  } catch (err) {
    clearInterval(stepInterval);
    overlay.classList.add('hidden');
    alert('Initialization failed: ' + err.message);
    initBtn.disabled = false;
  }
}

initBtn.addEventListener('click', initializeSystem);

// ── Render all sections ───────────────────────────────────────
function renderAll() {
  renderOverview();
  renderBiasMetrics();
  renderSHAP();
  renderComparison();
  initWhatIf();
}

// ── OVERVIEW section ──────────────────────────────────────────
function renderOverview() {
  const ov = state.overview;
  if (!ov) return;

  // Stat cards
  setInner('stat-total',    fmt(ov.n_samples));
  setInner('stat-approval', pct(ov.approval_rate));
  setInner('stat-income',   '$' + fmt(Math.round(ov.income_median)));
  setInner('stat-age',      ov.age_mean.toFixed(1) + ' yrs');

  // Approval by gender bar chart
  renderBarChart('chart-approval-gender', {
    labels: Object.keys(ov.approval_by_gender).map(capitalize),
    values: Object.values(ov.approval_by_gender).map(v => +(v * 100).toFixed(2)),
    colors: ['rgba(139,92,246,0.7)', 'rgba(244,114,182,0.7)'],
    label: 'Approval Rate (%)',
    ylabel: '%',
  });

  // Approval by region
  renderBarChart('chart-approval-region', {
    labels: Object.keys(ov.approval_by_region).map(capitalize),
    values: Object.values(ov.approval_by_region).map(v => +(v * 100).toFixed(2)),
    colors: ['rgba(59,130,246,0.7)', 'rgba(16,185,129,0.7)'],
    label: 'Approval Rate (%)',
    ylabel: '%',
  });

  // Approval by education
  const eduLabels = ['No Degree', 'High School', 'Some College', "Bachelor's", 'Graduate'];
  const eduVals   = Object.values(ov.approval_by_education).map(v => +(v * 100).toFixed(2));
  renderBarChart('chart-approval-education', {
    labels: eduLabels,
    values: eduVals,
    colors: eduVals.map(v => `rgba(${139 + (v - 40) * 2},${92 + v * 2},${246 - v},0.75)`),
    label: 'Approval Rate (%)',
    ylabel: '%',
    horizontal: false,
  });
}

// ── BIAS METRICS section ──────────────────────────────────────
function renderBiasMetrics() {
  const { before, after, accuracy_biased, accuracy_fair } = state.bias;
  if (!before) return;

  // Accuracy cards
  setInner('acc-biased', pct(accuracy_biased));
  setInner('acc-fair',   pct(accuracy_fair));
  const accDiff = (accuracy_fair - accuracy_biased) * 100;
  setInner('acc-tradeoff', (accDiff >= 0 ? '+' : '') + accDiff.toFixed(2) + '% accuracy change for fairness');

  function renderMetricBlock(prefix, beforeVal, afterVal, label, desc, lowerIsBetter = true) {
    const imp = lowerIsBetter
      ? ((beforeVal - afterVal) / Math.max(beforeVal, 0.001) * 100).toFixed(1)
      : ((afterVal - beforeVal) / Math.max(beforeVal, 0.001) * 100).toFixed(1);

    const row = document.getElementById(`row-${prefix}`);
    if (!row) return;
    row.innerHTML = `
      <div>
        <div class="metric-name">${label}</div>
        <div class="metric-desc">${desc}</div>
      </div>
      <div class="metric-values">
        <span class="metric-before" title="Biased model">${beforeVal.toFixed(4)}</span>
        <span class="metric-arrow">→</span>
        <span class="metric-after" title="Fair model">${afterVal.toFixed(4)}</span>
        <span class="metric-improvement">${lowerIsBetter ? '↓' : '↑'} ${imp}%</span>
      </div>
    `;
  }

  // Gender metrics
  renderMetricBlock('gender-dpd', before.gender_dpd, after.gender_dpd,
    'Demographic Parity Difference', 'Gender: |P(Ŷ=1|Male) − P(Ŷ=1|Female)|');
  renderMetricBlock('gender-eod', before.gender_eod, after.gender_eod,
    'Equal Opportunity Difference', 'Gender: |TPR(Male) − TPR(Female)|');
  renderMetricBlock('gender-dir', before.gender_dir, after.gender_dir,
    'Disparate Impact Ratio', 'Gender: min(P+) / max(P+)  — closer to 1 = fairer', false);

  // Region metrics
  renderMetricBlock('region-dpd', before.region_dpd, after.region_dpd,
    'Demographic Parity Difference', 'Region: |P(Ŷ=1|Urban) − P(Ŷ=1|Rural)|');
  renderMetricBlock('region-eod', before.region_eod, after.region_eod,
    'Equal Opportunity Difference', 'Region: |TPR(Urban) − TPR(Rural)|');
  renderMetricBlock('region-dir', before.region_dir, after.region_dir,
    'Disparate Impact Ratio', 'Region: min(P+) / max(P+)  — 0.8 threshold (80% rule)', false);

  // Approval rate by group charts (before vs after)
  renderGroupedBarChart('chart-approval-group', {
    labels: ['Male', 'Female', 'Urban', 'Rural'],
    datasets: [
      {
        label: 'Biased Model',
        data: [
          +(before.gender_approval_rates?.male  * 100).toFixed(2),
          +(before.gender_approval_rates?.female* 100).toFixed(2),
          +(before.region_approval_rates?.urban * 100).toFixed(2),
          +(before.region_approval_rates?.rural * 100).toFixed(2),
        ],
        backgroundColor: 'rgba(239,68,68,0.6)',
        borderColor: 'rgba(239,68,68,0.9)',
        borderWidth: 1.5,
      },
      {
        label: 'Fair Model',
        data: [
          +(after.gender_approval_rates?.male  * 100).toFixed(2),
          +(after.gender_approval_rates?.female* 100).toFixed(2),
          +(after.region_approval_rates?.urban * 100).toFixed(2),
          +(after.region_approval_rates?.rural * 100).toFixed(2),
        ],
        backgroundColor: 'rgba(16,185,129,0.6)',
        borderColor: 'rgba(16,185,129,0.9)',
        borderWidth: 1.5,
      },
    ],
  });
}

// ── SHAP section ──────────────────────────────────────────────
function renderSHAP() {
  const { biased, fair } = state.shapGlobal;
  if (!biased) return;

  renderHorizBarChart('chart-shap-biased', {
    labels: biased.feature_names,
    values: biased.mean_abs_shap,
    color: 'rgba(239,68,68,0.7)',
    label: 'Mean |SHAP|',
  });
  renderHorizBarChart('chart-shap-fair', {
    labels: fair.feature_names,
    values: fair.mean_abs_shap,
    color: 'rgba(16,185,129,0.7)',
    label: 'Mean |SHAP|',
  });
}

// ── COMPARISON section ────────────────────────────────────────
function renderComparison() {
  const { before, after, accuracy_biased, accuracy_fair } = state.bias;
  if (!before) return;

  const tbody = document.getElementById('compare-tbody');
  const rows = [
    ['Gender DPD (↓ better)', before.gender_dpd, after.gender_dpd, true],
    ['Gender EOD (↓ better)', before.gender_eod, after.gender_eod, true],
    ['Gender DIR (↑ better)', before.gender_dir, after.gender_dir, false],
    ['Region DPD (↓ better)', before.region_dpd, after.region_dpd, true],
    ['Region EOD (↓ better)', before.region_eod, after.region_eod, true],
    ['Region DIR (↑ better)', before.region_dir, after.region_dir, false],
    ['Accuracy', accuracy_biased, accuracy_fair, false],
  ];

  tbody.innerHTML = rows.map(([name, bv, av, lowerIsBetter]) => {
    const improved = lowerIsBetter ? av < bv : av > bv;
    const cls = improved ? 'improved' : 'worse';
    const icon = improved ? '✓' : '✗';
    const impPct = ((Math.abs(av - bv) / Math.max(bv, 0.001)) * 100).toFixed(1);
    return `<tr>
      <td style="text-align:left">${name}</td>
      <td style="color:var(--red)">${bv.toFixed(4)}</td>
      <td class="${cls}">${av.toFixed(4)}</td>
      <td class="${cls}">${icon} ${improved ? (lowerIsBetter ? '↓' : '↑') : (lowerIsBetter ? '↑' : '↓')} ${impPct}%</td>
    </tr>`;
  }).join('');

  // Radar-style bar chart for DPD comparison
  renderGroupedBarChart('chart-compare-full', {
    labels: ['Gender DPD', 'Gender EOD', 'Gender DIR', 'Region DPD', 'Region EOD', 'Region DIR'],
    datasets: [
      {
        label: 'Biased Model',
        data: [before.gender_dpd, before.gender_eod, before.gender_dir,
               before.region_dpd, before.region_eod, before.region_dir],
        backgroundColor: 'rgba(239,68,68,0.5)',
        borderColor: 'rgba(239,68,68,0.9)', borderWidth: 2,
      },
      {
        label: 'Fair Model',
        data: [after.gender_dpd, after.gender_eod, after.gender_dir,
               after.region_dpd, after.region_eod, after.region_dir],
        backgroundColor: 'rgba(16,185,129,0.5)',
        borderColor: 'rgba(16,185,129,0.9)', borderWidth: 2,
      },
    ],
  });
}

// ── WHAT-IF simulator ─────────────────────────────────────────
function initWhatIf() {
  const sliders = {
    income:    document.getElementById('wi-income'),
    age:       document.getElementById('wi-age'),
    education: document.getElementById('wi-education'),
  };
  const selects = {
    gender: document.getElementById('wi-gender'),
    region: document.getElementById('wi-region'),
  };
  const displays = {
    income:    document.getElementById('wi-income-val'),
    age:       document.getElementById('wi-age-val'),
    education: document.getElementById('wi-education-val'),
  };

  const EDU_LABELS = ['No Degree', 'High School', 'Some College', "Bachelor's", 'Graduate'];

  function updateDisplays() {
    displays.income.textContent    = '$' + fmt(+sliders.income.value);
    displays.age.textContent       = sliders.age.value + ' yrs';
    displays.education.textContent = EDU_LABELS[+sliders.education.value];
  }

  let debounce;
  function onChange() {
    updateDisplays();
    clearTimeout(debounce);
    debounce = setTimeout(runWhatIf, 220);
  }

  Object.values(sliders).forEach(el => el.addEventListener('input', onChange));
  Object.values(selects).forEach(el => el.addEventListener('change', onChange));

  updateDisplays();
  runWhatIf();

  async function runWhatIf() {
    if (!state.initialized) return;
    const body = {
      income:    +sliders.income.value,
      age:       +sliders.age.value,
      education: +sliders.education.value,
      gender:    selects.gender.value,
      region:    selects.region.value,
    };
    try {
      const res  = await fetch('/api/whatif', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      displayWhatIfResult(data);
    } catch (_) {}
  }

  window.runDetailedPredict = async function() {
    if (!state.initialized) return;
    const btn = document.getElementById('btn-detailed');
    btn.disabled = true; btn.textContent = 'Analyzing…';
    const body = {
      income:    +sliders.income.value,
      age:       +sliders.age.value,
      education: +sliders.education.value,
      gender:    selects.gender.value,
      region:    selects.region.value,
    };
    try {
      const res  = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      renderLocalSHAP(data);
    } catch (e) { alert('Error: ' + e.message); }
    finally { btn.disabled = false; btn.textContent = '🔬 Explain This Decision'; }
  };
}

function displayWhatIfResult(data) {
  const { biased_prob, fair_prob, biased_decision, fair_decision, bias_flag } = data;

  const biasedCard = document.getElementById('wi-biased');
  const fairCard   = document.getElementById('wi-fair');

  biasedCard.querySelector('.prediction-verdict').className =
    'prediction-verdict ' + (biased_decision === 'Approved' ? 'verdict-approved' : 'verdict-rejected');
  biasedCard.querySelector('.prediction-verdict').textContent =
    (biased_decision === 'Approved' ? '✓ ' : '✗ ') + biased_decision;
  biasedCard.querySelector('.prediction-prob').textContent =
    'Confidence: ' + pct(biased_prob);
  updateProgressBar(biasedCard, biased_prob);

  fairCard.querySelector('.prediction-verdict').className =
    'prediction-verdict ' + (fair_decision === 'Approved' ? 'verdict-approved' : 'verdict-rejected');
  fairCard.querySelector('.prediction-verdict').textContent =
    (fair_decision === 'Approved' ? '✓ ' : '✗ ') + fair_decision;
  fairCard.querySelector('.prediction-prob').textContent =
    'Confidence: ' + pct(fair_prob);
  updateProgressBar(fairCard, fair_prob);

  const banner = document.getElementById('wi-bias-flag');
  if (bias_flag) {
    banner.classList.remove('hidden', 'fair');
    banner.innerHTML = '⚠️ <strong>Bias Detected!</strong> Biased model rejects but fair model would approve this applicant.';
  } else if (biased_decision === 'Approved' && fair_decision === 'Approved') {
    banner.classList.remove('hidden');
    banner.classList.add('fair');
    banner.innerHTML = '✅ Both models agree: <strong>Approved</strong>. No bias detected for this profile.';
  } else if (biased_decision === 'Rejected' && fair_decision === 'Rejected') {
    banner.classList.remove('hidden');
    banner.classList.add('fair');
    banner.innerHTML = '📊 Both models agree: <strong>Rejected</strong>. Decision appears consistent.';
  } else {
    banner.classList.add('hidden');
  }
}

function updateProgressBar(card, prob) {
  const fill = card.querySelector('.progress-fill');
  if (fill) fill.style.width = (prob * 100).toFixed(1) + '%';
}

function renderLocalSHAP(data) {
  const container = document.getElementById('shap-local-container');
  container.style.display = 'block';

  const biased = data.biased;
  const fair   = data.fair;

  function buildWaterfall(contribs, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;
    const maxAbs = Math.max(...contribs.map(c => Math.abs(c.shap_value)), 0.01);
    el.innerHTML = contribs.map(c => {
      const pct = (Math.abs(c.shap_value) / maxAbs * 45).toFixed(1);
      const isPos = c.shap_value > 0;
      return `
        <div class="shap-item">
          <div class="shap-feature">${c.feature}</div>
          <div class="shap-bar-wrap">
            <div class="shap-center"></div>
            <div class="shap-bar ${isPos ? 'shap-pos' : 'shap-neg'}" style="width:${pct}%"></div>
          </div>
          <div class="shap-value ${isPos ? 'shap-positive' : 'shap-negative'}">
            ${isPos ? '+' : ''}${c.shap_value.toFixed(4)}
          </div>
        </div>`;
    }).join('');
  }

  buildWaterfall(biased.contributions, 'shap-biased-waterfall');
  buildWaterfall(fair.contributions,   'shap-fair-waterfall');

  document.getElementById('shap-biased-verdict').textContent =
    biased.approved ? '✓ Approved' : '✗ Rejected';
  document.getElementById('shap-biased-verdict').className =
    biased.approved ? 'verdict-approved' : 'verdict-rejected';
  document.getElementById('shap-biased-prob').textContent =
    'Approval probability: ' + pct(biased.prediction_prob);

  document.getElementById('shap-fair-verdict').textContent =
    fair.approved ? '✓ Approved' : '✗ Rejected';
  document.getElementById('shap-fair-verdict').className =
    fair.approved ? 'verdict-approved' : 'verdict-rejected';
  document.getElementById('shap-fair-prob').textContent =
    'Approval probability: ' + pct(fair.prediction_prob);
}

// ── Chart helpers ─────────────────────────────────────────────
const GRID_COLOR = 'rgba(255,255,255,0.06)';

function destroyChart(id) {
  if (state.charts[id]) { state.charts[id].destroy(); delete state.charts[id]; }
}

function renderBarChart(id, { labels, values, colors, label, ylabel, horizontal = false }) {
  destroyChart(id);
  const ctx = document.getElementById(id)?.getContext('2d');
  if (!ctx) return;

  state.charts[id] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label, data: values, backgroundColor: colors, borderRadius: 6, borderSkipped: false }],
    },
    options: {
      indexAxis: horizontal ? 'y' : 'x',
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: GRID_COLOR }, ticks: { color: '#94a3b8' } },
        y: { grid: { color: GRID_COLOR }, ticks: { color: '#94a3b8' }, beginAtZero: true },
      },
    },
  });
}

function renderHorizBarChart(id, { labels, values, color, label }) {
  destroyChart(id);
  const ctx = document.getElementById(id)?.getContext('2d');
  if (!ctx) return;

  const sorted = labels.map((l, i) => ({ l, v: values[i] })).sort((a, b) => b.v - a.v);

  state.charts[id] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map(s => s.l),
      datasets: [{ label, data: sorted.map(s => s.v), backgroundColor: color, borderRadius: 4 }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: GRID_COLOR }, ticks: { color: '#94a3b8' } },
        y: { grid: { color: GRID_COLOR }, ticks: { color: '#94a3b8' } },
      },
    },
  });
}

function renderGroupedBarChart(id, { labels, datasets }) {
  destroyChart(id);
  const ctx = document.getElementById(id)?.getContext('2d');
  if (!ctx) return;

  datasets.forEach(ds => { ds.borderRadius = 4; });

  state.charts[id] = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#94a3b8', boxWidth: 12, font: { size: 12 } } },
      },
      scales: {
        x: { grid: { color: GRID_COLOR }, ticks: { color: '#94a3b8' } },
        y: { grid: { color: GRID_COLOR }, ticks: { color: '#94a3b8' } },
      },
    },
  });
}

// ── Utility ───────────────────────────────────────────────────
function fmt(n)     { return (+n).toLocaleString(); }
function pct(v)     { return (v * 100).toFixed(2) + '%'; }
function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
function setInner(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

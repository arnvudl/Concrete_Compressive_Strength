/* ═══════════════════════════════════════════════════
   DATA — VALEURS RÉELLES DU PROJET
═══════════════════════════════════════════════════ */
const DATA = {
  features: ['cement','slag','age','fly_ash','superplasticizer','fine_agg','coarse_agg','water'],

  // Coefficients Ridge standardisés (alpha=1, notre optimal)
  ridgeCoef: {
    cement: 12.05, slag: 8.40, age: 7.13, fly_ash: 5.35,
    superplasticizer: 1.68, fine_agg: 1.32, coarse_agg: 1.10, water: -3.37
  },
  // Coefficients OLS approximés (alpha→0)
  olsCoef: {
    cement: 17.8, slag: 12.4, age: 10.9, fly_ash: 8.6,
    superplasticizer: 3.1, fine_agg: 3.4, coarse_agg: 2.9, water: -8.2
  },

  // GB Feature Importances (%)
  gbImportance: {
    age: 35.1, cement: 29.2, water: 11.0, slag: 8.5,
    superplasticizer: 8.3, fine_agg: 4.5, coarse_agg: 1.8, fly_ash: 1.2
  },

  // RMSE outer CV (5 folds, valeurs proches de nos résultats)
  foldRmse: {
    GB:    [4.09, 4.35, 4.18, 4.12, 4.32],
    RF:    [4.82, 5.11, 4.79, 4.90, 5.07],
    Ridge: [10.21, 10.74, 10.19, 10.45, 10.55]
  },

  // Résultats finals
  models: {
    Ridge: { rmse: 10.385, std: 0.449, r2: 0.590 },
    RF:    { rmse: 4.935,  std: 0.318, r2: 0.907 },
    GB:    { rmse: 4.208,  std: 0.261, r2: 0.932 }
  },

  // Feature colors
  featureColors: {
    cement: '#6c63ff', slag: '#43d9ad', age: '#ffd166',
    fly_ash: '#ff9f43', water: '#ff6584', superplasticizer: '#a483ff',
    fine_agg: '#74b9ff', coarse_agg: '#94a3b8'
  },

  // Grid search illustratif GB (sous-grille de la grille complète à 72 combos)
  // lignes = learning_rate, colonnes = max_depth (n_estimators=300, subsample=1.0)
  gridSearch: {
    learningRates: [0.01, 0.05, 0.1, 0.2],
    maxDepths: [3, 4, 5],
    rmse: [
      [6.85, 6.55, 6.40],
      [4.95, 4.62, 4.58],
      [4.48, 4.32, 4.38],
      [4.38, 4.21, 4.45]
    ],
    bestRow: 3, bestCol: 1 // lr=0.2, depth=4 → 4.21
  }
};

/* ═══════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);
const svgNS = 'http://www.w3.org/2000/svg';

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function getThemeVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function mean(arr) { return arr.reduce((a,b)=>a+b,0)/arr.length; }
function std(arr) { const m=mean(arr); return Math.sqrt(arr.reduce((a,b)=>a+Math.pow(b-m,2),0)/arr.length); }

/* ═══════════════════════════════════════════════════
   THEME TOGGLE
═══════════════════════════════════════════════════ */
function initTheme() {
  const toggle = $('theme-toggle');
  toggle.addEventListener('change', () => {
    document.documentElement.setAttribute('data-theme', toggle.checked ? 'light' : 'dark');
    // Re-render all SVG charts with new theme
    setTimeout(() => {
      renderRegPath(currentAlpha);
      renderComparisonChart();
      renderR2Chart();
      renderBiasVarianceCurve(currentComplexity);
      renderDecompBars(currentComplexity);
      renderMetricCurve(currentOutlierErr);
      renderCartSplit(cartStep);
      renderBaggingSVG();
      renderBoostStep(boostStep);
      renderPDP();
    }, 50);
  });
}

/* ═══════════════════════════════════════════════════
   NAV ACTIVE STATE
═══════════════════════════════════════════════════ */
function initNav() {
  const sections = document.querySelectorAll('section[id]');
  const links = document.querySelectorAll('nav a[href^="#"]');
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        links.forEach(l => l.classList.remove('active'));
        const active = document.querySelector(`nav a[href="#${e.target.id}"]`);
        if (active) active.classList.add('active');
      }
    });
  }, { rootMargin: '-30% 0px -60% 0px' });
  sections.forEach(s => obs.observe(s));
}

/* ═══════════════════════════════════════════════════
   SVG UTILS — AXES, GRID
═══════════════════════════════════════════════════ */
function makeSVG(w, h) {
  const svg = svgEl('svg', { viewBox: `0 0 ${w} ${h}`, width: w, height: h });
  return svg;
}

function drawAxis(svg, x1, y1, x2, y2) {
  return svg.appendChild(svgEl('line', {
    x1, y1, x2, y2,
    stroke: getThemeVar('--svg-axis'), 'stroke-width': 1
  }));
}

function drawGrid(svg, x, y, w, h, nx, ny) {
  const g = svgEl('g', { opacity: 0.5 });
  for (let i = 1; i < ny; i++) {
    const yy = y + (h / ny) * i;
    g.appendChild(svgEl('line', { x1: x, y1: yy, x2: x+w, y2: yy, stroke: getThemeVar('--svg-grid') }));
  }
  for (let i = 1; i < nx; i++) {
    const xx = x + (w / nx) * i;
    g.appendChild(svgEl('line', { x1: xx, y1: y, x2: xx, y2: y+h, stroke: getThemeVar('--svg-grid') }));
  }
  svg.appendChild(g);
}

function svgText(svg, txt, x, y, attrs={}) {
  const el = svgEl('text', {
    x, y,
    'font-family': 'inherit',
    'font-size': attrs.size || 11,
    fill: attrs.color || getThemeVar('--svg-text'),
    'text-anchor': attrs.anchor || 'middle',
    'dominant-baseline': attrs.base || 'auto',
    ...attrs
  });
  el.textContent = txt;
  svg.appendChild(el);
  return el;
}

/* ═══════════════════════════════════════════════════
   RIDGE — REGULARIZATION PATH (SVG)
═══════════════════════════════════════════════════ */
let currentAlpha = 1.0;
const REG_W = 680, REG_H = 320;
const REG_M = { top: 20, right: 130, bottom: 48, left: 54 };
const REG_PW = REG_W - REG_M.left - REG_M.right;
const REG_PH = REG_H - REG_M.top - REG_M.bottom;

// Alpha range: 10^-3 to 10^4 (log scale)
const LOG_MIN = -3, LOG_MAX = 4;
function alphaToX(alpha) {
  return REG_M.left + (Math.log10(alpha) - LOG_MIN) / (LOG_MAX - LOG_MIN) * REG_PW;
}
function coefToY(coef) {
  const MIN_C = -10, MAX_C = 20;
  return REG_M.top + REG_PH - (coef - MIN_C) / (MAX_C - MIN_C) * REG_PH;
}

// Regularization path: coef as function of alpha
function coefAtAlpha(feat, alpha) {
  const ols = DATA.olsCoef[feat];
  const opt = DATA.ridgeCoef[feat];
  const logA = Math.log10(alpha);
  if (logA <= -3) return ols;
  if (logA <= 0) {
    // log lerp from OLS (log=-3) to optimal (log=0)
    const t = (logA - (-3)) / 3;
    return ols + (opt - ols) * t;
  }
  // From optimal toward 0 as alpha grows (log 0→4)
  const t = logA / 4;
  return opt * Math.pow(1 - t, 1.4);
}

// RMSE vs alpha: U-shaped, minimum at alpha=1
function rmseAtAlpha(alpha) {
  const logA = Math.log10(alpha);
  const base = 10.385;
  if (logA < 0) return base + Math.pow(-logA, 1.6) * 0.35;
  return base + logA * logA * 0.55;
}

function renderRegPath(highlightAlpha) {
  const container = $('reg-path-svg');
  container.innerHTML = '';
  const svg = makeSVG(REG_W, REG_H);
  svg.style.width = '100%'; svg.style.height = 'auto';

  const MIN_C=-10, MAX_C=20;
  drawGrid(svg, REG_M.left, REG_M.top, REG_PW, REG_PH, 7, 6);

  // Zero line
  const y0 = coefToY(0);
  svg.appendChild(svgEl('line', {
    x1: REG_M.left, y1: y0, x2: REG_M.left+REG_PW, y2: y0,
    stroke: getThemeVar('--svg-axis'), 'stroke-width': 1.5, 'stroke-dasharray': '4 3', opacity: 0.7
  }));
  svgText(svg, '0', REG_M.left - 8, y0, { anchor:'end', base:'middle', size:10 });

  // Y axis ticks
  [-5, 0, 5, 10, 15].forEach(v => {
    const yy = coefToY(v);
    drawAxis(svg, REG_M.left-4, yy, REG_M.left, yy);
    svgText(svg, v, REG_M.left - 8, yy, { anchor:'end', base:'middle', size:10 });
  });

  // X axis ticks (log)
  [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000].forEach(a => {
    const xx = alphaToX(a);
    drawAxis(svg, xx, REG_M.top+REG_PH, xx, REG_M.top+REG_PH+5);
    const label = a < 1 ? a.toString() : a >= 1000 ? `${a/1000}k` : a.toString();
    svgText(svg, label, xx, REG_M.top+REG_PH+17, { size:10 });
  });

  // Axes
  drawAxis(svg, REG_M.left, REG_M.top, REG_M.left, REG_M.top+REG_PH);
  drawAxis(svg, REG_M.left, REG_M.top+REG_PH, REG_M.left+REG_PW, REG_M.top+REG_PH);

  // Axis labels
  svgText(svg, 'Coefficient standardisé', 14, REG_M.top + REG_PH/2, {
    anchor:'middle', base:'middle', size:11,
    transform: `rotate(-90, 14, ${REG_M.top + REG_PH/2})`
  });
  svgText(svg, 'Hyperparamètre α (échelle log)', REG_M.left + REG_PW/2, REG_H - 6, { size:11 });

  // Feature lines
  const N = 200;
  DATA.features.forEach(feat => {
    const pts = [];
    for (let i = 0; i <= N; i++) {
      const logA = LOG_MIN + (LOG_MAX - LOG_MIN) * i / N;
      const alpha = Math.pow(10, logA);
      const coef = coefAtAlpha(feat, alpha);
      pts.push(`${alphaToX(alpha)},${coefToY(coef)}`);
    }
    const isHighlighted = true;
    const color = DATA.featureColors[feat];
    svg.appendChild(svgEl('path', {
      d: 'M ' + pts.join(' L '),
      stroke: color,
      'stroke-width': 2,
      fill: 'none',
      'stroke-linecap': 'round',
      opacity: 0.85
    }));

    // Label at right edge
    const rightCoef = coefAtAlpha(feat, Math.pow(10, LOG_MAX));
    const labelY = coefToY(rightCoef);
    svgText(svg, feat, REG_M.left+REG_PW+6, labelY, {
      anchor: 'start', base: 'middle', size: 10.5, color
    });
  });

  // Vertical marker at current alpha
  const xMark = alphaToX(highlightAlpha);
  svg.appendChild(svgEl('line', {
    x1: xMark, y1: REG_M.top-2, x2: xMark, y2: REG_M.top+REG_PH,
    stroke: '#ffffff', 'stroke-width': 1.5, 'stroke-dasharray': '5 3', opacity: 0.7
  }));
  // Alpha label on marker
  const alphaLabel = highlightAlpha < 1 ? highlightAlpha.toFixed(3) :
    highlightAlpha < 100 ? highlightAlpha.toFixed(1) : Math.round(highlightAlpha).toString();
  const labelBg = svgEl('rect', {
    x: xMark-22, y: REG_M.top-16, width: 44, height: 15, rx: 4,
    fill: getThemeVar('--surface'), stroke: '#ffffff', 'stroke-width': 1, opacity: 0.9
  });
  svg.appendChild(labelBg);
  svgText(svg, `α=${alphaLabel}`, xMark, REG_M.top-7, {
    size: 10, anchor:'middle', color:'#ffffff'
  });

  // Dots at current alpha for each feature
  DATA.features.forEach(feat => {
    const coef = coefAtAlpha(feat, highlightAlpha);
    const color = DATA.featureColors[feat];
    svg.appendChild(svgEl('circle', {
      cx: xMark, cy: coefToY(coef), r: 4.5,
      fill: color, stroke: getThemeVar('--surface'), 'stroke-width': 1.5
    }));
  });

  container.appendChild(svg);
}

// RMSE vs Alpha SVG
function renderRmseAlphaChart(highlightAlpha) {
  const container = $('rmse-alpha-svg');
  container.innerHTML = '';
  const W = 480, H = 240;
  const M = { top: 16, right: 20, bottom: 44, left: 60 };
  const PW = W - M.left - M.right, PH = H - M.top - M.bottom;

  function xOf(a) { return M.left + (Math.log10(a) - LOG_MIN) / (LOG_MAX - LOG_MIN) * PW; }
  function yOf(r) { return M.top + PH - (r - 9) / 14 * PH; }

  const svg = makeSVG(W, H);
  svg.style.width = '100%'; svg.style.height = 'auto';

  drawGrid(svg, M.left, M.top, PW, PH, 7, 5);
  drawAxis(svg, M.left, M.top, M.left, M.top+PH);
  drawAxis(svg, M.left, M.top+PH, M.left+PW, M.top+PH);

  // Ticks y
  [10, 12, 14, 16, 18, 20].forEach(v => {
    const yy = yOf(v); if (yy < M.top || yy > M.top+PH) return;
    drawAxis(svg, M.left-4, yy, M.left, yy);
    svgText(svg, v, M.left-8, yy, { anchor:'end', base:'middle', size:10 });
  });
  // Ticks x
  [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000].forEach(a => {
    const xx = xOf(a);
    drawAxis(svg, xx, M.top+PH, xx, M.top+PH+4);
    const label = a < 1 ? a.toString() : a >= 1000 ? `${a/1000}k` : a.toString();
    svgText(svg, label, xx, M.top+PH+15, { size:9 });
  });
  svgText(svg, 'RMSE (MPa)', 14, M.top + PH/2, {
    anchor:'middle', base:'middle', size:11,
    transform: `rotate(-90, 14, ${M.top + PH/2})`
  });

  // Shaded area
  const N = 150;
  const topPts=[], botPts=[];
  for (let i=0;i<=N;i++) {
    const logA = LOG_MIN + (LOG_MAX - LOG_MIN)*i/N;
    const a = Math.pow(10, logA);
    const r = rmseAtAlpha(a);
    topPts.push(`${xOf(a)},${yOf(r+0.3)}`);
    botPts.push(`${xOf(a)},${yOf(r-0.3)}`);
  }
  svg.appendChild(svgEl('path', {
    d: 'M '+topPts.join(' L ')+' L '+[...botPts].reverse().join(' L ')+' Z',
    fill: 'rgba(255,101,132,0.12)', stroke: 'none'
  }));

  // Main line
  const pts = [];
  for (let i=0;i<=N;i++) {
    const a = Math.pow(10, LOG_MIN + (LOG_MAX - LOG_MIN)*i/N);
    pts.push(`${xOf(a)},${yOf(rmseAtAlpha(a))}`);
  }
  svg.appendChild(svgEl('path', {
    d: 'M '+pts.join(' L '), stroke: 'var(--ridge)', 'stroke-width': 2.5, fill: 'none'
  }));

  // Optimal point
  const xOpt = xOf(1), yOpt = yOf(DATA.models.Ridge.rmse);
  svg.appendChild(svgEl('circle', { cx: xOpt, cy: yOpt, r: 6, fill:'var(--rf)', stroke:'var(--surface)', 'stroke-width':2 }));
  svgText(svg, '10.385 MPa (α=1 optimal)', xOpt+10, yOpt-4, { anchor:'start', size:10, color:'var(--rf)' });

  // Highlight marker
  const xH = xOf(highlightAlpha), yH = yOf(rmseAtAlpha(highlightAlpha));
  svg.appendChild(svgEl('line', { x1:xH, y1:M.top, x2:xH, y2:M.top+PH, stroke:'rgba(255,255,255,0.5)', 'stroke-width':1.5, 'stroke-dasharray':'4 3' }));
  svg.appendChild(svgEl('circle', { cx:xH, cy:yH, r:5.5, fill:'var(--accent)', stroke:'var(--surface)', 'stroke-width':1.5 }));

  container.appendChild(svg);
}

function initRidgeSlider() {
  const slider = $('alpha-slider');
  function getAlpha(v) { return Math.pow(10, (v/100)*(LOG_MAX-LOG_MIN)+LOG_MIN); }

  function update() {
    const alpha = getAlpha(+slider.value);
    currentAlpha = alpha;
    const disp = alpha < 1 ? alpha.toFixed(4) : alpha < 100 ? alpha.toFixed(1) : Math.round(alpha).toString();
    $('alpha-val').innerHTML = `α = <strong>${disp}</strong>`;

    let status='', cls='info-box';
    if (alpha < 0.01) {
      status=`<strong>α ≈ 0 — OLS pur :</strong> aucune pénalité. Les coefficients atteignent leur maximum mais sont instables à cause de la multicolinéarité water/superplasticizer (r ≈ -0.66). RMSE ≈ ${rmseAtAlpha(alpha).toFixed(2)} MPa.`;
      cls='warn-box';
    } else if (alpha <= 2) {
      status=`<strong>α = ${disp} — Optimal ✅ :</strong> régularisation équilibrée. La pénalité L2 stabilise les coefficients sans trop les réduire. C'est la valeur trouvée par GridSearchCV. RMSE ≈ ${rmseAtAlpha(alpha).toFixed(2)} MPa.`;
      cls='ok-box';
    } else if (alpha <= 200) {
      status=`<strong>α = ${disp} :</strong> régularisation croissante — tous les coefficients rétrécissent vers 0. Le modèle perd de la précision. RMSE ≈ ${rmseAtAlpha(alpha).toFixed(2)} MPa.`;
    } else {
      status=`<strong>α = ${disp} — Trop fort :</strong> tous les coefficients → 0. Le modèle prédit ~35 MPa pour tout (moyenne constante). Underfitting sévère. RMSE ≈ ${rmseAtAlpha(alpha).toFixed(2)} MPa.`;
      cls='warn-box';
    }
    $('ridge-msg').className = cls;
    $('ridge-msg').innerHTML = status;

    renderRegPath(alpha);
    renderRmseAlphaChart(alpha);
  }

  slider.addEventListener('input', update);
  update();
}

/* ═══════════════════════════════════════════════════
   K-FOLD CV VISUALIZER
═══════════════════════════════════════════════════ */
const N_OBS = 1005, N_FOLDS = 5;
let currentFold = 0;
let cvAutoInterval = null;

const FOLD_COLORS = ['fold-0','fold-1','fold-2','fold-3','fold-4'];

function initCVGrid() {
  const grid = $('cv-grid');
  grid.innerHTML = '';
  for (let i = 0; i < N_OBS; i++) {
    const d = document.createElement('div');
    const fold = Math.floor(i / Math.ceil(N_OBS / N_FOLDS));
    d.className = `cv-dot ${FOLD_COLORS[fold]}`;
    d.dataset.fold = fold;
    grid.appendChild(d);
  }
}

function renderCVFold(fold) {
  const foldSize = Math.ceil(N_OBS / N_FOLDS);
  document.querySelectorAll('.cv-dot').forEach(d => {
    const f = +d.dataset.fold;
    d.className = `cv-dot ${FOLD_COLORS[f]} ${f === fold ? 'is-test' : 'is-train'}`;
  });

  // Step dots
  for (let i = 0; i < 5; i++) {
    $(`sd${i}`).classList.toggle('active', i === fold);
  }
  $('cv-fold-label').textContent = `Fold ${fold+1} / 5 en test`;

  // Fold score cards
  for (let f = 0; f < 5; f++) {
    const el = $(`fs${f}`);
    const val = el.querySelector('.fold-score-val');
    const lbl = el.querySelector('.fold-score-label');
    lbl.textContent = f === fold ? `Fold ${f+1} — TEST ←` : `Fold ${f+1}`;
    el.classList.remove('active','revealed');
    if (f < fold) {
      el.classList.add('revealed');
      val.textContent = DATA.foldRmse.GB[f].toFixed(2) + ' MPa';
    } else if (f === fold) {
      el.classList.add('active', 'revealed');
      val.textContent = DATA.foldRmse.GB[f].toFixed(2) + ' MPa';
    } else {
      val.textContent = '—';
    }
  }

  const completed = DATA.foldRmse.GB.slice(0, fold+1);
  const m = mean(completed), s = std(completed);
  $('cv-info').innerHTML = `<strong>Itération ${fold+1}/5 :</strong> entraînement sur ${N_OBS - foldSize} obs — test sur ${foldSize} obs. RMSE ce fold : <strong>${DATA.foldRmse.GB[fold].toFixed(2)} MPa</strong>. Moyenne courante : <strong>${m.toFixed(3)} ± ${s.toFixed(3)} MPa</strong>.`;

  if (fold === 4) {
    const m5 = mean(DATA.foldRmse.GB), s5 = std(DATA.foldRmse.GB);
    $('cv-final').innerHTML = `✅ <strong>Résultat final :</strong> RMSE = <strong style="color:var(--gb)">${m5.toFixed(3)} ± ${s5.toFixed(3)} MPa</strong> — estimation non biaisée de la GE du Gradient Boosting.`;
  } else {
    $('cv-final').textContent = '';
  }
}

function nextFold() {
  currentFold = (currentFold + 1) % N_FOLDS;
  renderCVFold(currentFold);
}
function resetCV() {
  if (cvAutoInterval) { clearInterval(cvAutoInterval); cvAutoInterval = null; }
  currentFold = 0;
  renderCVFold(0);
}
function autoPlayCV() {
  if (cvAutoInterval) { clearInterval(cvAutoInterval); cvAutoInterval = null; return; }
  cvAutoInterval = setInterval(() => {
    currentFold = (currentFold + 1) % N_FOLDS;
    renderCVFold(currentFold);
    if (currentFold === N_FOLDS-1) { clearInterval(cvAutoInterval); cvAutoInterval = null; }
  }, 1100);
}

/* ═══════════════════════════════════════════════════
   NESTED CV ANIMATION
═══════════════════════════════════════════════════ */
const NESTED_STEPS = [
  {
    outer: null, inner: null,
    msg: 'Point de départ : <strong>1005 observations</strong>. La boucle externe (outer_cv, seed=0) va créer 5 partitions. La boucle interne (inner_cv, seed=42) tune les HPs à l\'intérieur.',
    extra: ''
  },
  {
    outer: 0, inner: null,
    msg: '<strong>Outer fold 1 isolé comme TEST EXTERNE</strong> (~201 obs). Il est mis de côté — il ne participera jamais au tuning interne.',
    extra: 'Les 804 obs restantes (F2+F3+F4+F5) sont données à la boucle interne.'
  },
  {
    outer: 0, inner: -1,
    msg: '<strong>GridSearchCV démarre sur les 804 obs.</strong> Inner_cv crée 5 sous-folds. Pour chaque combinaison HP (72 au total), on entraîne 5 modèles et on moyenne les RMSE.',
    extra: '72 combos × 5 folds inner = 360 entraînements pour ce outer fold.'
  },
  {
    outer: 0, inner: 2,
    msg: 'La boucle interne a tourné sur tous ses folds. <strong>λ* trouvé :</strong> learning_rate=0.2, max_depth=4, n_estimators=300, subsample=1.0. On ré-entraîne sur les 804 obs avec λ*.',
    extra: 'Ce modèle final du outer fold 1 est prêt pour l\'évaluation.'
  },
  {
    outer: 0, inner: null, evalDone: true,
    msg: `<strong>Évaluation sur TEST EXTERNE F1</strong> → RMSE = <strong style="color:var(--accent2)">${DATA.foldRmse.GB[0].toFixed(2)} MPa</strong>. Score non biaisé : F1 n'a jamais été vu pendant le tuning.`,
    extra: 'On répète la même procédure pour F2, F3, F4, F5.'
  },
  {
    outer: 4, inner: null, allDone: true,
    msg: `<strong>5 outer folds terminés.</strong> RMSE outer CV = <strong style="color:var(--gb)">4.208 ± 0.261 MPa</strong>. Coût total : 5 outer × 72 combos × 5 inner = <strong>1 800 entraînements</strong> par modèle.`,
    extra: 'Estimation non biaisée car le test externe était isolé à chaque itération.'
  }
];

let nestedStep = 0;

function renderNestedStep() {
  const s = NESTED_STEPS[nestedStep];
  $('nested-step-lbl').textContent = `Étape ${nestedStep + 1} / ${NESTED_STEPS.length}`;

  // Outer folds
  for (let i = 0; i < 5; i++) {
    const el = $(`of${i}`);
    if (s.allDone) {
      el.className = 'mini-fold ' + (i % 2 === 0 ? 'mf-test-out' : 'mf-train-out');
      el.textContent = `F${i+1} ${DATA.foldRmse.GB[i].toFixed(1)}`;
    } else if (s.outer === null) {
      el.className = 'mini-fold mf-train-out';
      el.textContent = `F${i+1}`;
    } else if (i === s.outer) {
      el.className = 'mini-fold mf-test-out';
      el.textContent = `F${i+1} TEST`;
    } else if (i < s.outer || s.evalDone) {
      el.className = 'mini-fold mf-done-out';
      el.textContent = `F${i+1} ✓`;
    } else {
      el.className = 'mini-fold mf-train-out';
      el.textContent = `F${i+1}`;
    }
  }

  // Inner box
  const innerBox = $('inner-box');
  const showInner = s.inner !== null;
  innerBox.style.opacity = showInner ? '1' : '0.3';
  innerBox.style.pointerEvents = showInner ? 'auto' : 'none';

  // Inner folds
  for (let i = 0; i < 5; i++) {
    const el = $(`if${i}`);
    if (!showInner) {
      el.className = 'mini-fold mf-train-in';
      el.textContent = `i${i+1}`;
    } else if (s.inner === -1) {
      el.className = 'mini-fold mf-train-in';
      el.textContent = `i${i+1}`;
    } else if (i === s.inner) {
      el.className = 'mini-fold mf-test-in';
      el.textContent = `i${i+1} ←`;
    } else {
      el.className = 'mini-fold mf-train-in';
      el.textContent = `i${i+1}`;
    }
  }

  $('nested-msg').innerHTML = s.msg;
  $('nested-extra').textContent = s.extra;
}

function nextNestedStep() {
  if (nestedStep < NESTED_STEPS.length-1) nestedStep++;
  renderNestedStep();
}
function prevNestedStep() {
  if (nestedStep > 0) nestedStep--;
  renderNestedStep();
}
function resetNested() { nestedStep = 0; renderNestedStep(); }

/* ═══════════════════════════════════════════════════
   LEAKAGE ANIMATION
═══════════════════════════════════════════════════ */
const LEAK_STEPS = [
  {
    bad: ['🔴 <span class="hl-bad">scaler.fit_transform(X)</span>', '   # μ, σ calculés sur les 1005 obs', '   # (train + test mélangés !)'],
    good: ['🟢 <span class="hl-good">pipeline.fit(X_train_fold)</span>', '   # scaler.fit() sur les 804 obs', '   # du fold train uniquement']
  },
  {
    bad: ['🔴 μ_cement = 281.2 <span class="comment"># calculé sur tout X</span>', '   # → inclut les obs de test !'],
    good: ['🟢 μ_cement = 279.8 <span class="comment"># calculé sur train fold</span>', '   # → test set inconnu ✅']
  },
  {
    bad: ['🔴 X_test_scaled = scaler.transform(X_test)', '   # le scaler "connaît" déjà', '   # les stats du test set'],
    good: ['🟢 pipe.predict(X_test_fold)', '   # scaler transforme avec stats', '   # du train_fold uniquement ✅']
  },
  {
    bad: ['🔴 GE estimée : trop optimiste', '   # le modèle a vu (indirectement)', '   # les données de test'],
    good: ['🟢 GE estimée : honnête ✅', '   # aucune contamination', '   # estimation non biaisée']
  }
];

let leakStep = 0, leakInterval = null;

function renderLeakStep(step) {
  const s = LEAK_STEPS[step];
  const bad = $('leak-bad-steps');
  const good = $('leak-good-steps');
  bad.innerHTML = '';
  good.innerHTML = '';
  for (let i = 0; i <= step; i++) {
    LEAK_STEPS[i].bad.forEach(l => {
      const d = document.createElement('div');
      d.className = 'flow-step bad-step';
      d.innerHTML = `<span class="code-block" style="padding:4px 8px;border-radius:4px;display:inline">${l}</span>`;
      bad.appendChild(d);
    });
    LEAK_STEPS[i].good.forEach(l => {
      const d = document.createElement('div');
      d.className = 'flow-step good-step';
      d.innerHTML = `<span class="code-block" style="padding:4px 8px;border-radius:4px;display:inline">${l}</span>`;
      good.appendChild(d);
    });
  }
}

function animLeakage() {
  leakStep = 0;
  renderLeakStep(0);
  if (leakInterval) clearInterval(leakInterval);
  leakInterval = setInterval(() => {
    leakStep++;
    if (leakStep >= LEAK_STEPS.length) { clearInterval(leakInterval); leakInterval = null; return; }
    renderLeakStep(leakStep);
  }, 1400);
}

/* ═══════════════════════════════════════════════════
   FEATURE IMPORTANCE (SVG horizontal bars)
═══════════════════════════════════════════════════ */
let impMode = 'gb';

function renderImportanceBars(mode) {
  impMode = mode;
  $('btn-imp-gb').className    = mode==='gb'    ? 'btn btn-primary' : 'btn btn-secondary';
  $('btn-imp-ridge').className = mode==='ridge' ? 'btn btn-primary' : 'btn btn-secondary';

  const W = 600, BAR_H = 28, GAP = 10;
  const leftPad = 140, rightPad = 60;

  let entries;
  if (mode === 'gb') {
    $('imp-subtitle').className = 'info-box';
    $('imp-subtitle').innerHTML = '<strong>Impurity Importance GB :</strong> somme des réductions MSE à chaque split, sur tous les arbres. age + cement = 64% — cohérent avec la physique du béton.';
    entries = Object.entries(DATA.gbImportance).sort((a,b)=>b[1]-a[1]);
  } else {
    $('imp-subtitle').className = 'info-box';
    $('imp-subtitle').innerHTML = '<strong>Coefficients Ridge standardisés :</strong> après StandardScaler (μ=0, σ=1), les coefficients sont en "unités d\'écart-type" → directement comparables. Le signe indique la direction.';
    entries = Object.entries(DATA.ridgeCoef).sort((a,b)=>b[1]-a[1]);
  }

  const container = $('imp-svg');
  container.innerHTML = '';
  const H = entries.length * (BAR_H + GAP) + 20;
  const svg = makeSVG(W, H);
  svg.style.width = '100%'; svg.style.height = 'auto';

  if (mode === 'gb') {
    // Simple positive bars
    const maxVal = entries[0][1];
    entries.forEach(([feat, val], i) => {
      const y = i * (BAR_H + GAP) + 10;
      const barW = (val / maxVal) * (W - leftPad - rightPad);
      const color = DATA.featureColors[feat];

      // Feature label
      svgText(svg, feat, leftPad - 8, y + BAR_H/2, { anchor:'end', base:'middle', size:12.5, color: getThemeVar('--text') });

      // Bar background
      svg.appendChild(svgEl('rect', { x:leftPad, y, width: W-leftPad-rightPad, height:BAR_H, rx:4, fill: getThemeVar('--surface2') }));
      // Bar fill
      svg.appendChild(svgEl('rect', { x:leftPad, y, width: Math.max(barW,4), height:BAR_H, rx:4, fill:color, opacity:0.85 }));
      // Value
      svgText(svg, val.toFixed(1)+'%', leftPad + barW + 8, y + BAR_H/2, { anchor:'start', base:'middle', size:11, color });
    });
  } else {
    // Diverging bars (negative water)
    const maxAbs = Math.max(...entries.map(([,v])=>Math.abs(v)));
    const center = leftPad + (W - leftPad - rightPad) / 2;
    const scale = (W - leftPad - rightPad) / 2 / maxAbs;

    // Center line
    svg.appendChild(svgEl('line', { x1:center, y1:5, x2:center, y2:H-5, stroke: getThemeVar('--svg-axis'), 'stroke-width':1.5 }));
    svgText(svg, '0', center, H-3, { anchor:'middle', size:10 });

    entries.forEach(([feat, val], i) => {
      const y = i * (BAR_H + GAP) + 10;
      const color = val < 0 ? 'var(--ridge)' : DATA.featureColors[feat];
      const barW = Math.abs(val) * scale;
      const barX = val >= 0 ? center : center - barW;

      // Feature label
      svgText(svg, feat, leftPad - 8, y + BAR_H/2, { anchor:'end', base:'middle', size:12.5, color: getThemeVar('--text') });
      // Background half
      svg.appendChild(svgEl('rect', { x:leftPad, y, width:W-leftPad-rightPad, height:BAR_H, rx:4, fill: getThemeVar('--surface2') }));
      // Bar
      svg.appendChild(svgEl('rect', { x:barX, y, width:Math.max(barW,4), height:BAR_H, rx:4, fill:color, opacity:0.85 }));
      // Value
      const sign = val > 0 ? '+' : '';
      const vx = val >= 0 ? center + barW + 8 : center - barW - 8;
      svgText(svg, sign+val.toFixed(2), vx, y + BAR_H/2, {
        anchor: val >= 0 ? 'start' : 'end', base:'middle', size:11, color
      });
    });
  }

  container.appendChild(svg);
}

/* ═══════════════════════════════════════════════════
   COMPARISON CHARTS (SVG)
═══════════════════════════════════════════════════ */
function renderComparisonChart() {
  const container = $('comparison-svg');
  container.innerHTML = '';
  const W=500, H=260;
  const M={top:20,right:30,bottom:44,left:56};
  const PW=W-M.left-M.right, PH=H-M.top-M.bottom;

  const models = ['Ridge','RF','GB'];
  const rmses = [DATA.models.Ridge.rmse, DATA.models.RF.rmse, DATA.models.GB.rmse];
  const stds  = [DATA.models.Ridge.std,  DATA.models.RF.std,  DATA.models.GB.std];
  const colors= ['var(--ridge)', 'var(--rf)', 'var(--gb)'];
  const maxR = 12, barW = PW / models.length * 0.5;

  function xOf(i) { return M.left + (i+0.5) * PW/models.length; }
  function yOf(r) { return M.top + PH - (r / maxR) * PH; }

  const svg = makeSVG(W, H);
  svg.style.width='100%'; svg.style.height='auto';
  drawGrid(svg, M.left, M.top, PW, PH, 3, 6);
  drawAxis(svg, M.left, M.top, M.left, M.top+PH);
  drawAxis(svg, M.left, M.top+PH, M.left+PW, M.top+PH);

  [0,2,4,6,8,10,12].forEach(v=>{
    const yy=yOf(v);
    if(yy<M.top)return;
    drawAxis(svg,M.left-4,yy,M.left,yy);
    svgText(svg,v,M.left-8,yy,{anchor:'end',base:'middle',size:10});
  });
  svgText(svg,'RMSE (MPa)',14,M.top+PH/2,{anchor:'middle',base:'middle',size:11,transform:`rotate(-90,14,${M.top+PH/2})`});

  models.forEach((name,i)=>{
    const x=xOf(i), y=yOf(rmses[i]);
    const bx=x-barW/2, bh=yOf(0)-y;
    // Bar
    svg.appendChild(svgEl('rect',{x:bx,y,width:barW,height:bh,rx:4,fill:colors[i],opacity:0.8}));
    // Error bar
    const yTop=yOf(rmses[i]+stds[i]), yBot=yOf(rmses[i]-stds[i]);
    svg.appendChild(svgEl('line',{x1:x,y1:yTop,x2:x,y2:yBot,stroke:'white','stroke-width':2}));
    svg.appendChild(svgEl('line',{x1:x-6,y1:yTop,x2:x+6,y2:yTop,stroke:'white','stroke-width':2}));
    svg.appendChild(svgEl('line',{x1:x-6,y1:yBot,x2:x+6,y2:yBot,stroke:'white','stroke-width':2}));
    // Label
    svgText(svg,rmses[i].toFixed(2),x,y-8,{anchor:'middle',size:11,color:colors[i]});
    svgText(svg,name,x,M.top+PH+16,{anchor:'middle',size:12,color:getThemeVar('--text')});
  });

  container.appendChild(svg);
}

function renderR2Chart() {
  const container = $('r2-svg');
  container.innerHTML = '';
  const W=500, H=260;
  const M={top:20,right:30,bottom:44,left:56};
  const PW=W-M.left-M.right, PH=H-M.top-M.bottom;

  const models = ['Ridge','RF','GB'];
  const r2s   = [DATA.models.Ridge.r2, DATA.models.RF.r2, DATA.models.GB.r2];
  const colors= ['var(--ridge)', 'var(--rf)', 'var(--gb)'];
  const barW  = PW / models.length * 0.5;

  function xOf(i) { return M.left + (i+0.5) * PW/models.length; }
  function yOf(r) { return M.top + PH - r * PH; }

  const svg = makeSVG(W, H);
  svg.style.width='100%'; svg.style.height='auto';
  drawGrid(svg, M.left, M.top, PW, PH, 3, 5);
  drawAxis(svg, M.left, M.top, M.left, M.top+PH);
  drawAxis(svg, M.left, M.top+PH, M.left+PW, M.top+PH);

  [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(v=>{
    const yy=yOf(v);
    drawAxis(svg,M.left-4,yy,M.left,yy);
    svgText(svg,(v*100).toFixed(0)+'%',M.left-8,yy,{anchor:'end',base:'middle',size:10});
  });
  svgText(svg,'R²',14,M.top+PH/2,{anchor:'middle',base:'middle',size:11,transform:`rotate(-90,14,${M.top+PH/2})`});

  // 100% reference line
  svg.appendChild(svgEl('line',{x1:M.left,y1:yOf(1),x2:M.left+PW,y2:yOf(1),stroke:'rgba(255,255,255,0.3)','stroke-dasharray':'4 3'}));

  models.forEach((name,i)=>{
    const x=xOf(i), y=yOf(r2s[i]);
    const bx=x-barW/2, bh=yOf(0)-y;
    svg.appendChild(svgEl('rect',{x:bx,y,width:barW,height:bh,rx:4,fill:colors[i],opacity:0.8}));
    svgText(svg,(r2s[i]*100).toFixed(1)+'%',x,y-8,{anchor:'middle',size:11,color:colors[i]});
    svgText(svg,name,x,M.top+PH+16,{anchor:'middle',size:12,color:getThemeVar('--text')});
  });

  container.appendChild(svg);
}

/* ═══════════════════════════════════════════════════
   BIAIS-VARIANCE — COMPLEXITY CURVE & DECOMPOSITION
═══════════════════════════════════════════════════ */
let currentComplexity = 2;

// Modèles théoriques (illustratifs, pas les vraies valeurs RMSE)
function bias2(c)     { return 9 * Math.exp(-c/25); }
function variance(c)  { return 0.2 + 7 * Math.pow(c/100, 3); }
const NOISE = 1.2;
function trainErr(c)  { return 0.6 + bias2(c) * 0.5; }
function testErr(c)   { return bias2(c) + variance(c) + NOISE; }

const COMPLEXITY_MARKERS = [
  { c: 5,  label: 'Constant',  color: 'var(--muted)' },
  { c: 14, label: 'Ridge',     color: 'var(--ridge)' },
  { c: 50, label: 'RF',        color: 'var(--rf)' },
  { c: 58, label: 'GB',        color: 'var(--gb)' },
  { c: 95, label: 'Arbre profond', color: 'var(--accent2)' }
];

function renderBiasVarianceCurve(c) {
  const container = $('biasvar-curve-svg');
  container.innerHTML = '';
  const W = 660, H = 280;
  const M = { top: 20, right: 24, bottom: 44, left: 56 };
  const PW = W - M.left - M.right, PH = H - M.top - M.bottom;
  const MAXY = 11;

  function xOf(v) { return M.left + (v/100) * PW; }
  function yOf(v) { return M.top + PH - (v/MAXY) * PH; }

  const svg = makeSVG(W, H);
  svg.style.width = '100%'; svg.style.height = 'auto';

  drawGrid(svg, M.left, M.top, PW, PH, 5, 5);
  drawAxis(svg, M.left, M.top, M.left, M.top+PH);
  drawAxis(svg, M.left, M.top+PH, M.left+PW, M.top+PH);

  [0,2,4,6,8,10].forEach(v => {
    const yy = yOf(v);
    drawAxis(svg, M.left-4, yy, M.left, yy);
    svgText(svg, v, M.left-8, yy, { anchor:'end', base:'middle', size:10 });
  });
  svgText(svg, 'Erreur (illustratif)', 14, M.top+PH/2, {
    anchor:'middle', base:'middle', size:11, transform:`rotate(-90,14,${M.top+PH/2})`
  });
  svgText(svg, 'Complexité du modèle →', M.left+PW/2, H-6, { size:11 });

  // Curves
  const N = 100;
  function buildPath(fn) {
    const pts = [];
    for (let i=0;i<=N;i++) {
      const cc = i;
      pts.push(`${xOf(cc)},${yOf(fn(cc))}`);
    }
    return 'M ' + pts.join(' L ');
  }
  svg.appendChild(svgEl('path', { d: buildPath(trainErr), stroke: 'var(--rf)', 'stroke-width': 2.5, fill:'none' }));
  svg.appendChild(svgEl('path', { d: buildPath(testErr), stroke: 'var(--ridge)', 'stroke-width': 2.5, fill:'none' }));

  svgText(svg, 'Train error', M.left+PW-10, yOf(trainErr(100))-8, { anchor:'end', size:11, color:'var(--rf)' });
  svgText(svg, 'Test error (≈ GE)', M.left+PW-10, yOf(testErr(100))-8, { anchor:'end', size:11, color:'var(--ridge)' });

  // Markers for our models
  COMPLEXITY_MARKERS.forEach(m => {
    const x = xOf(m.c), y = yOf(testErr(m.c));
    svg.appendChild(svgEl('circle', { cx:x, cy:y, r:4, fill:m.color, stroke:getThemeVar('--surface'), 'stroke-width':1.5 }));
    svgText(svg, m.label, x, M.top+PH+16, { size:9.5, color:m.color });
  });

  // Current complexity marker
  const xC = xOf(c);
  svg.appendChild(svgEl('line', { x1:xC, y1:M.top, x2:xC, y2:M.top+PH, stroke: getThemeVar('--svg-axis'), 'stroke-width':1.5, 'stroke-dasharray':'4 3' }));
  svg.appendChild(svgEl('circle', { cx:xC, cy:yOf(testErr(c)), r:5.5, fill:'var(--accent)', stroke:getThemeVar('--surface'), 'stroke-width':2 }));
  svg.appendChild(svgEl('circle', { cx:xC, cy:yOf(trainErr(c)), r:5.5, fill:'var(--accent4)', stroke:getThemeVar('--surface'), 'stroke-width':2 }));

  container.appendChild(svg);
}

function renderDecompBars(c) {
  const container = $('biasvar-decomp-svg');
  const b2 = bias2(c), v = variance(c), n = NOISE;
  const maxV = 10;
  function row(label, val, color) {
    const pct = Math.min(100, (val/maxV)*100);
    return `<div class="r2-bar-row">
      <div class="r2-bar-label" style="color:${color}">${label}</div>
      <div class="r2-bar-track"><div class="r2-bar-fill" style="width:${pct}%; background:${color}"></div></div>
      <div class="r2-bar-pct">${val.toFixed(2)}</div>
    </div>`;
  }
  container.innerHTML =
    row('Biais²', b2, 'var(--ridge)') +
    row('Variance', v, 'var(--gb)') +
    row('σ²ε', n, 'var(--muted)') +
    `<div class="info-box" style="margin-top:10px"><strong>GE ≈ ${(b2+v+n).toFixed(2)}</strong> (somme des trois termes)</div>`;
}

function initComplexitySlider() {
  const slider = $('complexity-slider');
  function update() {
    const c = +slider.value;
    currentComplexity = c;
    let label;
    if (c < 8) label = 'Modèle constant (prédit la moyenne)';
    else if (c < 25) label = 'Ridge (linéaire)';
    else if (c < 75) label = 'RF / GB bien réglés';
    else label = 'Arbre profond non régularisé';
    $('complexity-val').textContent = label;

    let msg, cls;
    if (c < 8) {
      msg = '<strong>Sous-apprentissage extrême :</strong> le modèle prédit ~35 MPa pour tout. Biais maximal, variance quasi nulle. Train error ET test error sont élevées et proches.';
      cls = 'warn-box';
    } else if (c < 25) {
      msg = '<strong>Ridge :</strong> biais élevé (relation linéaire ne capte pas log(age)), variance faible. Train ≈ Test mais les deux restent élevées → underfitting.';
      cls = 'warn-box';
    } else if (c < 75) {
      msg = '<strong>RF / GB bien réglés :</strong> proche du minimum de l\'erreur de test — le meilleur compromis biais/variance. C\'est la zone visée par le tuning.';
      cls = 'ok-box';
    } else {
      msg = '<strong>Arbre profond non régularisé :</strong> biais quasi nul (colle aux données train), mais variance énorme → train error ≈ 0, test error remonte. Overfitting.';
      cls = 'warn-box';
    }
    $('biasvar-msg').className = cls;
    $('biasvar-msg').innerHTML = msg;

    renderBiasVarianceCurve(c);
    renderDecompBars(c);
  }
  slider.addEventListener('input', update);
  update();
}

/* ═══════════════════════════════════════════════════
   MÉTRIQUES — MAE vs RMSE SUR UN OUTLIER
═══════════════════════════════════════════════════ */
let currentOutlierErr = 5;

function renderMetricCurve(outlierErr) {
  const container = $('metric-curve-svg');
  container.innerHTML = '';
  const W = 640, H = 240;
  const M = { top: 16, right: 20, bottom: 36, left: 50 };
  const PW = W - M.left - M.right, PH = H - M.top - M.bottom;

  function maeOf(x) { return (9*2 + x) / 10; }
  function mseOf(x) { return (9*4 + x*x) / 10; }
  function rmseOf(x) { return Math.sqrt(mseOf(x)); }

  function xOf(v) { return M.left + (v/20) * PW; }
  function yOf(v) { return M.top + PH - (v/8) * PH; }

  const svg = makeSVG(W, H);
  svg.style.width = '100%'; svg.style.height = 'auto';
  drawGrid(svg, M.left, M.top, PW, PH, 5, 4);
  drawAxis(svg, M.left, M.top, M.left, M.top+PH);
  drawAxis(svg, M.left, M.top+PH, M.left+PW, M.top+PH);

  [0,2,4,6,8].forEach(v => {
    const yy = yOf(v);
    drawAxis(svg, M.left-4, yy, M.left, yy);
    svgText(svg, v, M.left-8, yy, { anchor:'end', base:'middle', size:10 });
  });
  [0,5,10,15,20].forEach(v => {
    const xx = xOf(v);
    drawAxis(svg, xx, M.top+PH, xx, M.top+PH+4);
    svgText(svg, v, xx, M.top+PH+15, { size:10 });
  });
  svgText(svg, "Erreur de l'outlier (MPa)", M.left+PW/2, H-4, { size:11 });

  const N = 80;
  function buildPath(fn) {
    const pts = [];
    for (let i=0;i<=N;i++) {
      const x = i/N*20;
      pts.push(`${xOf(x)},${yOf(fn(x))}`);
    }
    return 'M ' + pts.join(' L ');
  }
  svg.appendChild(svgEl('path', { d: buildPath(maeOf), stroke:'var(--accent3)', 'stroke-width':2.5, fill:'none' }));
  svg.appendChild(svgEl('path', { d: buildPath(rmseOf), stroke:'var(--ridge)', 'stroke-width':2.5, fill:'none' }));
  svgText(svg, 'MAE', xOf(20)-8, yOf(maeOf(20))-8, { anchor:'end', size:11, color:'var(--accent3)' });
  svgText(svg, 'RMSE', xOf(20)-8, yOf(rmseOf(20))-8, { anchor:'end', size:11, color:'var(--ridge)' });

  // Marker
  const xH = xOf(outlierErr);
  svg.appendChild(svgEl('line', { x1:xH, y1:M.top, x2:xH, y2:M.top+PH, stroke:getThemeVar('--svg-axis'), 'stroke-width':1.5, 'stroke-dasharray':'4 3' }));
  svg.appendChild(svgEl('circle', { cx:xH, cy:yOf(maeOf(outlierErr)), r:5, fill:'var(--accent3)', stroke:getThemeVar('--surface'), 'stroke-width':1.5 }));
  svg.appendChild(svgEl('circle', { cx:xH, cy:yOf(rmseOf(outlierErr)), r:5, fill:'var(--ridge)', stroke:getThemeVar('--surface'), 'stroke-width':1.5 }));

  container.appendChild(svg);
}

function initMetricSlider() {
  const slider = $('metric-err-slider');
  function update() {
    const x = +slider.value;
    currentOutlierErr = x;
    $('metric-err-val').textContent = x.toFixed(1) + ' MPa';

    const mae = (9*2+x)/10, mse = (9*4+x*x)/10, rmse = Math.sqrt(mse);
    $('metric-mae-val').textContent = mae.toFixed(2);
    $('metric-mse-val').textContent = mse.toFixed(2);
    $('metric-rmse-val').textContent = rmse.toFixed(2);

    let msg, cls;
    if (x <= 2) {
      msg = "<strong>Pas d'outlier :</strong> les 10 erreurs valent ~2 MPa. MAE ≈ RMSE — pour une seule observation, RMSE = |erreur| = MAE.";
      cls = 'info-box';
    } else if (x <= 10) {
      msg = `<strong>Outlier modéré (${x} MPa) :</strong> RMSE (${rmse.toFixed(2)}) commence à s'écarter de MAE (${mae.toFixed(2)}) — le carré amplifie cette seule grande erreur.`;
      cls = 'info-box';
    } else {
      msg = `<strong>Gros outlier (${x} MPa) :</strong> RMSE (${rmse.toFixed(2)}) explose par rapport à MAE (${mae.toFixed(2)}). Si notre dataset avait beaucoup d'erreurs comme ça, MAE serait plus représentatif — mais l'analyse IQR montre que ce n'est pas notre cas (0.8% d'outliers sur la target).`;
      cls = 'warn-box';
    }
    $('metric-msg').className = cls;
    $('metric-msg').innerHTML = msg;

    renderMetricCurve(x);
  }
  slider.addEventListener('input', update);
  update();
}

/* ═══════════════════════════════════════════════════
   GRID SEARCH HEATMAP (Fiche 04)
═══════════════════════════════════════════════════ */
let gridAnimInterval = null;

function rmseToColor(v) {
  const { rmse } = DATA.gridSearch;
  let min = Infinity, max = -Infinity;
  rmse.forEach(row => row.forEach(x => { min = Math.min(min,x); max = Math.max(max,x); }));
  const t = (v - min) / (max - min); // 0 = best (green), 1 = worst (red)
  const hue = 145 - t * 145; // 145=green -> 0=red
  return `hsla(${hue}, 65%, 50%, ${0.18 + t*0.12})`;
}

function buildGridHeatmap() {
  const container = $('grid-heatmap');
  container.innerHTML = '';
  const grid = document.createElement('div');
  grid.className = 'heatmap-grid';

  // Header row
  grid.appendChild(Object.assign(document.createElement('div'), { className:'heatmap-cell head', textContent:'lr \\ depth' }));
  DATA.gridSearch.maxDepths.forEach(d => {
    grid.appendChild(Object.assign(document.createElement('div'), { className:'heatmap-cell head', textContent:`depth=${d}` }));
  });

  DATA.gridSearch.learningRates.forEach((lr, i) => {
    grid.appendChild(Object.assign(document.createElement('div'), { className:'heatmap-cell head', textContent:`lr=${lr}` }));
    DATA.gridSearch.maxDepths.forEach((d, j) => {
      const val = DATA.gridSearch.rmse[i][j];
      const cell = document.createElement('div');
      cell.className = 'heatmap-cell';
      cell.id = `hm-${i}-${j}`;
      cell.textContent = val.toFixed(2);
      cell.style.background = rmseToColor(val);
      grid.appendChild(cell);
    });
  });

  container.appendChild(grid);
}

function animGridSearch() {
  resetGridSearch();
  const cells = [];
  DATA.gridSearch.learningRates.forEach((lr,i) => DATA.gridSearch.maxDepths.forEach((d,j) => cells.push([i,j])));
  let idx = 0;
  $('grid-msg').className = 'info-box';
  $('grid-msg').innerHTML = '<strong>GridSearchCV scanne les 12 combinaisons</strong> (ici un sous-ensemble illustratif des 72)...';
  gridAnimInterval = setInterval(() => {
    if (idx > 0) {
      const [pi,pj] = cells[idx-1];
      $(`hm-${pi}-${pj}`).classList.remove('scanning');
    }
    if (idx >= cells.length) {
      clearInterval(gridAnimInterval); gridAnimInterval = null;
      const { bestRow, bestCol, learningRates, maxDepths, rmse } = DATA.gridSearch;
      $(`hm-${bestRow}-${bestCol}`).classList.add('best');
      $('grid-msg').className = 'ok-box';
      $('grid-msg').innerHTML = `<strong>Meilleure combo :</strong> learning_rate=${learningRates[bestRow]}, max_depth=${maxDepths[bestCol]} → RMSE inner ≈ ${rmse[bestRow][bestCol]} MPa. C'est cette config qui sera ré-entraînée sur le train externe complet.`;
      return;
    }
    const [i,j] = cells[idx];
    $(`hm-${i}-${j}`).classList.add('scanning');
    idx++;
  }, 180);
}

function resetGridSearch() {
  if (gridAnimInterval) { clearInterval(gridAnimInterval); gridAnimInterval = null; }
  buildGridHeatmap();
  $('grid-msg').className = 'info-box';
  $('grid-msg').innerHTML = "Clique sur <strong>« Scanner la grille »</strong> pour voir GridSearchCV évaluer chaque combinaison par 5-fold inner CV.";
}

/* ═══════════════════════════════════════════════════
   CART SPLIT ANIMATION (Fiche 08)
═══════════════════════════════════════════════════ */
function seededRandom(seed) {
  let s = seed;
  return () => { s = (s*9301+49297) % 233280; return s/233280; };
}

const CART_POINTS = (() => {
  const rnd = seededRandom(42);
  const pts = [];
  for (let i=0;i<40;i++) {
    const x = rnd()*100;
    const y = rnd()*100;
    let cls;
    if (x > 50 && y > 50) cls = 3;       // cement haut, age haut → forte résistance
    else if (x > 50 && y <= 50) cls = 2; // cement haut, age bas
    else if (x <= 50 && y > 50) cls = 1; // cement bas, age haut
    else cls = 0;                         // cement bas, age bas → faible résistance
    pts.push({x,y,cls});
  }
  return pts;
})();
const CART_CLASS_COLORS = ['var(--ridge)', 'var(--accent4)', 'var(--accent)', 'var(--rf)'];

const CART_STEPS = [
  { lines: [], msg: "<strong>Nœud racine :</strong> 40 observations mélangées, forte variance de résistance. CART cherche le split (feature, seuil) qui réduit le plus la variance." },
  { lines: [{type:'v', val:50}], msg: "<strong>Split 1 — cement &lt; ~330 kg/m³ ?</strong> Sépare en deux groupes : à gauche (cement bas), à droite (cement élevé). C'est le split qui réduit le plus la variance globale." },
  { lines: [{type:'v', val:50}, {type:'h', val:50, xmin:50, xmax:100}], msg: "<strong>Split 2 (branche droite) — age &lt; ~28 jours ?</strong> Le sous-groupe 'cement élevé' est encore hétérogène → nouveau split sur age." },
  { lines: [{type:'v', val:50}, {type:'h', val:50, xmin:50, xmax:100}, {type:'h', val:50, xmin:0, xmax:50}], msg: "<strong>Split 3 (branche gauche) — age &lt; ~28 jours ?</strong> Même logique appliquée récursivement à gauche." },
  { lines: [{type:'v', val:50}, {type:'h', val:50, xmin:50, xmax:100}, {type:'h', val:50, xmin:0, xmax:50}], msg: "<strong>4 feuilles = 4 prédictions constantes</strong> (moyenne du groupe). Plus de profondeur → régions plus petites → variance ↑ (overfitting si on continue trop). C'est pour ça que RF/GB limitent ou compensent la profondeur.", final: true }
];
let cartStep = 0;

function renderCartSplit(step) {
  const s = CART_STEPS[step];
  const container = $('cart-split-svg');
  container.innerHTML = '';
  const W = 420, H = 420;
  const M = 30;
  const PW = W - 2*M, PH = H - 2*M;
  function xOf(v) { return M + (v/100)*PW; }
  function yOf(v) { return M + PH - (v/100)*PH; }

  const svg = makeSVG(W, H);
  svg.style.width = '100%'; svg.style.height = 'auto'; svg.style.maxWidth = '420px'; svg.style.display = 'block'; svg.style.margin = '0 auto';

  // Frame
  svg.appendChild(svgEl('rect', { x:M, y:M, width:PW, height:PH, fill:'none', stroke:getThemeVar('--svg-axis') }));
  svgText(svg, 'cement →', M+PW/2, H-6, { size:11 });
  svgText(svg, 'age →', 12, M+PH/2, { size:11, transform:`rotate(-90,12,${M+PH/2})` });

  // Split lines
  s.lines.forEach(l => {
    if (l.type === 'v') {
      svg.appendChild(svgEl('line', { x1:xOf(l.val), y1:M, x2:xOf(l.val), y2:M+PH, stroke:'var(--accent)', 'stroke-width':2, 'stroke-dasharray':'5 3' }));
    } else {
      svg.appendChild(svgEl('line', { x1:xOf(l.xmin), y1:yOf(l.val), x2:xOf(l.xmax), y2:yOf(l.val), stroke:'var(--accent)', 'stroke-width':2, 'stroke-dasharray':'5 3' }));
    }
  });

  // Points
  CART_POINTS.forEach(p => {
    svg.appendChild(svgEl('circle', { cx:xOf(p.x), cy:yOf(p.y), r:4.5, fill: CART_CLASS_COLORS[p.cls], opacity: s.final ? 0.9 : 0.65 }));
  });

  // Final region labels
  if (s.final) {
    const regions = [
      { x:25, y:25, label:'faible' }, { x:25, y:75, label:'moyen' },
      { x:75, y:25, label:'moyen+' }, { x:75, y:75, label:'élevé' }
    ];
    regions.forEach(r => {
      svgText(svg, r.label, xOf(r.x), yOf(r.y), { size:12, color:getThemeVar('--text'), 'font-weight':700 });
    });
  }

  container.appendChild(svg);
  $('cart-msg').innerHTML = s.msg;
}

function nextCartSplit() {
  if (cartStep < CART_STEPS.length-1) cartStep++;
  renderCartSplit(cartStep);
}
function resetCartSplit() { cartStep = 0; renderCartSplit(0); }

/* ═══════════════════════════════════════════════════
   BAGGING (RANDOM FOREST) — VARIANCE REDUCTION
═══════════════════════════════════════════════════ */
const TRUE_VALUE = 35;
let baggingPreds = [];

function regenBaggingPreds() {
  baggingPreds = [];
  for (let i=0;i<8;i++) {
    baggingPreds.push(TRUE_VALUE + (Math.random()-0.5) * 30);
  }
}
regenBaggingPreds();

function renderBaggingSVG() {
  const container = $('bagging-svg');
  container.innerHTML = '';
  const W = 560, H = 160;
  const M = { top: 24, right: 24, bottom: 32, left: 24 };
  const PW = W - M.left - M.right;
  const MIN_V = 10, MAX_V = 60;
  function xOf(v) { return M.left + (v-MIN_V)/(MAX_V-MIN_V) * PW; }

  const svg = makeSVG(W, H);
  svg.style.width = '100%'; svg.style.height = 'auto';

  // Axis
  svg.appendChild(svgEl('line', { x1:M.left, y1:H-M.bottom, x2:W-M.right, y2:H-M.bottom, stroke:getThemeVar('--svg-axis') }));
  [10,20,30,40,50,60].forEach(v => {
    const xx = xOf(v);
    drawAxis(svg, xx, H-M.bottom, xx, H-M.bottom+4);
    svgText(svg, v, xx, H-M.bottom+16, { size:10 });
  });
  svgText(svg, 'Résistance prédite (MPa)', M.left+PW/2, H-2, { size:10 });

  // True value line
  svg.appendChild(svgEl('line', { x1:xOf(TRUE_VALUE), y1:M.top-6, x2:xOf(TRUE_VALUE), y2:H-M.bottom, stroke:'var(--rf)', 'stroke-width':2, 'stroke-dasharray':'5 3' }));
  svgText(svg, `vraie valeur = ${TRUE_VALUE}`, xOf(TRUE_VALUE), M.top-10, { size:10, color:'var(--rf)' });

  // Individual tree predictions
  baggingPreds.forEach((p,i) => {
    const y = M.top + 18 + (i % 4) * 16;
    svg.appendChild(svgEl('circle', { cx:xOf(p), cy:y, r:4.5, fill:'var(--ridge)', opacity:0.75 }));
  });

  // Average marker
  const avg = mean(baggingPreds);
  const yAvg = M.top + 18 + 1.5*16;
  const d = 6;
  const ax = xOf(avg);
  svg.appendChild(svgEl('path', { d:`M ${ax} ${yAvg-d} L ${ax+d} ${yAvg} L ${ax} ${yAvg+d} L ${ax-d} ${yAvg} Z`, fill:'var(--accent)' }));
  svgText(svg, `moyenne (8) = ${avg.toFixed(1)}`, ax, yAvg+22, { size:10, color:'var(--accent)' });

  container.appendChild(svg);
}

function redrawBagging() { regenBaggingPreds(); renderBaggingSVG(); }

/* ═══════════════════════════════════════════════════
   GRADIENT BOOSTING — RESIDUAL SHRINKAGE ANIMATION
═══════════════════════════════════════════════════ */
const BOOST_BASE_RESIDUALS = [8,12,5,15,9,3,11,7];
const BOOST_DECAY = 0.7;
let boostStep = 0;

function boostResiduals(step) {
  return BOOST_BASE_RESIDUALS.map(r => r * Math.pow(BOOST_DECAY, step));
}
function boostRmse(step) {
  const r = boostResiduals(step);
  return Math.sqrt(mean(r.map(x=>x*x)));
}

function renderBoostStep(step) {
  const container = $('boost-svg');
  container.innerHTML = '';
  const residuals = boostResiduals(step);
  const W = 420, H = 200;
  const M = { top: 16, right: 10, bottom: 30, left: 36 };
  const PW = W - M.left - M.right, PH = H - M.top - M.bottom;
  const BAR_W = PW / residuals.length * 0.6;
  const MAXR = 16;

  function xOf(i) { return M.left + (i+0.5) * PW/residuals.length; }
  function yOf(v) { return M.top + PH - (v/MAXR)*PH; }

  const svg = makeSVG(W, H);
  svg.style.width = '100%'; svg.style.height = 'auto';
  drawGrid(svg, M.left, M.top, PW, PH, residuals.length, 4);
  drawAxis(svg, M.left, M.top, M.left, M.top+PH);
  drawAxis(svg, M.left, M.top+PH, M.left+PW, M.top+PH);
  [0,4,8,12,16].forEach(v=>{
    const yy=yOf(v);
    drawAxis(svg,M.left-4,yy,M.left,yy);
    svgText(svg,v,M.left-8,yy,{anchor:'end',base:'middle',size:9});
  });
  svgText(svg, '|résidu| par observation (MPa)', M.left+PW/2, H-2, { size:10 });

  residuals.forEach((r,i) => {
    const x = xOf(i)-BAR_W/2, y = yOf(r), h = yOf(0)-y;
    svg.appendChild(svgEl('rect', { x, y, width:BAR_W, height:h, rx:3, fill:'var(--gb)', opacity:0.55+0.45*(1-step/3) }));
  });

  container.appendChild(svg);

  const rmse = boostRmse(step);
  const labels = [
    "<strong>Itération 0 :</strong> le modèle initial prédit la moyenne globale. Résidus = écarts bruts à la vraie résistance.",
    "<strong>Itération 1 (+arbre 1, lr=0.2) :</strong> chaque arbre corrige une fraction (≈20%) du résidu de l'étape précédente.",
    "<strong>Itération 2 (+arbre 2) :</strong> les résidus continuent de rétrécir — chaque arbre se concentre sur les erreurs restantes.",
    "<strong>Itération 3 (+arbre 3 ... jusqu'à 300) :</strong> après 300 arbres, les résidus sont très faibles → RMSE final = 4.208 MPa."
  ];
  $('boost-msg').innerHTML = `${labels[step]} RMSE résiduel ≈ <strong>${rmse.toFixed(2)} MPa</strong>.`;
}

function nextBoostStep() {
  if (boostStep < 3) boostStep++;
  renderBoostStep(boostStep);
}
function resetBoost() { boostStep = 0; renderBoostStep(0); }

/* ═══════════════════════════════════════════════════
   PARTIAL DEPENDENCE — RÉSISTANCE vs ÂGE (Fiche 10)
═══════════════════════════════════════════════════ */
function gbStrengthAtAge(age) {
  return 20 + 20 * Math.log10(age+1) / Math.log10(366);
}
function ridgeStrengthAtAge(age) {
  const a1 = gbStrengthAtAge(1), a365 = gbStrengthAtAge(365);
  return a1 + (a365-a1) * (age-1)/364;
}

function renderPDP() {
  const container = $('pdp-svg');
  container.innerHTML = '';
  const W = 640, H = 280;
  const M = { top: 20, right: 110, bottom: 40, left: 50 };
  const PW = W - M.left - M.right, PH = H - M.top - M.bottom;

  function xOf(age) { return M.left + (Math.log10(age) / Math.log10(365)) * PW; }
  function yOf(s) { return M.top + PH - ((s-15)/(45-15)) * PH; }

  const svg = makeSVG(W, H);
  svg.style.width = '100%'; svg.style.height = 'auto';
  drawGrid(svg, M.left, M.top, PW, PH, 5, 5);
  drawAxis(svg, M.left, M.top, M.left, M.top+PH);
  drawAxis(svg, M.left, M.top+PH, M.left+PW, M.top+PH);

  [15,20,25,30,35,40,45].forEach(v=>{
    const yy=yOf(v);
    drawAxis(svg,M.left-4,yy,M.left,yy);
    svgText(svg,v,M.left-8,yy,{anchor:'end',base:'middle',size:10});
  });
  [1,7,28,90,365].forEach(a=>{
    const xx=xOf(a);
    drawAxis(svg,xx,M.top+PH,xx,M.top+PH+4);
    svgText(svg,a,xx,M.top+PH+15,{size:10});
  });
  svgText(svg, 'Résistance prédite (MPa)', 14, M.top+PH/2, { anchor:'middle', base:'middle', size:11, transform:`rotate(-90,14,${M.top+PH/2})` });
  svgText(svg, 'Âge (jours, échelle log)', M.left+PW/2, H-4, { size:11 });

  const N = 100, pathsGB=[], pathsRidge=[];
  for (let i=0;i<=N;i++) {
    const age = Math.pow(10, Math.log10(365)*i/N);
    pathsGB.push(`${xOf(age)},${yOf(gbStrengthAtAge(age))}`);
    pathsRidge.push(`${xOf(age)},${yOf(ridgeStrengthAtAge(age))}`);
  }
  svg.appendChild(svgEl('path', { d:'M '+pathsGB.join(' L '), stroke:'var(--gb)', 'stroke-width':2.5, fill:'none' }));
  svg.appendChild(svgEl('path', { d:'M '+pathsRidge.join(' L '), stroke:'var(--ridge)', 'stroke-width':2.5, 'stroke-dasharray':'6 4', fill:'none' }));

  svgText(svg, 'GB — capture log(age)', M.left+PW+8, yOf(gbStrengthAtAge(365)), { anchor:'start', size:11, color:'var(--gb)' });
  svgText(svg, 'Ridge — droite (biais)', M.left+PW+8, yOf(ridgeStrengthAtAge(365))+14, { anchor:'start', size:11, color:'var(--ridge)' });

  container.appendChild(svg);
}

/* ═══════════════════════════════════════════════════
   EXPOSE GLOBAL HANDLERS (onclick in HTML)
═══════════════════════════════════════════════════ */
window.nextFold      = nextFold;
window.resetCV       = resetCV;
window.autoPlayCV    = autoPlayCV;
window.nextNestedStep= nextNestedStep;
window.prevNestedStep= prevNestedStep;
window.resetNested   = resetNested;
window.animLeakage   = animLeakage;
window.renderImportanceBars = renderImportanceBars;
window.animGridSearch  = animGridSearch;
window.resetGridSearch = resetGridSearch;
window.nextCartSplit   = nextCartSplit;
window.resetCartSplit  = resetCartSplit;
window.redrawBagging   = redrawBagging;
window.nextBoostStep   = nextBoostStep;
window.resetBoost      = resetBoost;

/* ═══════════════════════════════════════════════════
   INIT
═══════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNav();
  initComplexitySlider();
  initMetricSlider();
  initRidgeSlider();
  initCVGrid();
  renderCVFold(0);
  resetGridSearch();
  renderNestedStep();
  renderCartSplit(0);
  renderBaggingSVG();
  renderBoostStep(0);
  renderLeakStep(0);
  renderImportanceBars('gb');
  renderPDP();
  renderComparisonChart();
  renderR2Chart();
});

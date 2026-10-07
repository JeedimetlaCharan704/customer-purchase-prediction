const API = window.location.protocol.startsWith('http') ? window.location.origin : 'http://127.0.0.1:5001';
const $ = id => document.getElementById(id);
let charts = {};

function chartDefaults() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#b7c5d8',
          font: { size: 10 }
        }
      }
    },
    scales: {
      x: {
        ticks: { color: '#8191a9', font: { size: 10 } },
        grid: { color: 'rgba(255,255,255,.05)' }
      },
      y: {
        ticks: { color: '#8191a9', font: { size: 10 } },
        grid: { color: 'rgba(255,255,255,.05)' }
      }
    }
  };
}

function makeChart(id, type, labels, data, label) {
  if (charts[id]) charts[id].destroy();
  const canvas = $(id);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  charts[id] = new Chart(ctx, {
    type,
    data: {
      labels,
      datasets: [{
        label,
        data,
        borderWidth: 2,
        borderRadius: 8,
        tension: 0.35,
        backgroundColor: ['#6ee7ff', '#8b7cff', '#54e6a7', '#ffbd7a', '#ff7892', '#9f8cff'],
        borderColor: '#6ee7ff',
        fill: type === 'line',
        pointRadius: 3
      }]
    },
    options: chartDefaults()
  });
}

async function loadHealth() {
  try {
    const r = await fetch(API + '/api/health');
    const d = await r.json();
    $('health').className = 'health ' + (d.model_loaded ? 'ok' : 'bad');
    $('health').innerHTML = `<span></span>${d.model_loaded ? 'Model ready' : 'Model not trained'}`;
  } catch (e) {
    $('health').className = 'health bad';
    $('health').innerHTML = '<span></span>Backend offline';
  }
}

async function loadCharts() {
  try {
    const r = await fetch(API + '/api/charts');
    const d = await r.json();
    makeChart('purchaseChart', 'doughnut', d.purchase_distribution.labels, d.purchase_distribution.values, 'Sessions');
    makeChart('pagesChart', 'bar', d.page_visits.labels, d.page_visits.values, 'Average pages');
    makeChart('durationChart', 'bar', d.duration.labels, d.duration.values, 'Average seconds');
    makeChart('monthChart', 'line', d.monthly_purchase_rate.map(x => x.month), d.monthly_purchase_rate.map(x => x.rate), 'Purchase rate %');
  } catch (e) {
    console.warn('Charts unavailable', e);
  }
}

function formData() {
  const fd = new FormData($('predictForm'));
  const o = {};
  for (const [k, v] of fd.entries()) {
    o[k] = ['Month', 'VisitorType'].includes(k) ? v : Number(v);
  }
  o.Weekend = Number(o.Weekend);
  return o;
}

$('predictForm').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = document.querySelector('.predict-btn');
  btn.disabled = true;
  btn.innerHTML = '⏳ Running model...';
  $('resultMessage').textContent = 'Applying feature engineering, scaling, PCA and Random Forest...';

  try {
    const r = await fetch(API + '/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData())
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Prediction failed');

    const p = d.analytics.probability;
    const isPurchase = d.prediction === 1;

    $('resultLabel').textContent = d.label;
    $('resultLabel').style.color = isPurchase ? '#54e6a7' : '#ff7892';
    $('resultIcon').textContent = isPurchase ? '✓' : '×';
    $('resultIcon').style.color = isPurchase ? '#54e6a7' : '#ff7892';

    $('probability').textContent = p.toFixed(1) + '%';
    $('probBar').style.width = Math.min(Math.max(p, 2), 100) + '%';

    const confScore = isPurchase ? p : (100 - p);
    $('confidence').textContent = confScore >= 75 ? `High (${confScore.toFixed(0)}%)` : (confScore >= 50 ? `Moderate (${confScore.toFixed(0)}%)` : `Low (${confScore.toFixed(0)}%)`);

    $('totalPages').textContent = d.analytics.total_pages;
    $('totalDuration').textContent = d.analytics.total_duration.toFixed(1) + 's';
    $('avgProduct').textContent = d.analytics.average_product_time.toFixed(1) + 's';
    $('engagement').textContent = d.analytics.engagement_score + '%';
    $('resultMessage').textContent = isPurchase
      ? 'The trained model predicts a purchase for this session.'
      : 'The trained model predicts no purchase for this session.';
  } catch (err) {
    $('resultLabel').textContent = 'Prediction unavailable';
    $('resultLabel').style.color = '#ff7892';
    $('resultIcon').textContent = '!';
    $('resultIcon').style.color = '#ff7892';
    $('resultMessage').textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>✦</span> Predict Purchase';
  }
});

$('resetBtn').addEventListener('click', () => {
  $('predictForm').reset();
  $('resultLabel').textContent = 'Ready for prediction';
  $('resultLabel').style.color = '';
  $('resultIcon').textContent = '◌';
  $('resultIcon').style.color = '';
  $('probability').textContent = '—';
  $('probBar').style.width = '0';
  $('confidence').textContent = 'Waiting';
  ['totalPages', 'totalDuration', 'avgProduct', 'engagement'].forEach(id => $(id).textContent = '—');
  $('resultMessage').textContent = 'Enter a customer session and run the trained model.';
});

loadHealth();
loadCharts();

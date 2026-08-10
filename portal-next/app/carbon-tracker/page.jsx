'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const gradeColor = g =>
  g === 'A+' || g === 'A' ? 'success' :
  g === 'B'               ? 'primary' :
  g === 'C'               ? 'warning' : 'danger';

const priorityColor = p =>
  p === 'high' ? 'danger' : p === 'medium' ? 'warning' : 'secondary';

function KpiCard({ label, value, unit = '', sub = '', color = 'primary' }) {
  return (
    <div className={`card border-${color} mb-3`}>
      <div className="card-body py-2 px-3">
        <div className={`fw-bold text-${color} fs-5`}>{value}{unit && <small className="fs-6 ms-1">{unit}</small>}</div>
        <div className="text-muted small">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: '0.72rem' }}>{sub}</div>}
      </div>
    </div>
  );
}

function ScoreGauge({ score, grade }) {
  const color = gradeColor(grade);
  const pct   = Math.min(100, Math.max(0, score));
  return (
    <div className="text-center mb-3">
      <div className={`display-4 fw-bold text-${color}`}>{grade}</div>
      <div className="text-muted small mb-1">Efficiency Grade</div>
      <div className="progress" style={{ height: 18 }}>
        <div
          className={`progress-bar bg-${color}`}
          style={{ width: `${pct}%` }}
          title={`${score}/100`}
        >
          {score}/100
        </div>
      </div>
    </div>
  );
}

export default function CarbonTrackerPage() {
  const [ov,   setOv]   = useState(null);
  const [bk,   setBk]   = useState(null);
  const [df,   setDf]   = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [err,  setErr]  = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/carbon-tracker/overview`).then(r => r.json()),
      fetch(`${API}/api/carbon-tracker/breakdown`).then(r => r.json()),
      fetch(`${API}/api/carbon-tracker/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err)  return <div className="alert alert-danger m-4">Error: {err}</div>;
  if (!ov)  return <div className="text-center mt-5 text-muted">Loading Carbon Tracker…</div>;

  const kpis = ov.kpis || {};
  const score = ov.efficiency_breakdown || {};
  const equiv = ov.equivalences || {};
  const power = ov.current_power_detail || {};
  const recs  = ov.recommendations || [];

  return (
    <div className="container-fluid p-4">
      <h2 className="mb-1">🌿 Green AI / Carbon Footprint</h2>
      <p className="text-muted small mb-3">
        Live power estimation · Annual CO₂ projection · Sustainability scoring
        — Region: <strong>{ov.region}</strong> ({ov.carbon_intensity_kg_per_kwh} kg CO₂/kWh)
        · Generated {ov.generated_at ? ov.generated_at.slice(0, 19) + 'Z' : '—'}
      </p>

      <ul className="nav nav-tabs mb-3">
        {['overview','breakdown','equivalences','recommendations','definitions'].map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active' : ''}`}
              onClick={() => setTab(t)}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW TAB ──────────────────────────────────────────── */}
      {tab === 'overview' && (
        <>
          <div className="row g-3 mb-3">
            <div className="col-md-2">
              <div className="card h-100 text-center border-0 bg-light py-3">
                <ScoreGauge score={score.total || 0} grade={kpis.efficiency_grade || '—'} />
                <div className="small text-muted px-2">
                  Region: {score.region_score}/50 &nbsp;|&nbsp;
                  Efficiency: {score.efficiency_score}/30 &nbsp;|&nbsp;
                  Volume: {score.volume_score}/20
                </div>
              </div>
            </div>
            <div className="col-md-10">
              <div className="row g-2">
                <div className="col-6 col-md-3">
                  <KpiCard label="Current Power" value={kpis.current_power_w} unit="W" color="primary" />
                </div>
                <div className="col-6 col-md-3">
                  <KpiCard label="Daily Energy" value={kpis.daily_kwh} unit="kWh" color="info" />
                </div>
                <div className="col-6 col-md-3">
                  <KpiCard label="Annual Energy" value={kpis.annual_kwh} unit="kWh" color="warning" />
                </div>
                <div className="col-6 col-md-3">
                  <KpiCard label="Annual CO₂" value={kpis.annual_co2_kg} unit=" kg" color="danger" />
                </div>
                <div className="col-6 col-md-3">
                  <KpiCard label="Offset Cost/Year" value={`$${kpis.offset_cost_usd_yr}`} color="secondary" />
                </div>
                <div className="col-6 col-md-3">
                  <KpiCard label="Training CO₂" value={kpis.total_training_co2_kg} unit=" kg" sub={`${kpis.total_training_kwh} kWh total`} color="dark" />
                </div>
                <div className="col-6 col-md-3">
                  <KpiCard label="kWh / Prediction" value={kpis.kwh_per_prediction} color="success" />
                </div>
                <div className="col-6 col-md-3">
                  <KpiCard label="g CO₂ / Prediction" value={kpis.co2_g_per_prediction} unit=" g" color="success" />
                </div>
              </div>
            </div>
          </div>

          {/* Live Power Breakdown */}
          <div className="card mb-3">
            <div className="card-header fw-bold">⚡ Live Power Breakdown ({power.total_w} W total)</div>
            <div className="card-body">
              <div className="row g-2 mb-2">
                {[
                  { label: 'CPU', w: power.cpu_w,   note: `${power.cpu_pct}% load` },
                  { label: 'GPU', w: power.gpu_w,   note: 'inference mode' },
                  { label: 'RAM', w: power.ram_w,   note: `${power.memory_gb} GB used` },
                  { label: 'Other', w: power.other_w, note: 'SSD + network' },
                ].map(c => (
                  <div key={c.label} className="col-6 col-md-3">
                    <div className="p-2 bg-light rounded text-center">
                      <div className="fw-bold">{c.w} W</div>
                      <div className="small">{c.label}</div>
                      <div className="text-muted" style={{ fontSize: '0.72rem' }}>{c.note}</div>
                    </div>
                  </div>
                ))}
              </div>
              {/* Stacked bar */}
              <div className="progress" style={{ height: 20 }}>
                {[
                  { key: 'CPU',   val: power.cpu_w,   color: 'primary' },
                  { key: 'GPU',   val: power.gpu_w,   color: 'warning' },
                  { key: 'RAM',   val: power.ram_w,   color: 'info'    },
                  { key: 'Other', val: power.other_w, color: 'secondary'},
                ].map(c => (
                  <div
                    key={c.key}
                    className={`progress-bar bg-${c.color}`}
                    style={{ width: `${(c.val / power.total_w * 100).toFixed(1)}%` }}
                    title={`${c.key}: ${c.val}W`}
                  >
                    {c.key}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Training summary */}
          <div className="card mb-3">
            <div className="card-header fw-bold">🏋️ Training Footprint ({(ov.training_summary || {}).total_runs || 0} runs)</div>
            <div className="card-body">
              <div className="row g-2">
                <div className="col-md-4 text-center">
                  <div className="fs-4 fw-bold text-dark">{(ov.training_summary || {}).total_kwh || 0}</div>
                  <div className="text-muted small">kWh consumed</div>
                </div>
                <div className="col-md-4 text-center">
                  <div className="fs-4 fw-bold text-danger">{(ov.training_summary || {}).total_co2_kg || 0}</div>
                  <div className="text-muted small">kg CO₂ emitted</div>
                </div>
                <div className="col-md-4 text-center">
                  <div className="fs-4 fw-bold text-success">
                    {((ov.training_summary || {}).total_co2_kg || 0) < 1 ? '<1' : Math.round((ov.training_summary || {}).total_co2_kg)}
                  </div>
                  <div className="text-muted small">kg CO₂ total (all model runs)</div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── BREAKDOWN TAB ─────────────────────────────────────────── */}
      {tab === 'breakdown' && bk && (
        <>
          {/* Region comparison */}
          <div className="card mb-3">
            <div className="card-header fw-bold">🌍 Region CO₂ Comparison (same workload)</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark">
                  <tr>
                    <th>Region</th>
                    <th>kg CO₂/kWh</th>
                    <th>Annual CO₂ (kg)</th>
                    <th>Offset Cost/yr</th>
                    <th>Note</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.region_comparison || []).map(r => (
                    <tr key={r.region} className={r.active ? 'table-info fw-bold' : ''}>
                      <td>{r.label}{r.active && <span className="badge bg-primary ms-1">active</span>}</td>
                      <td>{r.kg_co2_kwh}</td>
                      <td>{r.annual_co2_kg}</td>
                      <td>${r.offset_cost_usd}</td>
                      <td className="text-muted small">{r.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Per-model CO2 */}
          <div className="card mb-3">
            <div className="card-header fw-bold">🤖 Training CO₂ by Model Type</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark">
                  <tr>
                    <th>Model Type</th>
                    <th>Runs</th>
                    <th>Total kWh</th>
                    <th>Total CO₂ (kg)</th>
                    <th>Avg CO₂/Run (g)</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.model_co2_table || []).map(m => (
                    <tr key={m.model_type}>
                      <td>{m.model_type}</td>
                      <td>{m.runs}</td>
                      <td>{m.total_kwh}</td>
                      <td>{m.total_co2_kg}</td>
                      <td>{m.avg_co2_g_per_run}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Prediction scenarios */}
          <div className="card mb-3">
            <div className="card-header fw-bold">📊 Prediction Volume Scenarios</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark">
                  <tr>
                    <th>Daily Predictions</th>
                    <th>Annual kWh</th>
                    <th>Annual CO₂ (kg)</th>
                    <th>Offset Cost/yr</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.prediction_scenarios || []).map(s => (
                    <tr key={s.daily_predictions}>
                      <td>{s.daily_predictions.toLocaleString()}</td>
                      <td>{s.annual_kwh}</td>
                      <td>{s.annual_co2_kg}</td>
                      <td>${s.offset_cost_usd}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Carbon offset tiers */}
          <div className="card mb-3">
            <div className="card-header fw-bold">💸 Carbon Offset Market Tiers</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark">
                  <tr><th>Provider</th><th>$/ton CO₂</th><th>Note</th></tr>
                </thead>
                <tbody>
                  {(bk.carbon_offset_tiers || []).map(t => (
                    <tr key={t.provider}>
                      <td>{t.provider}</td>
                      <td>${t.usd_per_ton}</td>
                      <td className="text-muted small">{t.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── EQUIVALENCES TAB ──────────────────────────────────────── */}
      {tab === 'equivalences' && (
        <div className="card">
          <div className="card-header fw-bold">🌎 Annual CO₂ Equivalences ({kpis.annual_co2_kg} kg CO₂/year)</div>
          <div className="card-body">
            <div className="row g-3">
              {[
                { label: '🚗 Car driving',      value: `${equiv.car_driving_km?.toLocaleString() || '—'} km`,   note: 'at 210 g CO₂/km' },
                { label: '🌳 Trees to offset',  value: `${equiv.trees_to_offset || '—'}`,                        note: 'trees absorbing 21 kg CO₂/year each' },
                { label: '🏠 Homes powered',    value: `${((equiv.homes_powered_fraction || 0) * 100).toFixed(2)}%`, note: 'fraction of one home (7,500 kWh/year)' },
                { label: '📱 Smartphones',      value: `${equiv.smartphones_charged?.toLocaleString() || '—'}`,  note: 'fully charged at 12 Wh/charge' },
                { label: '✈️ Flights NYC→LON',  value: `${equiv.flights_nyc_lon || '—'}`,                        note: 'per flight ≈ 1,100 kg CO₂' },
              ].map(e => (
                <div key={e.label} className="col-md-4">
                  <div className="card h-100 border-success">
                    <div className="card-body">
                      <div className="fs-4 fw-bold text-success">{e.value}</div>
                      <div className="fw-bold">{e.label}</div>
                      <div className="text-muted small">{e.note}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── RECOMMENDATIONS TAB ───────────────────────────────────── */}
      {tab === 'recommendations' && (
        <div className="card">
          <div className="card-header fw-bold">💡 Green AI Recommendations</div>
          <div className="card-body">
            {recs.length === 0 && <div className="text-muted">No recommendations at this time.</div>}
            {recs.map((r, i) => (
              <div key={i} className={`alert alert-${priorityColor(r.priority)} mb-2`}>
                <span className={`badge bg-${priorityColor(r.priority)} me-2`}>{r.priority?.toUpperCase()}</span>
                {r.action}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── DEFINITIONS TAB ───────────────────────────────────────── */}
      {tab === 'definitions' && df && (
        <>
          <div className="card mb-3">
            <div className="card-header fw-bold">📖 Glossary</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark">
                  <tr><th style={{ width: '25%' }}>Term</th><th>Definition</th><th>Source</th></tr>
                </thead>
                <tbody>
                  {(df.glossary || []).map(g => (
                    <tr key={g.term}>
                      <td className="fw-bold">{g.term}</td>
                      <td className="small">{g.definition}</td>
                      <td className="text-muted small">{g.source}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="card mb-3">
            <div className="card-header fw-bold">📏 Metrics Reference</div>
            <div className="card-body">
              <table className="table table-sm mb-0">
                <tbody>
                  {Object.entries(df.metrics_reference || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="text-muted small fw-bold" style={{ width: '40%' }}>{k.replace(/_/g, ' ')}</td>
                      <td className="small">{String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="card">
            <div className="card-header fw-bold">📋 Standards</div>
            <div className="card-body">
              <ul className="mb-0">
                {(df.standards || []).map((s, i) => <li key={i} className="small">{s}</li>)}
              </ul>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

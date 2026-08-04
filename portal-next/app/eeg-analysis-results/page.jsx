'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const DISEASE_COLORS = {
  epilepsy: '#ef4444',
  sleep_disorder: '#3b82f6',
  depression: '#f59e0b',
  parkinsons: '#8b5cf6',
  alzheimers: '#10b981',
};

const QUALITY_COLORS = {
  Excellent: '#22c55e',
  Good: '#3b82f6',
  Fair: '#f59e0b',
  Poor: '#ef4444',
};

const TIER_COLORS = {
  High: '#22c55e',
  Medium: '#f59e0b',
  Low: '#ef4444',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 col-lg-2 mb-2">
      <div className="card text-center shadow-sm border-0">
        <div className="card-body py-2 px-1">
          <div className="h4 mb-0 fw-bold" style={{ color: color || '#3b82f6' }}>{value ?? '—'}</div>
          <div className="text-muted" style={{ fontSize: '0.75rem' }}>{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function MiniBar({ value, max, color }) {
  const pct = Math.min(100, Math.max(0, (value / (max || 1)) * 100));
  return (
    <div className="progress" style={{ height: 8, minWidth: 60 }}>
      <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color || '#3b82f6' }} />
    </div>
  );
}

function ConfidenceMeter({ value }) {
  const pct = Math.round((value || 0) * 100);
  const color = pct >= 75 ? '#22c55e' : pct >= 50 ? '#f59e0b' : '#ef4444';
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="small fw-semibold" style={{ color, minWidth: 36 }}>{pct}%</span>
    </div>
  );
}

export default function EEGAnalysisResultsDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [diseaseFilter, setDiseaseFilter] = useState('all');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/eeg-analysis-results/overview`).then(r => r.json()),
      fetch(`${API}/api/eeg-analysis-results/breakdown`).then(r => r.json()),
      fetch(`${API}/api/eeg-analysis-results/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading EEG analysis results…</div>;

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'disease', label: '🧠 By Disease' },
    { id: 'quality', label: '📶 Signal Quality' },
    { id: 'records', label: '📋 Records' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const k = ov.kpis;
  const analyses = (bd?.analyses || []);
  const filtered = diseaseFilter === 'all' ? analyses : analyses.filter(a => a.disease === diseaseFilter);

  return (
    <div className="p-3">
      <h3>🔬 EEG Analysis Results Dashboard</h3>
      <p className="text-muted">
        AI model outputs across {k.diseases_covered} diseases — {k.total_analyses} analyses, {k.total_patients} patients,
        avg confidence {Math.round((k.avg_confidence || 0) * 100)}%
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <div>
          <div className="row mb-3">
            <KPI label="Total Analyses" value={k.total_analyses} color="#3b82f6" />
            <KPI label="Patients" value={k.total_patients} color="#6366f1" />
            <KPI label="Diseases Covered" value={k.diseases_covered} color="#8b5cf6" />
            <KPI label="Avg Confidence" value={`${Math.round((k.avg_confidence || 0) * 100)}%`} color="#22c55e" />
            <KPI label="High Confidence" value={k.high_confidence_count} color="#10b981" sub="≥ 75%" />
            <KPI label="Low Confidence" value={k.low_confidence_count} color="#ef4444" sub="< 50%" />
            <KPI label="Excellent Quality" value={`${k.signal_quality_excellent_pct}%`} color="#22c55e" />
            <KPI label="Poor Quality" value={`${k.signal_quality_poor_pct}%`} color="#ef4444" />
          </div>

          <div className="row mb-3">
            {/* Disease Distribution */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-semibold mb-2">Disease Distribution</h6>
                  <table className="table table-sm table-hover">
                    <thead><tr><th>Disease</th><th>Count</th><th>Share</th></tr></thead>
                    <tbody>
                      {(ov.disease_dist || []).map(d => (
                        <tr key={d.disease}>
                          <td>
                            <span className="badge me-1" style={{ backgroundColor: DISEASE_COLORS[d.disease] || '#6b7280' }}>
                              {d.disease.replace('_', ' ')}
                            </span>
                          </td>
                          <td className="fw-bold">{d.count}</td>
                          <td style={{ minWidth: 100 }}>
                            <MiniBar value={d.count} max={k.total_analyses} color={DISEASE_COLORS[d.disease]} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Confidence Tiers */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-semibold mb-2">Confidence Tiers</h6>
                  <table className="table table-sm">
                    <thead><tr><th>Tier</th><th>Count</th><th>Bar</th></tr></thead>
                    <tbody>
                      {(ov.confidence_tiers || []).map(t => (
                        <tr key={t.tier}>
                          <td>
                            <span className="badge" style={{ backgroundColor: TIER_COLORS[t.tier] || '#6b7280' }}>
                              {t.tier}
                            </span>
                          </td>
                          <td className="fw-bold">{t.count}</td>
                          <td style={{ minWidth: 100 }}>
                            <MiniBar value={t.count} max={k.total_analyses} color={TIER_COLORS[t.tier]} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <hr />
                  <h6 className="fw-semibold mb-2">Predicted Label Distribution</h6>
                  <table className="table table-sm">
                    <thead><tr><th>Label</th><th>Count</th></tr></thead>
                    <tbody>
                      {(ov.label_dist || []).map(l => (
                        <tr key={l.label}>
                          <td>{l.label}</td>
                          <td className="fw-bold">{l.count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Monthly Trend */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-semibold mb-2">Monthly Analysis Trend</h6>
              <div className="table-responsive">
                <table className="table table-sm">
                  <thead>
                    <tr><th>Month</th><th>Analyses</th><th>Avg Confidence</th><th>Volume Bar</th></tr>
                  </thead>
                  <tbody>
                    {(ov.monthly_trend || []).map(m => (
                      <tr key={m.month}>
                        <td className="fw-semibold">{m.month}</td>
                        <td>{m.analyses}</td>
                        <td>
                          <ConfidenceMeter value={m.avg_confidence} />
                        </td>
                        <td style={{ minWidth: 120 }}>
                          <MiniBar value={m.analyses} max={Math.max(...(ov.monthly_trend || []).map(x => x.analyses))} color="#3b82f6" />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Disease Tab ── */}
      {tab === 'disease' && (
        <div>
          <h6 className="fw-semibold mb-3">Confidence by Disease</h6>
          <div className="row">
            {(ov.disease_confidence || []).map(d => {
              const color = DISEASE_COLORS[d.disease] || '#6b7280';
              const avgPct = Math.round((d.avg_confidence || 0) * 100);
              const minPct = Math.round((d.min_confidence || 0) * 100);
              const maxPct = Math.round((d.max_confidence || 0) * 100);
              return (
                <div key={d.disease} className="col-md-6 col-lg-4 mb-3">
                  <div className="card shadow-sm h-100">
                    <div className="card-body">
                      <div className="d-flex align-items-center gap-2 mb-2">
                        <span className="badge" style={{ backgroundColor: color, fontSize: '0.85rem' }}>
                          {d.disease.replace('_', ' ')}
                        </span>
                        <span className="text-muted small">{d.count} analyses</span>
                      </div>
                      <div className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>Avg Confidence</span>
                          <span className="fw-bold">{avgPct}%</span>
                        </div>
                        <div className="progress" style={{ height: 12 }}>
                          <div className="progress-bar" style={{ width: `${avgPct}%`, backgroundColor: color }} />
                        </div>
                      </div>
                      <div className="d-flex justify-content-between small text-muted mt-2">
                        <span>Min: {minPct}%</span>
                        <span>Max: {maxPct}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Signal Quality Tab ── */}
      {tab === 'quality' && (
        <div>
          <h6 className="fw-semibold mb-3">Signal Quality Distribution — {k.total_analyses} analyses</h6>
          <div className="row mb-3">
            {(ov.quality_dist || []).map(q => {
              const color = QUALITY_COLORS[q.quality] || '#6b7280';
              const pct = Math.round((q.count / k.total_analyses) * 100);
              return (
                <div key={q.quality} className="col-6 col-md-3 mb-3">
                  <div className="card shadow-sm text-center">
                    <div className="card-body py-3">
                      <div className="h3 fw-bold mb-0" style={{ color }}>{q.count}</div>
                      <div className="small text-muted">{q.quality}</div>
                      <div className="small text-muted">{pct}% of total</div>
                      <div className="progress mt-2" style={{ height: 8 }}>
                        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="alert alert-info">
            <strong>Signal Quality Context:</strong> {k.signal_quality_excellent_pct}% excellent recordings support high-confidence classification;
            {k.signal_quality_poor_pct}% poor recordings may require re-acquisition or artifact cleaning before clinical use.
          </div>
        </div>
      )}

      {/* ── Records Tab ── */}
      {tab === 'records' && (
        <div>
          <div className="d-flex align-items-center gap-2 mb-3 flex-wrap">
            <span className="fw-semibold">Filter by disease:</span>
            {['all', 'epilepsy', 'sleep_disorder', 'depression', 'parkinsons', 'alzheimers'].map(d => (
              <button
                key={d}
                className={`btn btn-sm ${diseaseFilter === d ? 'btn-primary' : 'btn-outline-secondary'}`}
                onClick={() => setDiseaseFilter(d)}
              >
                {d === 'all' ? 'All' : d.replace('_', ' ')}
              </button>
            ))}
            <span className="text-muted small ms-2">{filtered.length} records</span>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-bordered">
              <thead className="table-light">
                <tr>
                  <th>ID</th>
                  <th>Patient</th>
                  <th>Disease</th>
                  <th>Predicted Label</th>
                  <th>Confidence</th>
                  <th>Signal Quality</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 50).map(a => {
                  const confPct = Math.round((a.confidence || 0) * 100);
                  const confColor = confPct >= 75 ? '#22c55e' : confPct >= 50 ? '#f59e0b' : '#ef4444';
                  const qualColor = QUALITY_COLORS[a.signal_quality] || '#6b7280';
                  return (
                    <tr key={a.id}>
                      <td className="text-muted small">#{a.id}</td>
                      <td className="fw-semibold">{a.patient_id}</td>
                      <td>
                        <span className="badge" style={{ backgroundColor: DISEASE_COLORS[a.disease] || '#6b7280' }}>
                          {(a.disease || '').replace('_', ' ')}
                        </span>
                      </td>
                      <td>{a.predicted_label}</td>
                      <td>
                        <div className="d-flex align-items-center gap-1">
                          <div className="progress" style={{ height: 8, width: 60 }}>
                            <div className="progress-bar" style={{ width: `${confPct}%`, backgroundColor: confColor }} />
                          </div>
                          <span className="small fw-bold" style={{ color: confColor }}>{confPct}%</span>
                        </div>
                      </td>
                      <td>
                        <span className="badge" style={{ backgroundColor: qualColor }}>{a.signal_quality}</span>
                      </td>
                      <td className="small text-muted">{(a.created_at || '').slice(0, 10)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {filtered.length > 50 && (
              <div className="text-muted small">Showing 50 of {filtered.length} records</div>
            )}
          </div>
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs && (
        <div>
          <h6 className="fw-semibold mb-3">Field Definitions</h6>
          <table className="table table-sm table-bordered">
            <thead className="table-light">
              <tr><th>Field</th><th>Description</th></tr>
            </thead>
            <tbody>
              {Object.entries(defs.fields || {}).map(([field, desc]) => (
                <tr key={field}>
                  <td className="fw-semibold text-nowrap"><code>{field}</code></td>
                  <td>{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h6 className="fw-semibold mt-3 mb-2">Diseases</h6>
          <div className="row">
            {(defs.diseases || []).map(d => (
              <div key={d.name} className="col-md-4 mb-2">
                <div className="card shadow-sm">
                  <div className="card-body py-2">
                    <span className="badge mb-1" style={{ backgroundColor: DISEASE_COLORS[d.name] || '#6b7280' }}>
                      {d.name}
                    </span>
                    <div className="small text-muted">{d.description}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

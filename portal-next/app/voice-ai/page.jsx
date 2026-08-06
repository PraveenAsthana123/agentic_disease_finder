'use client';
import { useEffect, useState } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KpiCard({ label, value, color }) {
  return (
    <div className="card text-center shadow-sm h-100">
      <div className="card-body py-3">
        <div className="fs-3 fw-bold" style={{ color: color || '#6366f1' }}>{value}</div>
        <div className="small fw-semibold text-muted">{label}</div>
      </div>
    </div>
  );
}

function ScoreBar({ pct, color = '#6366f1' }) {
  return (
    <div style={{ background: '#e5e7eb', borderRadius: 4, height: 8, width: '100%' }}>
      <div style={{ width: `${Math.min(100, pct * 100)}%`, background: color, borderRadius: 4, height: 8 }} />
    </div>
  );
}

const LEVEL_COLOR = { normal: '#10b981', Average: '#10b981', mild: '#f59e0b', Mild: '#f59e0b', moderate: '#f97316', Moderate: '#f97316', severe: '#ef4444', Severe: '#ef4444' };
const levelBadge = level => {
  const c = LEVEL_COLOR[level] || '#6b7280';
  return <span className="badge" style={{ background: c }}>{level}</span>;
};

export default function VoiceAIDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [sort, setSort] = useState({ col: 'pct', dir: -1 });
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/voice-ai/overview`).then(r => r.json()),
      fetch(`${API}/api/voice-ai/breakdown`).then(r => r.json()),
      fetch(`${API}/api/voice-ai/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;
  if (!ov) return <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}><div className="spinner-border text-primary" /></div>;

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'assessments', label: '🎙️ Assessments' },
    { id: 'patients', label: '👥 Per Patient' },
    { id: 'pipeline', label: '⚙️ Pipeline' },
    { id: 'definitions', label: '📚 Definitions' },
  ];

  const kpis = ov.kpis || [];

  // Filtered assessment inventory
  const inventory = bd?.assessment_inventory || [];
  const filtered = search
    ? inventory.filter(a =>
        a.patient_id?.toLowerCase().includes(search.toLowerCase()) ||
        a.instrument_label?.toLowerCase().includes(search.toLowerCase()) ||
        a.domain?.toLowerCase().includes(search.toLowerCase()) ||
        a.level?.toLowerCase().includes(search.toLowerCase())
      )
    : inventory;
  const sorted = [...filtered].sort((a, b) => sort.dir * ((b[sort.col] ?? 0) - (a[sort.col] ?? 0)));

  const toggleSort = col => setSort(s => ({ col, dir: s.col === col ? -s.dir : -1 }));
  const sortIcon = col => sort.col === col ? (sort.dir === -1 ? '▼' : '▲') : '⇅';

  const instrColors = ['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ef4444'];

  return (
    <div className="container-fluid py-4">
      <h2 className="fw-bold mb-1">🎙️ Voice AI Assessment Dashboard</h2>
      <p className="text-muted mb-3">
        {ov.total_assessments} assessments · {ov.patients_assessed} patients · {ov.instruments_used} instruments ·
        mean score {(ov.mean_normalized_score * 100).toFixed(1)}% · abnormal rate {(ov.abnormal_rate * 100).toFixed(1)}%
      </p>

      {/* KPI row */}
      <div className="row g-3 mb-4">
        {kpis.map((k, i) => (
          <div key={i} className="col-6 col-md-3 col-xl-2">
            <KpiCard label={k.label} value={k.value} color={k.color} />
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW tab ── */}
      {tab === 'overview' && (
        <div>
          <div className="row g-4">
            {/* Instrument scores */}
            <div className="col-lg-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Instrument Score Comparison</div>
                <div className="card-body">
                  {(ov.instrument_scores || []).map((ins, i) => (
                    <div key={ins.instrument} className="mb-3">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="small fw-semibold">{ins.label}</span>
                        <span className="small text-muted">{(ins.mean_score * 100).toFixed(1)}% (n={ins.n})</span>
                      </div>
                      <ScoreBar pct={ins.mean_score} color={instrColors[i % instrColors.length]} />
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Severity distribution */}
            <div className="col-lg-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Severity Distribution</div>
                <div className="card-body">
                  {(ov.severity_distribution || []).map(sv => (
                    <div key={sv.level} className="d-flex justify-content-between align-items-center mb-2">
                      <span className="badge" style={{ background: sv.color, minWidth: 80 }}>{sv.level}</span>
                      <span className="fw-bold">{sv.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Instrument distribution */}
            <div className="col-lg-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Instruments Used</div>
                <div className="card-body">
                  {(ov.instrument_distribution || []).map((ins, i) => (
                    <div key={ins.instrument} className="mb-2">
                      <div className="d-flex justify-content-between">
                        <span className="small">{ins.label}</span>
                        <span className="badge" style={{ background: instrColors[i % instrColors.length] }}>{ins.count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Instrument stats table */}
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Instrument Statistics</div>
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Instrument</th>
                        <th>Domain</th>
                        <th>Count</th>
                        <th>Mean Score</th>
                        <th>Normal</th>
                        <th>Mild</th>
                        <th>Moderate</th>
                        <th>Severe</th>
                        <th>Alerts</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bd?.instrument_stats || []).map(ins => (
                        <tr key={ins.instrument}>
                          <td><span className="fw-semibold">{ins.instrument}</span><br /><span className="text-muted small">{ins.label}</span></td>
                          <td>{ins.domain}</td>
                          <td>{ins.count}</td>
                          <td>
                            <ScoreBar pct={ins.mean_score} color="#6366f1" />
                            <span className="small">{(ins.mean_score * 100).toFixed(1)}%</span>
                          </td>
                          <td><span className="badge bg-success">{ins.normal}</span></td>
                          <td><span className="badge bg-warning text-dark">{ins.mild}</span></td>
                          <td><span className="badge bg-danger">{ins.moderate}</span></td>
                          <td><span className="badge bg-secondary">{ins.severe}</span></td>
                          <td>{ins.alerts > 0 ? <span className="badge bg-danger">{ins.alerts}</span> : <span className="text-muted">—</span>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Clinical alerts */}
            {(bd?.clinical_alerts || []).length > 0 && (
              <div className="col-12">
                <div className="card shadow-sm border-warning">
                  <div className="card-header fw-semibold text-warning">⚠️ Clinical Alerts ({bd.clinical_alerts.length})</div>
                  <div className="card-body p-0">
                    <div className="table-responsive">
                      <table className="table table-sm mb-0">
                        <thead className="table-light">
                          <tr><th>Patient</th><th>Instrument</th><th>Score</th><th>Level</th><th>Alert</th><th>Examiner</th></tr>
                        </thead>
                        <tbody>
                          {bd.clinical_alerts.map((a, i) => (
                            <tr key={i}>
                              <td className="fw-semibold">{a.patient_id}</td>
                              <td>{a.instrument_label}</td>
                              <td>{a.score}/{a.max_score} ({a.pct?.toFixed(1)}%)</td>
                              <td>{levelBadge(a.level)}</td>
                              <td><span className="text-danger small">{a.alert}</span></td>
                              <td className="text-muted small">{a.examiner}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── ASSESSMENTS tab ── */}
      {tab === 'assessments' && (
        <div>
          <div className="mb-3">
            <input
              className="form-control"
              style={{ maxWidth: 360 }}
              placeholder="Search patient, instrument, domain, level…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Assessment Inventory ({sorted.length} records)</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th>
                    <th>Instrument</th>
                    <th>Domain</th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('score')}>Score {sortIcon('score')}</th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('pct')}>% {sortIcon('pct')}</th>
                    <th>Level</th>
                    <th>Examiner</th>
                    <th>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.slice(0, 100).map((a, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{a.patient_id}</td>
                      <td>{a.instrument_label || a.instrument}</td>
                      <td className="text-muted small">{a.domain}</td>
                      <td>{a.score}/{a.max_score}</td>
                      <td>
                        <div>{a.pct?.toFixed(1)}%</div>
                        <ScoreBar pct={a.pct / 100} color={LEVEL_COLOR[a.level] || '#6366f1'} />
                      </td>
                      <td>{levelBadge(a.level)}</td>
                      <td className="text-muted small">{a.examiner}</td>
                      <td className="text-muted small">{a.created_at?.slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {sorted.length > 100 && (
              <div className="card-footer text-muted small">Showing 100 of {sorted.length} records</div>
            )}
          </div>
        </div>
      )}

      {/* ── PER PATIENT tab ── */}
      {tab === 'patients' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Patient Profiles ({(bd?.patient_profiles || []).length} patients)</div>
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Assessments</th>
                  <th>Instruments</th>
                  <th>Mean Score</th>
                  <th>Worst Level</th>
                  <th>Alerts</th>
                  <th>Latest</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.patient_profiles || []).map((p, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{p.patient_id}</td>
                    <td>{p.assessment_count}</td>
                    <td>
                      {(p.instruments || []).map(ins => (
                        <span key={ins} className="badge bg-secondary me-1">{ins}</span>
                      ))}
                    </td>
                    <td>
                      <div>{(p.mean_score * 100).toFixed(1)}%</div>
                      <ScoreBar pct={p.mean_score} color="#6366f1" />
                    </td>
                    <td>{levelBadge(p.worst_level)}</td>
                    <td>
                      {p.alert_count > 0
                        ? <span className="badge bg-danger">{p.alert_count}</span>
                        : <span className="text-muted">—</span>}
                    </td>
                    <td className="text-muted small">{p.latest_date?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── PIPELINE tab ── */}
      {tab === 'pipeline' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Voice AI Pipeline Events ({(bd?.pipeline_events || []).length})</div>
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Event</th>
                  <th>Component</th>
                  <th>Status</th>
                  <th>Duration (ms)</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.pipeline_events || []).map((e, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{e.event_type || e.event}</td>
                    <td className="text-muted small">{e.component}</td>
                    <td>
                      <span className={`badge ${e.status === 'success' || e.status === 'completed' ? 'bg-success' : e.status === 'error' ? 'bg-danger' : 'bg-secondary'}`}>
                        {e.status}
                      </span>
                    </td>
                    <td>{e.duration_ms != null ? e.duration_ms : '—'}</td>
                    <td className="text-muted small">{(e.timestamp || e.created_at || '').slice(0, 16)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS tab ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-4">
          {/* Concepts */}
          <div className="col-lg-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Concepts ({(defs.concepts || []).length})</div>
              <div className="list-group list-group-flush">
                {(defs.concepts || []).map((c, i) => (
                  <div key={i} className="list-group-item">
                    <div className="fw-semibold">{c.term}</div>
                    <div className="text-muted small">{c.definition}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Assessment instruments */}
          <div className="col-lg-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Assessment Instruments ({(defs.assessment_instruments || []).length})</div>
              <div className="list-group list-group-flush">
                {(defs.assessment_instruments || []).map((ins, i) => (
                  <div key={i} className="list-group-item">
                    <div className="d-flex justify-content-between">
                      <span className="fw-semibold">{ins.code || ins.instrument}</span>
                      <span className="badge bg-primary">{ins.domain}</span>
                    </div>
                    <div className="small text-muted">{ins.full_name || ins.label}</div>
                    {ins.range && <div className="small text-muted">Range: {ins.range}</div>}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Quality metrics */}
          <div className="col-lg-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Quality Metrics</div>
              <div className="list-group list-group-flush">
                {(defs.quality_metrics || []).map((m, i) => (
                  <div key={i} className="list-group-item">
                    <div className="fw-semibold">{m.metric}</div>
                    <div className="text-muted small">{m.definition}</div>
                    {m.target && <div className="small text-success">Target: {m.target}</div>}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Compliance */}
          {defs.compliance && (
            <div className="col-lg-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Compliance &amp; Standards</div>
                <div className="card-body">
                  {Array.isArray(defs.compliance)
                    ? defs.compliance.map((c, i) => (
                        <div key={i} className="mb-2">
                          <span className="fw-semibold">{c.standard || c.name}: </span>
                          <span className="text-muted small">{c.description || c.status}</span>
                        </div>
                      ))
                    : Object.entries(defs.compliance).map(([k, v]) => (
                        <div key={k} className="mb-2">
                          <span className="fw-semibold">{k}: </span>
                          <span className="text-muted small">{typeof v === 'string' ? v : JSON.stringify(v)}</span>
                        </div>
                      ))
                  }
                </div>
              </div>
            </div>
          )}

          {/* Remediation */}
          {defs.remediation && (
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Remediation Actions</div>
                <div className="card-body">
                  {Array.isArray(defs.remediation)
                    ? defs.remediation.map((r, i) => (
                        <div key={i} className="mb-2">
                          <span className="badge bg-warning text-dark me-2">{r.priority || r.level}</span>
                          <span className="small">{r.action || r.description}</span>
                        </div>
                      ))
                    : Object.entries(defs.remediation).map(([k, v]) => (
                        <div key={k} className="mb-2">
                          <span className="fw-semibold">{k}: </span>
                          <span className="text-muted small">{typeof v === 'string' ? v : JSON.stringify(v)}</span>
                        </div>
                      ))
                  }
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

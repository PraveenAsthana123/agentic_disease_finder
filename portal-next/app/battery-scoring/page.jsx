'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const tierColor = t =>
  t === 'normal'     ? 'success' :
  t === 'borderline' ? 'warning' :
  t === 'mild'       ? 'orange'  :
  t === 'severe'     ? 'danger'  : 'secondary';

const tierBadge = t =>
  t === 'normal'     ? 'bg-success'  :
  t === 'borderline' ? 'bg-warning text-dark' :
  t === 'mild'       ? 'bg-orange'   :
  t === 'severe'     ? 'bg-danger'   : 'bg-secondary';

function KpiCard({ label, value, unit = '', sub = '', color = 'primary' }) {
  return (
    <div className={`card border-${color} mb-3`}>
      <div className="card-body py-2 px-3">
        <div className={`fw-bold text-${color} fs-5`}>
          {value}{unit && <small className="fs-6 ms-1">{unit}</small>}
        </div>
        <div className="text-muted small">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: '0.72rem' }}>{sub}</div>}
      </div>
    </div>
  );
}

function ZBar({ label, z, normMean, mean }) {
  // z typically −3 to +3; map to 0–100% bar
  const clamp = v => Math.max(0, Math.min(100, v));
  const pct = clamp(((z + 3) / 6) * 100);
  const color = z >= -0.5 ? 'success' : z >= -1.0 ? 'warning' : z >= -2.0 ? 'orange' : 'danger';
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className={`text-${color} fw-semibold`}>
          {mean !== undefined ? `${mean}% (z=${z > 0 ? '+' : ''}${z})` : `z=${z > 0 ? '+' : ''}${z}`}
        </span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function BatteryScoringPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);
  const [selectedTest, setSelectedTest] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/battery-scoring/overview`).then(r => r.json()),
      fetch(`${API}/api/battery-scoring/breakdown`).then(r => r.json()),
      fetch(`${API}/api/battery-scoring/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefs(df); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border text-primary" /></div>;
  if (err) return <div className="container py-4"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpis = overview?.kpis || {};
  const domainSummary = overview?.domain_summary || [];
  const testSummary = overview?.test_summary || [];
  const patientProfiles = breakdown?.patient_profiles || [];
  const rawTables = breakdown?.raw_score_tables || {};

  const tabs = [
    { id: 'overview',  label: '📊 Overview' },
    { id: 'domains',   label: '🧠 Domains' },
    { id: 'patients',  label: '👥 Patients' },
    { id: 'tests',     label: '🔬 Test Scores' },
    { id: 'defs',      label: '📖 Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1">🔬 Neuropsychological Battery Scoring</h4>
      <p className="text-muted small mb-3">
        {defs?.description || 'Normative comparison across 11 validated cognitive tests — 25 epilepsy patients.'}
      </p>

      {/* KPI row */}
      <div className="row g-2 mb-3">
        <div className="col-6 col-md-2"><KpiCard label="Total Records"        value={kpis.total_records}          color="primary" /></div>
        <div className="col-6 col-md-2"><KpiCard label="Patients Profiled"    value={kpis.patients_profiled}      color="info" /></div>
        <div className="col-6 col-md-2"><KpiCard label="Distinct Tests"       value={kpis.distinct_tests}         color="secondary" /></div>
        <div className="col-6 col-md-2"><KpiCard label="Cognitive Domains"    value={kpis.cognitive_domains}      color="secondary" /></div>
        <div className="col-6 col-md-2"><KpiCard label="Impaired Records"     value={kpis.impaired_records}       color="danger" sub={`${kpis.impaired_pct}% of total`} /></div>
        <div className="col-6 col-md-2"><KpiCard label="Patients w/ Impairment" value={kpis.patients_with_any_impairment} color="warning" sub="any domain z<−1" /></div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li className="nav-item" key={t.id}>
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div>
          <h6 className="fw-semibold mb-2">Test-level normative comparison</h6>
          <p className="text-muted small mb-3">
            Z-score = (patient mean − normative mean) / SD. Bar left = impaired, bar right = intact.
          </p>
          <div className="card mb-4">
            <div className="card-body">
              {testSummary.map(t => (
                <ZBar key={t.test} label={`${t.test} (${t.domain})`} z={t.z_score} mean={t.mean_accuracy_pct} />
              ))}
            </div>
          </div>

          <h6 className="fw-semibold mb-2">Impairment rate per test</h6>
          <div className="table-responsive">
            <table className="table table-sm table-bordered">
              <thead className="table-dark">
                <tr>
                  <th>Test</th><th>Domain</th><th>N</th><th>Mean Acc%</th>
                  <th>Norm Mean</th><th>Z-score</th><th>Impaired</th><th>Impaired%</th>
                </tr>
              </thead>
              <tbody>
                {testSummary.map(t => (
                  <tr key={t.test}>
                    <td>{t.test}</td>
                    <td><span className="badge bg-secondary">{t.domain}</span></td>
                    <td>{t.n}</td>
                    <td>{t.mean_accuracy_pct}%</td>
                    <td>{t.norm_mean}%</td>
                    <td className={t.z_score < -1 ? 'text-danger fw-bold' : t.z_score < -0.5 ? 'text-warning' : 'text-success'}>
                      {t.z_score > 0 ? '+' : ''}{t.z_score}
                    </td>
                    <td>{t.impaired_count}</td>
                    <td>
                      <span className={`badge ${t.impaired_pct > 50 ? 'bg-danger' : t.impaired_pct > 30 ? 'bg-warning text-dark' : 'bg-success'}`}>
                        {t.impaired_pct}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Domains tab */}
      {tab === 'domains' && (
        <div>
          <h6 className="fw-semibold mb-2">Domain-level cognitive profile</h6>
          <div className="row g-3">
            {domainSummary.map(d => (
              <div className="col-md-6 col-lg-4" key={d.domain}>
                <div className={`card border-${tierColor(d.impairment_tier)}`}>
                  <div className="card-body">
                    <div className="d-flex justify-content-between align-items-start mb-2">
                      <h6 className="card-title mb-0">{d.domain}</h6>
                      <span className={`badge ${tierBadge(d.impairment_tier)}`}>{d.impairment_tier}</span>
                    </div>
                    <div className="small text-muted mb-2">{d.n_records} records</div>
                    <div className="d-flex justify-content-between small">
                      <span>Patient mean: <strong>{d.mean_accuracy_pct}%</strong></span>
                      <span>Norm: {d.norm_mean}%</span>
                    </div>
                    <div className="progress mt-2" style={{ height: 10 }}>
                      <div
                        className={`progress-bar bg-${tierColor(d.impairment_tier)}`}
                        style={{ width: `${Math.min(100, d.mean_accuracy_pct)}%` }}
                      />
                    </div>
                    <div className="text-end small mt-1">
                      <span className={d.z_score < -1 ? 'text-danger' : d.z_score < -0.5 ? 'text-warning' : 'text-success'}>
                        z = {d.z_score > 0 ? '+' : ''}{d.z_score}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Patients tab */}
      {tab === 'patients' && (
        <div>
          <h6 className="fw-semibold mb-2">Per-patient cognitive profiles (sorted: most impaired first)</h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-bordered">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th><th>Tests</th><th>Mean Acc%</th><th>Mean RT (ms)</th>
                  <th>Overall Tier</th><th>Worst Domain</th><th>Worst Z</th>
                  <th>Impaired Domains</th><th>Last Test</th>
                </tr>
              </thead>
              <tbody>
                {patientProfiles.map(p => (
                  <tr key={p.patient_id}>
                    <td className="fw-semibold">{p.patient_id}</td>
                    <td>{p.distinct_tests} / {p.n_records}</td>
                    <td>{p.mean_accuracy_pct}%</td>
                    <td>{p.mean_reaction_time_ms ? Math.round(p.mean_reaction_time_ms).toLocaleString() : '—'}</td>
                    <td><span className={`badge ${tierBadge(p.overall_tier)}`}>{p.overall_tier}</span></td>
                    <td>{p.worst_domain || '—'}</td>
                    <td className={p.worst_z < -1 ? 'text-danger fw-bold' : p.worst_z < -0.5 ? 'text-warning' : 'text-success'}>
                      {p.worst_z !== null ? (p.worst_z > 0 ? '+' : '') + p.worst_z : '—'}
                    </td>
                    <td>
                      {(p.impaired_domains || []).length > 0
                        ? p.impaired_domains.map(d => <span key={d} className="badge bg-danger me-1">{d}</span>)
                        : <span className="text-success">None</span>}
                    </td>
                    <td>{p.last_test || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Test Scores tab */}
      {tab === 'tests' && (
        <div>
          <h6 className="fw-semibold mb-2">Raw score tables per test</h6>
          <div className="mb-3">
            <select className="form-select form-select-sm w-auto d-inline-block"
              value={selectedTest || ''}
              onChange={e => setSelectedTest(e.target.value || null)}>
              <option value="">— Select a test —</option>
              {Object.keys(rawTables).sort().map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          {selectedTest && rawTables[selectedTest] && (
            <div className="table-responsive">
              <table className="table table-sm table-bordered">
                <thead className="table-dark">
                  <tr>
                    <th>Patient</th><th>Accuracy%</th><th>Z-score</th><th>Tier</th>
                    <th>RT (ms)</th><th>Administered by</th><th>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {rawTables[selectedTest].map((r, i) => (
                    <tr key={i}>
                      <td>{r.patient_id}</td>
                      <td>{r.accuracy_pct}%</td>
                      <td className={r.z_score < -1 ? 'text-danger fw-bold' : r.z_score < -0.5 ? 'text-warning' : 'text-success'}>
                        {r.z_score > 0 ? '+' : ''}{r.z_score}
                      </td>
                      <td><span className={`badge ${tierBadge(r.tier)}`}>{r.tier}</span></td>
                      <td>{r.reaction_time_ms ? Math.round(r.reaction_time_ms).toLocaleString() : '—'}</td>
                      <td>{r.administered_by || '—'}</td>
                      <td>{r.date || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {!selectedTest && (
            <div className="text-muted">Select a test above to view individual scores.</div>
          )}
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'defs' && defs && (
        <div>
          <h6 className="fw-semibold mb-2">Test glossary</h6>
          <div className="table-responsive mb-4">
            <table className="table table-sm table-bordered">
              <thead className="table-dark">
                <tr><th>Test</th><th>Domain</th><th>Description</th><th>Norm Mean</th><th>Max Score</th></tr>
              </thead>
              <tbody>
                {(defs.tests || []).map(t => (
                  <tr key={t.test}>
                    <td className="fw-semibold">{t.test}</td>
                    <td><span className="badge bg-secondary">{t.domain}</span></td>
                    <td>{t.description}</td>
                    <td>{t.norm_mean}%</td>
                    <td>{t.max_score}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h6 className="fw-semibold mb-2">Impairment tiers</h6>
          <div className="table-responsive mb-4">
            <table className="table table-sm table-bordered">
              <thead className="table-dark">
                <tr><th>Tier</th><th>Z-range</th><th>Description</th></tr>
              </thead>
              <tbody>
                {(defs.impairment_tiers || []).map(t => (
                  <tr key={t.tier}>
                    <td><span className={`badge ${tierBadge(t.tier)}`}>{t.tier}</span></td>
                    <td>{t.z_range}</td>
                    <td>{t.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card mb-3">
            <div className="card-body">
              <h6 className="fw-semibold">Normative Reference</h6>
              <p className="small mb-0">{defs.normative_reference}</p>
            </div>
          </div>
          <div className="card">
            <div className="card-body">
              <h6 className="fw-semibold">Clinical Context</h6>
              <p className="small mb-0">{defs.clinical_context}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

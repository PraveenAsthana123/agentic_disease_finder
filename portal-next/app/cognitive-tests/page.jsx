'use client';
import { useEffect, useState } from 'react';

const API = '/api/cognitive-tests';

function KpiCard({ label, value, sub }) {
  return (
    <div className="card text-center shadow-sm h-100">
      <div className="card-body">
        <div className="fs-3 fw-bold text-primary">{value}</div>
        <div className="small fw-semibold">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: '0.75rem' }}>{sub}</div>}
      </div>
    </div>
  );
}

const DOMAIN_COLOR = {
  'Executive function': '#6366f1',
  'Working memory': '#06b6d4',
  'Processing speed': '#f59e0b',
  'Memory': '#10b981',
  'Attention': '#3b82f6',
  'Sustained attention': '#8b5cf6',
  'Visuospatial': '#ef4444',
  'Impulse control': '#f97316',
  'Language': '#84cc16',
};

function AccuracyBar({ value, max = 100, color = '#6366f1' }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div style={{ background: '#e5e7eb', borderRadius: 4, height: 10, width: '100%' }}>
      <div style={{ width: `${pct}%`, background: color, borderRadius: 4, height: 10 }} />
    </div>
  );
}

export default function CognitiveTestsDashboard() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState('overview');
  const [patientSort, setPatientSort] = useState('total');
  const [patientDir, setPatientDir] = useState(-1);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/overview`).then(r => r.json()),
      fetch(`${API}/breakdown`).then(r => r.json()),
      fetch(`${API}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, def]) => { setOverview(ov); setBreakdown(br); setDefinitions(def); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}><div className="spinner-border text-primary" /></div>;
  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;

  const sortedPatients = breakdown?.per_patient
    ? [...breakdown.per_patient].sort((a, b) => patientDir * (b[patientSort] - a[patientSort]))
    : [];

  const tabs = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'tests', label: '🧪 By Test' },
    { id: 'domains', label: '🧠 Domains' },
    { id: 'patients', label: '👥 Per Patient' },
    { id: 'definitions', label: '📚 Definitions' },
  ];

  return (
    <div className="container-fluid py-4">
      <h2 className="fw-bold mb-1">🧠 Cognitive Tests Dashboard</h2>
      <p className="text-muted mb-3">
        {overview?.total_tests} tests · {overview?.total_patients} patients · {overview?.total_test_types} test types · avg accuracy {overview?.avg_accuracy}%
      </p>

      {/* KPI row */}
      <div className="row g-3 mb-4">
        <div className="col-6 col-md-3"><KpiCard label="Total Tests" value={overview?.total_tests} /></div>
        <div className="col-6 col-md-3"><KpiCard label="Patients" value={overview?.total_patients} /></div>
        <div className="col-6 col-md-3"><KpiCard label="Test Types" value={overview?.total_test_types} /></div>
        <div className="col-6 col-md-3"><KpiCard label="Avg Accuracy" value={`${overview?.avg_accuracy}%`} /></div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW tab ── */}
      {tab === 'overview' && (
        <div>
          <div className="row g-4">
            {/* Test Distribution */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Test Distribution</div>
                <div className="card-body">
                  {overview?.test_distribution && Object.entries(overview.test_distribution)
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, cnt]) => (
                      <div key={name} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>{name}</span><span className="fw-semibold">{cnt}</span>
                        </div>
                        <AccuracyBar value={cnt} max={Math.max(...Object.values(overview.test_distribution))} color="#6366f1" />
                      </div>
                    ))}
                </div>
              </div>
            </div>

            {/* Domain Distribution */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Domain Distribution</div>
                <div className="card-body">
                  {overview?.domain_distribution && Object.entries(overview.domain_distribution)
                    .sort((a, b) => b[1] - a[1])
                    .map(([domain, cnt]) => (
                      <div key={domain} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>
                            <span className="badge me-1" style={{ background: DOMAIN_COLOR[domain] || '#888', fontSize: '0.65rem' }}>&nbsp;</span>
                            {domain}
                          </span>
                          <span className="fw-semibold">{cnt}</span>
                        </div>
                        <AccuracyBar value={cnt} max={Math.max(...Object.values(overview.domain_distribution))} color={DOMAIN_COLOR[domain] || '#888'} />
                      </div>
                    ))}
                </div>
              </div>
            </div>

            {/* Monthly Volume */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Monthly Volume</div>
                <div className="card-body">
                  {overview?.monthly_volume?.map(m => (
                    <div key={m.month} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{m.month}</span><span className="fw-semibold">{m.count}</span>
                      </div>
                      <AccuracyBar value={m.count} max={Math.max(...(overview.monthly_volume.map(x => x.count)))} color="#10b981" />
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Administrator Workload */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Administrator Workload</div>
                <div className="card-body">
                  {overview?.admin_distribution && Object.entries(overview.admin_distribution)
                    .sort((a, b) => b[1] - a[1])
                    .map(([admin, cnt]) => (
                      <div key={admin} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>{admin}</span><span className="fw-semibold">{cnt}</span>
                        </div>
                        <AccuracyBar value={cnt} max={Math.max(...Object.values(overview.admin_distribution))} color="#f59e0b" />
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TESTS tab ── */}
      {tab === 'tests' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Per-Test Performance</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-hover table-sm mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Test Name</th>
                    <th>Domain</th>
                    <th>Avg Score</th>
                    <th>Max Score</th>
                    <th>Avg Accuracy</th>
                    <th>Accuracy Bar</th>
                  </tr>
                </thead>
                <tbody>
                  {overview?.test_performance?.map(t => (
                    <tr key={t.test_name}>
                      <td className="fw-semibold">{t.test_name}</td>
                      <td>
                        <span className="badge" style={{ background: DOMAIN_COLOR[overview?.domain_distribution ? Object.keys(DOMAIN_COLOR).find(d => d) : ''] || '#6366f1', fontSize: '0.7rem' }}>
                          {definitions?.tests?.find(d => d.name === t.test_name)?.domain || '—'}
                        </span>
                      </td>
                      <td>{t.avg_score?.toFixed(1)}</td>
                      <td>{t.max_score}</td>
                      <td className="fw-bold">{t.avg_accuracy?.toFixed(1)}%</td>
                      <td style={{ minWidth: 120 }}>
                        <AccuracyBar value={t.avg_accuracy} max={100} color="#6366f1" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── DOMAINS tab ── */}
      {tab === 'domains' && (
        <div className="row g-4">
          {overview?.domain_accuracy?.map(d => (
            <div key={d.domain} className="col-md-6 col-lg-4">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <div className="d-flex align-items-center gap-2 mb-2">
                    <span style={{ display: 'inline-block', width: 14, height: 14, borderRadius: '50%', background: DOMAIN_COLOR[d.domain] || '#888' }} />
                    <span className="fw-semibold">{d.domain}</span>
                  </div>
                  <div className="fs-4 fw-bold mb-1">{d.avg_accuracy?.toFixed(1)}%</div>
                  <div className="text-muted small mb-2">Average Accuracy</div>
                  <AccuracyBar value={d.avg_accuracy} max={100} color={DOMAIN_COLOR[d.domain] || '#888'} />
                  <div className="mt-2 small text-muted">
                    {overview.domain_distribution?.[d.domain] || '—'} tests administered
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── PER PATIENT tab ── */}
      {tab === 'patients' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold d-flex justify-content-between align-items-center">
            <span>Per-Patient Summary ({sortedPatients.length} patients)</span>
            <select className="form-select form-select-sm" style={{ width: 'auto' }}
              value={patientSort} onChange={e => setPatientSort(e.target.value)}>
              <option value="total">Sort: Total Tests</option>
              <option value="avg_accuracy">Sort: Avg Accuracy</option>
              <option value="tests_taken">Sort: Test Types</option>
            </select>
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-hover table-sm mb-0">
                <thead className="table-light">
                  <tr>
                    <th onClick={() => { setPatientSort('patient_id'); setPatientDir(d => -d); }} style={{ cursor: 'pointer' }}>Patient</th>
                    <th onClick={() => { setPatientSort('total'); setPatientDir(d => -d); }} style={{ cursor: 'pointer' }}>Total ↕</th>
                    <th onClick={() => { setPatientSort('tests_taken'); setPatientDir(d => -d); }} style={{ cursor: 'pointer' }}>Types ↕</th>
                    <th onClick={() => { setPatientSort('avg_accuracy'); setPatientDir(d => -d); }} style={{ cursor: 'pointer' }}>Avg Accuracy ↕</th>
                    <th>Avg Score</th>
                    <th>First Test</th>
                    <th>Last Test</th>
                    <th>Accuracy Bar</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedPatients.map(p => (
                    <tr key={p.patient_id}>
                      <td className="fw-semibold">{p.patient_id}</td>
                      <td>{p.total}</td>
                      <td>{p.tests_taken}</td>
                      <td>
                        <span className={`badge ${p.avg_accuracy >= 70 ? 'bg-success' : p.avg_accuracy >= 60 ? 'bg-warning text-dark' : 'bg-danger'}`}>
                          {p.avg_accuracy?.toFixed(1)}%
                        </span>
                      </td>
                      <td>{p.avg_score?.toFixed(1)}</td>
                      <td className="text-muted small">{p.first_test}</td>
                      <td className="text-muted small">{p.last_test}</td>
                      <td style={{ minWidth: 100 }}>
                        <AccuracyBar value={p.avg_accuracy} max={100} color={p.avg_accuracy >= 70 ? '#10b981' : p.avg_accuracy >= 60 ? '#f59e0b' : '#ef4444'} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS tab ── */}
      {tab === 'definitions' && (
        <div className="row g-4">
          {definitions?.tests?.map(t => (
            <div key={t.name} className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header d-flex justify-content-between align-items-center">
                  <span className="fw-semibold">{t.name}</span>
                  <span className="badge" style={{ background: DOMAIN_COLOR[t.domain] || '#888', fontSize: '0.7rem' }}>{t.domain}</span>
                </div>
                <div className="card-body small">
                  <p className="mb-1"><strong>Description:</strong> {t.description}</p>
                  <p className="mb-1"><strong>Scoring:</strong> {t.scoring}</p>
                  <p className="mb-0 text-muted"><strong>Clinical relevance:</strong> {t.clinical_relevance}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

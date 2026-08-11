'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const VISIT_COLORS = {
  'completed': '#22c55e',
  'confirmed': '#3b82f6',
  'no-show': '#ef4444',
  'booked': '#6366f1',
  'cancelled': '#f59e0b',
  'rescheduled': '#f97316',
};

const INSTR_COLORS = [
  '#6366f1','#3b82f6','#22c55e','#f59e0b','#ef4444',
  '#8b5cf6','#10b981','#f97316','#06b6d4','#e11d48',
  '#84cc16','#a855f7','#0ea5e9','#14b8a6','#fb923c','#64748b',
];

function StatCard({ label, value, sub, color = '#6366f1' }) {
  return (
    <div className="col-6 col-md mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className="h4 mb-0 fw-bold" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function MiniBar({ value, total, color }) {
  const pct = total ? Math.min(100, (value / total) * 100) : 0;
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 10, minWidth: 80 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color || '#6366f1' }} />
      </div>
      <span className="small fw-bold">{value}</span>
    </div>
  );
}

function ComplianceBadge({ pct }) {
  const color = pct >= 80 ? '#22c55e' : pct >= 60 ? '#f59e0b' : '#ef4444';
  const label = pct >= 80 ? 'Good' : pct >= 60 ? 'At Risk' : 'Critical';
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 4,
      fontSize: '0.72rem', fontWeight: 700, color: '#fff', backgroundColor: color,
    }}>{label}</span>
  );
}

export default function ResearchCoordinatorDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [patSort, setPatSort] = useState('assessments_count');
  const [patDir, setPatDir] = useState(-1);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/research-coordinator/overview`).then(r => r.json()),
      fetch(`${API}/api/research-coordinator/breakdown`).then(r => r.json()),
      fetch(`${API}/api/research-coordinator/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading research coordinator data…</div>;

  const totalSubj = ov.total_subjects || 0;
  const compPct = ov.visit_compliance_pct || 0;

  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'subjects', label: 'Subject Inventory' },
    { id: 'instruments', label: 'Assessment Instruments' },
    { id: 'visits', label: 'Visit Tracking' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const sortedPatients = bd ? [...(bd.subject_inventory || [])].sort((a, b) => {
    const av = a[patSort] ?? -1;
    const bv = b[patSort] ?? -1;
    return (av > bv ? 1 : av < bv ? -1 : 0) * patDir;
  }) : [];

  function sortBy(col) {
    if (patSort === col) setPatDir(d => -d);
    else { setPatSort(col); setPatDir(-1); }
  }

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1">📋 Research Coordinator Dashboard</h4>
      <p className="text-muted small mb-3">
        Clinical trial &amp; study management · Subject enrollment · Visit compliance · Assessment instruments ·
        ICH-GCP · IRB compliance · {totalSubj} enrolled subjects
      </p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div>
          {/* KPI row */}
          <div className="row">
            <StatCard label="Enrolled Subjects" value={ov.total_subjects} color="#6366f1" />
            <StatCard label="Total Assessments" value={ov.total_assessments} color="#3b82f6" />
            <StatCard label="Total Visits" value={ov.total_visits} color="#8b5cf6" />
            <StatCard label="Visit Compliance" value={`${compPct}%`}
              color={compPct >= 80 ? '#22c55e' : compPct >= 60 ? '#f59e0b' : '#ef4444'}
              sub={compPct < 80 ? 'Below 80% target' : 'On target'} />
            <StatCard label="EEG Uploads" value={ov.total_eeg_uploads} color="#10b981" />
            <StatCard label="Analyses Complete" value={ov.analyses_complete} color="#06b6d4" />
            <StatCard label="Seizure Events" value={ov.total_seizure_events} color="#ef4444" />
            <StatCard label="Instruments Used" value={ov.instruments_used} color="#f59e0b" />
          </div>

          <div className="row g-3">
            {/* Visit compliance alert */}
            {compPct < 80 && (
              <div className="col-12">
                <div className="alert alert-warning py-2 mb-0">
                  <strong>⚠ Visit Compliance Below Target:</strong> Current {compPct}% vs. 80% ICH-GCP target.
                  Implement retention interventions — appointment reminders, flexible scheduling, transportation support.
                </div>
              </div>
            )}

            {/* Visit status distribution */}
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Visit Status Distribution</div>
                <div className="card-body">
                  {(ov.visit_status_distribution || []).map(v => {
                    const color = VISIT_COLORS[v.status] || '#6b7280';
                    return (
                      <div key={v.status} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span className="text-capitalize fw-semibold">{v.status}</span>
                          <span style={{ color }}>{v.count} ({Math.round(100 * v.count / (ov.total_visits || 1))}%)</span>
                        </div>
                        <MiniBar value={v.count} total={ov.total_visits} color={color} />
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Enrollment by month */}
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Enrollment Timeline</div>
                <div className="card-body">
                  <div className="small text-muted mb-2">
                    Study period: {ov.date_range?.earliest} → {ov.date_range?.latest}
                  </div>
                  {(ov.enrollment_by_month || []).map(m => (
                    <div key={m.month} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="fw-semibold">{m.month}</span>
                        <span className="text-primary">{m.count} subjects</span>
                      </div>
                      <MiniBar value={m.count} total={totalSubj} color="#6366f1" />
                    </div>
                  ))}
                  {(ov.enrollment_by_month || []).length === 0 && (
                    <div className="text-muted small">No monthly enrollment data</div>
                  )}
                </div>
              </div>
            </div>

            {/* Top instruments */}
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Assessment Instrument Usage</div>
                <div className="card-body">
                  {(ov.instrument_coverage || []).slice(0, 10).map((instr, i) => (
                    <div key={instr.instrument} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="fw-semibold">{instr.instrument}</span>
                        <span className="text-muted">{instr.count} assessments / {instr.patients_assessed} pts</span>
                      </div>
                      <MiniBar value={instr.count} total={(ov.instrument_coverage || []).reduce((s, x) => s + x.count, 0) / (ov.instrument_coverage || [1]).length} color={INSTR_COLORS[i % INSTR_COLORS.length]} />
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Disease distribution */}
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Study Cohort Disease Distribution</div>
                <div className="card-body">
                  {(ov.disease_distribution || []).map((d, i) => (
                    <div key={d.disease} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="fw-semibold text-capitalize">{d.disease}</span>
                        <span className="text-primary">{d.count} subjects ({Math.round(100 * d.count / totalSubj)}%)</span>
                      </div>
                      <MiniBar value={d.count} total={totalSubj} color={INSTR_COLORS[i % INSTR_COLORS.length]} />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUBJECT INVENTORY TAB */}
      {tab === 'subjects' && bd && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">
            Subject Inventory ({sortedPatients.length} subjects)
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    {[
                      ['patient_id', 'Subject ID'],
                      ['name', 'Name'],
                      ['age', 'Age'],
                      ['gender', 'Gender'],
                      ['disease', 'Disease'],
                      ['assessments_count', 'Assessments'],
                      ['visits_count', 'Visits'],
                      ['seizure_events', 'Seizures'],
                      ['uploads', 'EEG Uploads'],
                      ['enrollment_date', 'Enrolled'],
                    ].map(([col, label]) => (
                      <th key={col} style={{ cursor: 'pointer', whiteSpace: 'nowrap' }}
                        onClick={() => sortBy(col)}>
                        {label} {patSort === col ? (patDir === -1 ? '▼' : '▲') : ''}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedPatients.map(p => (
                    <tr key={p.patient_id}>
                      <td className="fw-semibold"><code>{p.patient_id}</code></td>
                      <td>{p.name || '—'}</td>
                      <td>{p.age != null ? p.age : '—'}</td>
                      <td>{p.gender || '—'}</td>
                      <td className="text-capitalize">{p.disease || '—'}</td>
                      <td>
                        <span className="badge" style={{ backgroundColor: '#6366f1' }}>{p.assessments_count}</span>
                      </td>
                      <td>{p.visits_count}</td>
                      <td>
                        {p.seizure_events > 0
                          ? <span className="badge bg-danger">{p.seizure_events}</span>
                          : <span className="badge bg-success">0</span>}
                      </td>
                      <td>{p.uploads}</td>
                      <td className="text-muted small">{p.enrollment_date}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ASSESSMENT INSTRUMENTS TAB */}
      {tab === 'instruments' && (
        <div className="row g-3">
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Assessment Instrument Coverage ({ov.instruments_used} instruments)</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Instrument</th>
                        <th>Total Administered</th>
                        <th>Patients Assessed</th>
                        <th>Coverage %</th>
                        <th>Avg / Patient</th>
                        <th>Bar</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(ov.instrument_coverage || []).sort((a, b) => b.count - a.count).map((instr, i) => {
                        const coveragePct = Math.round(100 * instr.patients_assessed / totalSubj);
                        const avgPerPat = instr.patients_assessed > 0
                          ? (instr.count / instr.patients_assessed).toFixed(1)
                          : '—';
                        return (
                          <tr key={instr.instrument}>
                            <td className="fw-semibold">{instr.instrument}</td>
                            <td>{instr.count}</td>
                            <td>{instr.patients_assessed}</td>
                            <td>
                              <span style={{ color: coveragePct >= 80 ? '#22c55e' : coveragePct >= 50 ? '#f59e0b' : '#ef4444' }}>
                                {coveragePct}%
                              </span>
                            </td>
                            <td>{avgPerPat}x</td>
                            <td style={{ minWidth: 80 }}>
                              <div className="progress" style={{ height: 8 }}>
                                <div className="progress-bar" style={{
                                  width: `${coveragePct}%`,
                                  backgroundColor: INSTR_COLORS[i % INSTR_COLORS.length],
                                }} />
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <div className="col-12">
            <div className="alert alert-info small mb-0">
              <strong>Protocol Guidance:</strong> Target ≥ 80% instrument coverage per enrolled subject for primary
              neuropsychological endpoints. CSSRS and PHQ-9 require 100% coverage per ethics protocol.
              Missing assessments should be flagged for follow-up within 5 business days.
            </div>
          </div>
        </div>
      )}

      {/* VISIT TRACKING TAB */}
      {tab === 'visits' && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Visit Compliance Summary</div>
              <div className="card-body">
                <div className="d-flex align-items-center gap-3 mb-3">
                  <div className="display-6 fw-bold" style={{ color: compPct >= 80 ? '#22c55e' : compPct >= 60 ? '#f59e0b' : '#ef4444' }}>
                    {compPct}%
                  </div>
                  <div>
                    <ComplianceBadge pct={compPct} />
                    <div className="text-muted small mt-1">Target: ≥ 80% (ICH-GCP)</div>
                  </div>
                </div>

                <div className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>Completed visits</span>
                    <span className="fw-bold text-success">{ov.completed_visits} / {ov.total_visits}</span>
                  </div>
                  <MiniBar value={ov.completed_visits} total={ov.total_visits} color="#22c55e" />
                </div>

                <div className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>No-shows</span>
                    <span className="fw-bold text-danger">
                      {(ov.visit_status_distribution || []).find(v => v.status === 'no-show')?.count || 0}
                    </span>
                  </div>
                  <MiniBar
                    value={(ov.visit_status_distribution || []).find(v => v.status === 'no-show')?.count || 0}
                    total={ov.total_visits}
                    color="#ef4444"
                  />
                </div>

                <div className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>Cancelled</span>
                    <span className="fw-bold text-warning">
                      {(ov.visit_status_distribution || []).find(v => v.status === 'cancelled')?.count || 0}
                    </span>
                  </div>
                  <MiniBar
                    value={(ov.visit_status_distribution || []).find(v => v.status === 'cancelled')?.count || 0}
                    total={ov.total_visits}
                    color="#f59e0b"
                  />
                </div>

                {compPct < 80 && (
                  <div className="alert alert-warning small mt-3 mb-0 py-2">
                    <strong>Action Required:</strong> Implement retention interventions.
                    Review no-show patterns. Escalate to PI if compliance falls below 70%.
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">All Visit Statuses</div>
              <div className="card-body">
                {(ov.visit_status_distribution || []).map(v => {
                  const color = VISIT_COLORS[v.status] || '#6b7280';
                  const pct = Math.round(100 * v.count / (ov.total_visits || 1));
                  return (
                    <div key={v.status} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-capitalize fw-semibold">{v.status}</span>
                        <span style={{ color }}>{v.count} visits ({pct}%)</span>
                      </div>
                      <div className="progress" style={{ height: 12 }}>
                        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Data Completeness Summary</div>
              <div className="card-body">
                <div className="row g-2">
                  {[
                    { label: 'Subjects with Assessments', value: Math.round(100 * ov.total_subjects / ov.total_subjects), color: '#22c55e' },
                    { label: 'EEG Upload Rate', value: Math.round(100 * ov.total_eeg_uploads / (ov.total_subjects || 1)), color: '#6366f1', note: 'uploads/subject' },
                    { label: 'Analyses vs Uploads', value: Math.round(100 * ov.analyses_complete / (ov.total_eeg_uploads || 1)), color: '#3b82f6' },
                    { label: 'Visit Compliance', value: compPct, color: compPct >= 80 ? '#22c55e' : '#ef4444' },
                  ].map(m => (
                    <div key={m.label} className="col-md-3 col-6">
                      <div className="border rounded p-2 text-center">
                        <div className="h4 fw-bold mb-0" style={{ color: m.color }}>{m.value}%</div>
                        <div className="small text-muted">{m.label}</div>
                        {m.note && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{m.note}</div>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-12">
            <div className="alert alert-info small mb-0">
              Research Coordinator definitions, study phase protocols, compliance references, and remediation strategies
              per ICH-GCP E6(R2), HIPAA, and IRB/IEC requirements.
            </div>
          </div>

          {/* Concepts */}
          <div className="col-12">
            <h6 className="fw-bold">Key Concepts</h6>
            <div className="row g-2">
              {(defs.concepts || []).map(c => (
                <div key={c.name} className="col-md-4">
                  <div className="card shadow-sm h-100">
                    <div className="card-body p-2">
                      <div className="fw-semibold small mb-1" style={{ color: '#6366f1' }}>{c.name}</div>
                      <div className="text-muted small">{c.description}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Quality Metrics */}
          <div className="col-md-6">
            <h6 className="fw-bold">Quality Metrics</h6>
            {(defs.quality_metrics || []).map(m => (
              <div key={m.name} className="mb-2 p-2 border rounded">
                <div className="fw-semibold small">{m.name}</div>
                <div className="text-muted small">{m.description}</div>
              </div>
            ))}
          </div>

          {/* Study Phases */}
          <div className="col-md-6">
            <h6 className="fw-bold">Study Phases</h6>
            {(defs.study_phases || []).map((p, i) => (
              <div key={p.name} className="mb-2 p-2 border rounded" style={{ borderLeft: `3px solid ${INSTR_COLORS[i]}` }}>
                <div className="fw-semibold small">{i + 1}. {p.name}</div>
                <div className="text-muted small">{p.description}</div>
              </div>
            ))}
          </div>

          {/* Compliance References */}
          <div className="col-md-6">
            <h6 className="fw-bold">Regulatory References</h6>
            {(defs.compliance_refs || []).map(r => (
              <div key={r.name} className="mb-2 p-2 bg-light rounded">
                <div className="fw-semibold small text-primary">{r.name}</div>
                <div className="text-muted small">{r.scope}</div>
              </div>
            ))}
          </div>

          {/* Remediation */}
          <div className="col-md-6">
            <h6 className="fw-bold">Remediation Strategies</h6>
            {(defs.remediation || []).map(r => (
              <div key={r.strategy} className="mb-2 p-2 border rounded">
                <div className="fw-semibold small text-warning-emphasis">{r.strategy}</div>
                <div className="text-muted small">{r.description}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

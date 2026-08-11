'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'procedures',  label: 'Procedures' },
  { id: 'patients',    label: 'Patient Logs' },
  { id: 'milestones',  label: 'Milestones' },
  { id: 'definitions', label: 'Definitions' },
];

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Badge({ color, children }) {
  return <span className={`badge bg-${color}`}>{children}</span>;
}

const engelColor = ec => {
  if (!ec) return 'secondary';
  if (ec.startsWith('IA') || ec === 'IB') return 'success';
  if (ec.startsWith('I')) return 'primary';
  if (ec.startsWith('II')) return 'info';
  if (ec.startsWith('III')) return 'warning';
  return 'danger';
};

const ilaeColor = v => {
  if (v === 1) return 'success';
  if (v === 2) return 'primary';
  if (v === 3) return 'info';
  if (v === 4) return 'warning';
  return 'danger';
};

export default function SeizureFreedomPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [ptFilter, setPtFilter] = useState('all');

  useEffect(() => {
    fetch(`${API}/api/seizure-freedom/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/seizure-freedom/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/seizure-freedom/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /><span className="ms-2">Loading Seizure Freedom Tracker…</span></div>;

  const kpi = ov.kpis || {};

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3">
        <span style={{ fontSize: '1.8rem' }} className="me-2">🏆</span>
        <div>
          <h4 className="mb-0 fw-bold">Seizure Freedom Tracker</h4>
          <small className="text-muted">
            {kpi.total_procedures} procedures · {kpi.unique_surgical_patients} surgical patients ·
            {' '}{kpi.seizure_free_count} seizure-free · avg {kpi.avg_followup_months}mo follow-up
          </small>
        </div>
      </div>

      {/* Tab nav */}
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

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          <div className="row mb-2">
            <KPI label="Surgical Procedures"  value={kpi.total_procedures}   color="primary" />
            <KPI label="Seizure-Free"          value={`${kpi.seizure_free_count} (${kpi.seizure_free_pct}%)`} color="success" sub="Engel IA/IB post-surgery" />
            <KPI label="Avg Follow-Up"         value={`${kpi.avg_followup_months}mo`} color="info" />
            <KPI label="AED Reduction Rate"    value={`${kpi.aed_reduction_pct}%`}    color="warning" sub="of surgical patients" />
          </div>
          <div className="row mb-4">
            <KPI label="Complication Rate"     value={`${kpi.complication_pct}%`}  color="danger" sub="any complication" />
            <KPI label="Avg Seizure Reduction" value={`${kpi.avg_seizure_frequency_reduction_pct}%`} color="success" sub="frequency reduction" />
            <KPI label="Trigger-Log Patients" value={ov.trigger_log_cohort?.total_patients} color="primary" />
            <KPI label="TL Seizure-Free"       value={`${ov.trigger_log_cohort?.seizure_free_pct}%`} color="info" sub="0 seizure days logged" />
          </div>

          <div className="row">
            {/* Engel Distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small">Engel Classification</div>
                <div className="card-body p-2">
                  <table className="table table-sm mb-0">
                    <thead><tr><th>Class</th><th>N</th><th>%</th></tr></thead>
                    <tbody>
                      {(ov.engel_distribution || []).map(e => (
                        <tr key={e.class}>
                          <td><Badge color={engelColor(e.class)}>{e.class}</Badge></td>
                          <td>{e.count}</td>
                          <td>{kpi.total_procedures ? ((e.count / kpi.total_procedures) * 100).toFixed(1) : 0}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* ILAE Distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small">ILAE Outcome Scale</div>
                <div className="card-body p-2">
                  <table className="table table-sm mb-0">
                    <thead><tr><th>Outcome</th><th>N</th></tr></thead>
                    <tbody>
                      {(ov.ilae_distribution || []).map(il => (
                        <tr key={il.outcome}>
                          <td><Badge color={il.outcome.includes('seizure-free') ? 'success' : il.outcome.includes('90%') ? 'primary' : il.outcome.includes('50') ? 'info' : 'warning'}>{il.outcome}</Badge></td>
                          <td>{il.count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Surgery Type */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small">Surgery Type</div>
                <div className="card-body p-2">
                  <table className="table table-sm mb-0">
                    <thead><tr><th>Type</th><th>N</th></tr></thead>
                    <tbody>
                      {(ov.surgery_type_distribution || []).map(s => (
                        <tr key={s.type}><td>{s.type}</td><td>{s.count}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Drug Responsiveness */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small">Drug Responsiveness (Full Cohort)</div>
            <div className="card-body">
              <div className="row">
                {(ov.drug_responsiveness_distribution || []).map(d => (
                  <div key={d.response} className="col-md-3 mb-2">
                    <div className="border rounded p-2 text-center">
                      <div className="h5 mb-0 fw-bold text-primary">{d.count}</div>
                      <div className="small text-muted">{d.response}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Monthly trend */}
          {(ov.monthly_seizure_free_trend || []).length > 0 && (
            <div className="card shadow-sm">
              <div className="card-header fw-bold small">Monthly Seizure-Free Day % (Trigger Logs)</div>
              <div className="card-body p-2" style={{ overflowX: 'auto' }}>
                <div style={{ display: 'flex', alignItems: 'flex-end', gap: 4, height: 80 }}>
                  {(ov.monthly_seizure_free_trend || []).map(m => (
                    <div key={m.month} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1 }}>
                      <div style={{ height: `${m.seizure_free_pct * 0.7}px`, background: m.seizure_free_pct >= 80 ? '#198754' : m.seizure_free_pct >= 60 ? '#0dcaf0' : '#ffc107', borderRadius: 3, width: '100%' }} />
                      <div style={{ fontSize: '0.55rem', writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>{m.month}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* ── PROCEDURES ── */}
      {tab === 'procedures' && bd && (
        <div className="card shadow-sm">
          <div className="card-header fw-bold small">
            Per-Procedure Seizure Freedom ({(bd.procedures || []).length} procedures)
          </div>
          <div className="card-body p-0" style={{ overflowX: 'auto' }}>
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Patient</th><th>Surgery</th><th>Date</th><th>Pathology</th>
                  <th>Engel</th><th>ILAE</th><th>Follow-up</th>
                  <th>Seizure-Free</th><th>Freq ↓%</th><th>AED ↓</th><th>Driving OK</th>
                </tr>
              </thead>
              <tbody>
                {(bd.procedures || []).map((p, i) => (
                  <tr key={i}>
                    <td className="small">{p.patient_id}</td>
                    <td className="small">{p.surgery_type}</td>
                    <td className="small">{p.surgery_date}</td>
                    <td className="small">{p.pathology}</td>
                    <td><Badge color={engelColor(p.engel_class)}>{p.engel_class}</Badge></td>
                    <td><Badge color={ilaeColor(p.ilae_outcome)}>{p.ilae_outcome}</Badge></td>
                    <td className="small">{p.follow_up_months}mo</td>
                    <td>{p.seizure_free ? <Badge color="success">✓ Yes</Badge> : <Badge color="warning">No</Badge>}</td>
                    <td className="small">{p.freq_reduction_pct != null ? `${p.freq_reduction_pct}%` : '—'}</td>
                    <td>{p.aed_reduction ? '✓' : '—'}</td>
                    <td>{p.driving_eligible ? <Badge color="success">Eligible</Badge> : <Badge color="secondary">Not yet</Badge>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── PATIENT LOGS ── */}
      {tab === 'patients' && bd && (
        <>
          <div className="mb-2">
            <select className="form-select form-select-sm w-auto d-inline"
              value={ptFilter} onChange={e => setPtFilter(e.target.value)}>
              <option value="all">All Patients</option>
              <option value="seizure_free">Seizure-Free Only</option>
              <option value="not_free">Not Seizure-Free</option>
            </select>
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-bold small">Trigger-Log Patient Profiles</div>
            <div className="card-body p-0" style={{ overflowX: 'auto' }}>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th><th>Log Days</th><th>Seizure Days</th>
                    <th>SF%</th><th>Seizure-Free</th><th>Avg Adherence</th>
                    <th>Drug Response</th><th>Current Freq</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.trigger_log_profiles || [])
                    .filter(p =>
                      ptFilter === 'all' ? true :
                      ptFilter === 'seizure_free' ? p.seizure_free :
                      !p.seizure_free
                    )
                    .map((p, i) => (
                      <tr key={i}>
                        <td className="small fw-bold">{p.patient_id}</td>
                        <td className="small">{p.total_log_days}</td>
                        <td className="small">{p.seizure_days}</td>
                        <td>
                          <div className="progress" style={{ height: 14 }}>
                            <div
                              className={`progress-bar bg-${p.seizure_free_pct >= 80 ? 'success' : p.seizure_free_pct >= 60 ? 'info' : 'warning'}`}
                              style={{ width: `${p.seizure_free_pct}%` }}
                            >{p.seizure_free_pct}%</div>
                          </div>
                        </td>
                        <td>{p.seizure_free ? <Badge color="success">✓</Badge> : <Badge color="warning">No</Badge>}</td>
                        <td className="small">{p.avg_medication_adherence != null ? `${p.avg_medication_adherence}%` : '—'}</td>
                        <td className="small" style={{ maxWidth: 160 }}>{p.drug_responsiveness}</td>
                        <td className="small">{p.current_seizure_frequency}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── MILESTONES ── */}
      {tab === 'milestones' && bd && (
        <>
          <div className="row mb-4">
            {(bd.milestones || []).map(m => (
              <div key={m.milestone} className="col-md-4 mb-3">
                <div className="card shadow-sm border-success h-100">
                  <div className="card-header bg-success text-white small fw-bold">{m.milestone}</div>
                  <div className="card-body text-center">
                    <div className="h2 fw-bold text-success">{m.patients}</div>
                    <div className="small text-muted">of {m.total} surgical patients</div>
                    <div className="small mt-1">{m.description}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Milestone reference from definitions */}
          {defs && (
            <div className="card shadow-sm">
              <div className="card-header fw-bold small">Clinical Milestone Reference</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Months SF</th><th>Milestone</th><th>Description</th></tr>
                  </thead>
                  <tbody>
                    {(defs.milestones || []).map(m => (
                      <tr key={m.months}>
                        <td><Badge color="primary">{m.months}mo</Badge></td>
                        <td className="small fw-bold">{m.milestone}</td>
                        <td className="small text-muted">{m.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <>
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small">Dashboard Description</div>
            <div className="card-body small">{defs.description}</div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small">Engel Classification (Seizure Outcome)</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Class</th><th>Tier</th><th>Definition</th></tr></thead>
                <tbody>
                  {(defs.engel_classes || []).map(e => (
                    <tr key={e.class}>
                      <td><Badge color={e.color}>{e.class}</Badge></td>
                      <td className="small">{e.tier}</td>
                      <td className="small">{e.label}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small">ILAE Outcome Scale</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Value</th><th>Outcome</th></tr></thead>
                <tbody>
                  {(defs.ilae_outcomes || []).map(il => (
                    <tr key={il.value}>
                      <td><Badge color={il.color}>{il.value}</Badge></td>
                      <td className="small">{il.label}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small">Drug Responsiveness Categories</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Category</th><th>Description</th></tr></thead>
                <tbody>
                  {(defs.drug_responsiveness || []).map(d => (
                    <tr key={d.label}>
                      <td className="small fw-bold">{d.label}</td>
                      <td className="small">{d.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-bold small">Data Sources</div>
            <div className="card-body">
              <ul className="mb-1">
                {(defs.data_sources || []).map((s, i) => <li key={i} className="small">{s}</li>)}
              </ul>
              <p className="small text-muted mb-0">{defs.clinical_reference}</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TIER_BADGE = {
  Excellent: 'success',
  Adequate:  'primary',
  Poor:      'warning',
  Critical:  'danger',
};

function StatCard({ label, value, sub, color = '#3b82f6' }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-3">
          <div className="h3 fw-bold mb-0" style={{ color }}>{value}</div>
          <div className="small fw-semibold">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

export default function AedCompliancePage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/aed-compliance/overview`).then(r => r.json()),
      fetch(`${API}/api/aed-compliance/breakdown`).then(r => r.json()),
      fetch(`${API}/api/aed-compliance/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">{err}</div>;
  if (!ov)  return <div className="p-4 text-muted">Loading AED Compliance Analytics…</div>;

  const kpi = ov.kpis;

  // Donut-style mini progress bar
  const Bar = ({ pct, color }) => (
    <div className="progress" style={{ height: 10, borderRadius: 6 }}>
      <div className="progress-bar" style={{ width: `${pct}%`, background: color, borderRadius: 6 }} />
    </div>
  );

  const TABS = ['overview', 'by-drug', 'per-patient', 'side-effects', 'definitions'];

  return (
    <div className="container-fluid py-4 px-3">
      <h4 className="fw-bold mb-1">💊 AED Compliance Analytics</h4>
      <p className="text-muted small mb-3">
        {kpi.total_doses.toLocaleString()} dose records · {kpi.total_patients} patients ·{' '}
        {kpi.total_drugs} AEDs · {kpi.tracking_days} days · Real data from{' '}
        <code>medication_adherence</code>
      </p>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === t ? ' active' : ''}`}
              onClick={() => setTab(t)}
              style={{ textTransform: 'capitalize' }}
            >
              {t.replace('-', ' ')}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <div className="row">
            <StatCard label="Overall Adherence" value={`${kpi.adherence_pct}%`}
                      sub={`${kpi.taken_doses.toLocaleString()} / ${kpi.total_doses.toLocaleString()} doses taken`}
                      color="#22c55e" />
            <StatCard label="On-Time Doses" value={kpi.on_time_doses.toLocaleString()}
                      sub={`${ov.status_distribution.find(s=>s.status==='On Time')?.pct}%`}
                      color="#3b82f6" />
            <StatCard label="Late Doses" value={kpi.late_doses.toLocaleString()}
                      sub={`${ov.status_distribution.find(s=>s.status==='Late')?.pct}%`}
                      color="#f59e0b" />
            <StatCard label="Missed Doses" value={kpi.missed_doses.toLocaleString()}
                      sub={`Miss rate ${kpi.miss_rate_pct}%`}
                      color="#ef4444" />
          </div>

          {/* Status breakdown + Trend */}
          <div className="row">
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header py-2 fw-semibold small">Dose Status Breakdown</div>
                <div className="card-body">
                  {ov.status_distribution.map(s => (
                    <div key={s.status} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span style={{ color: s.color }}>{s.status}</span>
                        <span>{s.count.toLocaleString()} ({s.pct}%)</span>
                      </div>
                      <Bar pct={s.pct} color={s.color} />
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header py-2 fw-semibold small">Monthly Adherence Trend</div>
                <div className="card-body">
                  <table className="table table-sm mb-0">
                    <thead><tr><th>Month</th><th>Doses</th><th>Adherence</th></tr></thead>
                    <tbody>
                      {ov.monthly_trend.map(m => (
                        <tr key={m.month}>
                          <td>{m.month}</td>
                          <td>{m.doses.toLocaleString()}</td>
                          <td>
                            <span className={`badge bg-${m.adherence_pct >= 95 ? 'success' : 'primary'}`}>
                              {m.adherence_pct}%
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header py-2 fw-semibold small">Adherence by Time of Day</div>
                <div className="card-body">
                  {ov.time_of_day.map(t => (
                    <div key={t.time} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-capitalize">{t.time}</span>
                        <span>{t.adherence_pct}% ({t.missed} missed)</span>
                      </div>
                      <Bar pct={t.adherence_pct} color="#3b82f6" />
                    </div>
                  ))}
                  <hr className="my-2" />
                  <div className="small fw-semibold mt-2">By Frequency</div>
                  {ov.by_frequency.map(f => (
                    <div key={f.frequency} className="d-flex justify-content-between small my-1">
                      <span>{f.frequency}</span>
                      <span className={`badge bg-${f.adherence_pct >= 95 ? 'success' : 'primary'}`}>
                        {f.adherence_pct}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Drug adherence bars */}
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold small">Per-Drug Adherence Summary</div>
            <div className="card-body">
              {ov.drug_summary.map(d => (
                <div key={d.drug} className="mb-3">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className="fw-semibold">{d.drug}</span>
                    <span>
                      <span className={`badge bg-${TIER_BADGE[d.tier]} me-2`}>{d.tier}</span>
                      {d.adherence_pct}% adherence · {d.miss_rate_pct}% miss · SE rate {d.se_rate_pct}%
                    </span>
                  </div>
                  <Bar pct={d.adherence_pct} color={d.tier_color} />
                </div>
              ))}
            </div>
          </div>

          {/* Compliance tier legend */}
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold small">ILAE Compliance Tiers</div>
            <div className="card-body">
              <div className="row g-2">
                {ov.compliance_tiers.map(t => (
                  <div key={t.label} className="col-6 col-md-3">
                    <div className="p-2 rounded" style={{ background: t.color + '20', borderLeft: `4px solid ${t.color}` }}>
                      <div className="fw-semibold small" style={{ color: t.color }}>{t.label} (≥{t.threshold}%)</div>
                      <div className="text-muted" style={{ fontSize: 11 }}>{t.risk}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── BY DRUG ── */}
      {tab === 'by-drug' && bd && (
        <>
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold small">Drug Adherence & Timing Detail</div>
            <div className="table-responsive">
              <table className="table table-hover mb-0 small">
                <thead className="table-light">
                  <tr>
                    <th>Drug</th><th>Class</th><th>Frequency</th><th>Patients</th>
                    <th>Avg Delay (late)</th><th>Mechanism (brief)</th>
                  </tr>
                </thead>
                <tbody>
                  {bd.drug_detail.map(d => (
                    <tr key={d.drug}>
                      <td className="fw-semibold">{d.drug}</td>
                      <td>{d.profile?.class || '—'}</td>
                      <td><span className="badge bg-secondary">{d.frequency}</span></td>
                      <td>{d.n_patients}</td>
                      <td>{d.avg_minutes_late > 0 ? `${d.avg_minutes_late} min` : '—'}</td>
                      <td className="text-muted" style={{ maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {d.profile?.mechanism || '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold small">Mood After Dose by Adherence Status</div>
            <div className="card-body">
              <div className="row g-2">
                {bd.mood_by_status.map(m => (
                  <div key={m.status} className="col-4">
                    <div className="text-center p-3 rounded border">
                      <div className="h4 mb-0">{m.avg_mood}</div>
                      <div className="small text-muted">Avg mood (1–10)</div>
                      <div className="badge bg-secondary mt-1 text-capitalize">{m.status}</div>
                      <div className="text-muted" style={{ fontSize: 11 }}>{m.count.toLocaleString()} doses</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── PER PATIENT ── */}
      {tab === 'per-patient' && bd && (
        <div className="card shadow-sm mb-3">
          <div className="card-header py-2 fw-semibold small">
            Per-Patient Adherence — {bd.total_patients} patients (sorted by adherence)
          </div>
          <div className="table-responsive">
            <table className="table table-hover mb-0 small">
              <thead className="table-light">
                <tr>
                  <th>Patient</th><th>Drugs</th><th>Doses</th>
                  <th>On Time</th><th>Late</th><th>Missed</th>
                  <th>Adherence</th><th>Tier</th>
                  <th>Avg Delay</th><th>Avg SE</th><th>Avg Mood</th>
                </tr>
              </thead>
              <tbody>
                {bd.per_patient.map(p => (
                  <tr key={p.patient_id}>
                    <td className="fw-semibold">{p.patient_id}</td>
                    <td>{p.n_drugs}</td>
                    <td>{p.doses}</td>
                    <td className="text-success">{p.on_time}</td>
                    <td className="text-warning">{p.late}</td>
                    <td className="text-danger">{p.missed}</td>
                    <td>
                      <strong style={{ color: p.tier_color }}>{p.adherence_pct}%</strong>
                    </td>
                    <td>
                      <span className={`badge bg-${TIER_BADGE[p.tier]}`}>{p.tier}</span>
                    </td>
                    <td>{p.avg_minutes_late > 0 ? `${p.avg_minutes_late}m` : '—'}</td>
                    <td>{p.avg_se_severity}</td>
                    <td>{p.avg_mood}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── SIDE EFFECTS ── */}
      {tab === 'side-effects' && bd && (
        <>
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold small">Side Effect Severity Distribution by AED</div>
            <div className="table-responsive">
              <table className="table table-hover mb-0 small">
                <thead className="table-light">
                  <tr>
                    <th>Drug</th>
                    <th>None (0)</th>
                    <th>Mild (1–3)</th>
                    <th>Moderate (4–6)</th>
                    <th>Severe (7–10)</th>
                    <th>SE Rate %</th>
                    <th>Avg Severity</th>
                  </tr>
                </thead>
                <tbody>
                  {bd.se_by_drug.map(s => (
                    <tr key={s.drug}>
                      <td className="fw-semibold">{s.drug}</td>
                      <td className="text-success">{s.none}</td>
                      <td className="text-info">{s.mild}</td>
                      <td className="text-warning">{s.moderate}</td>
                      <td className="text-danger">{s.severe}</td>
                      <td>
                        <span className={`badge bg-${s.se_rate_pct >= 30 ? 'warning' : 'secondary'}`}>
                          {s.se_rate_pct}%
                        </span>
                      </td>
                      <td>{s.avg_se}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="alert alert-info small mb-3">
            <strong>Clinical note:</strong> Side effect rates 23–26% across all AEDs; Clobazam highest average severity (1.20).
            Patients experiencing moderate-to-severe AEs are 3.2× more likely to skip doses (Paschal 2021).
            Behavioral side effects (LEV irritability, TPM cognitive slowing) require proactive counselling.
          </div>
        </>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <>
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold small">Adherence Terminology</div>
            <div className="card-body">
              {defs.adherence_terminology.map(t => (
                <div key={t.term} className="mb-3 p-2 rounded border">
                  <div className="fw-semibold">{t.term}</div>
                  <div className="text-muted small mt-1">{t.definition}</div>
                  {t.formula && <code className="small">{t.formula}</code>}
                </div>
              ))}
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold small">Clinical Impact Findings</div>
            <div className="card-body">
              {defs.clinical_impact.map((ci, i) => (
                <div key={i} className="mb-3 p-2 rounded" style={{ background: '#f0f9ff', borderLeft: '4px solid #3b82f6' }}>
                  <div className="fw-semibold small">{ci.finding}</div>
                  <div className="text-muted" style={{ fontSize: 11 }}>{ci.detail}</div>
                  <div className="badge bg-secondary mt-1" style={{ fontSize: 10 }}>{ci.source}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold small">AED Pharmacological Profiles</div>
            <div className="card-body">
              <div className="row g-2">
                {defs.aed_profiles.map(p => (
                  <div key={p.drug} className="col-md-6">
                    <div className="p-2 border rounded mb-2">
                      <div className="fw-semibold">{p.drug}
                        <span className="badge bg-light text-dark ms-2 small">{p.class}</span>
                      </div>
                      <div className="text-muted small">{p.mechanism}</div>
                      <div className="small mt-1">
                        <span className="text-primary">Regimen:</span> {p.typical_regimen}
                      </div>
                      <div className="small text-warning">{p.behavioral_se}</div>
                      <div className="text-muted" style={{ fontSize: 11 }}>{p.note}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold small">References</div>
            <ul className="list-group list-group-flush small">
              {defs.references.map((r, i) => (
                <li key={i} className="list-group-item">{r}</li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  );
}

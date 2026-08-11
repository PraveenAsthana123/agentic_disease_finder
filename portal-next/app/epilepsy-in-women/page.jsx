'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TIER_COLOR = { HIGH: '#ef4444', MODERATE: '#f97316', LOW: '#22c55e', Unknown: '#94a3b8' };
const TIER_BG = { HIGH: '#fef2f2', MODERATE: '#fff7ed', LOW: '#f0fdf4', Unknown: '#f8fafc' };

function Badge({ text, color }) {
  return (
    <span style={{
      background: `${color}22`, color, border: `1px solid ${color}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600,
      textTransform: 'uppercase', whiteSpace: 'nowrap',
    }}>{text}</span>
  );
}

function TierBadge({ tier }) {
  const c = TIER_COLOR[tier] || '#94a3b8';
  return <Badge text={tier} color={c} />;
}

function KPI({ label, value, sub, color }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div style={{ fontSize: 28, fontWeight: 700, color: color || '#3b82f6' }}>{value ?? '—'}</div>
          <div className="text-muted small fw-semibold">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function MiniBar({ label, count, total, color }) {
  const pct = total ? Math.round((count / total) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-bold">{count} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" role="progressbar"
          style={{ width: `${pct}%`, background: color || '#3b82f6' }} />
      </div>
    </div>
  );
}

export default function EpilepsyInWomenDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/epilepsy-in-women/overview`).then(r => r.json()),
      fetch(`${API}/api/epilepsy-in-women/breakdown`).then(r => r.json()),
      fetch(`${API}/api/epilepsy-in-women/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading epilepsy in women data…</div>;

  const k = ov.kpis;
  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'aed', label: 'AED Safety' },
    { id: 'mentalhealth', label: 'Mental Health' },
    { id: 'patients', label: 'Per Patient' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: 28 }}>♀️</span>
        <div>
          <h4 className="mb-0 fw-bold">Epilepsy in Women</h4>
          <div className="text-muted small">AED teratogenicity · hormonal interactions · mental health · {k.total_female_patients} female EPAT patients</div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <>
          <div className="row">
            <KPI label="Female EPAT Patients" value={k.total_female_patients} color="#8b5cf6" />
            <KPI label="Childbearing Age (18–45)" value={k.childbearing_age_count} sub={`${k.childbearing_age_pct}% of cohort`} color="#ec4899" />
            <KPI label="HIGH Teratogenicity AED" value={k.high_teratogenicity_count} sub={`${k.high_teratogenicity_pct}% of cohort`} color="#ef4444" />
            <KPI label="CBA + HIGH Risk AED" value={k.high_risk_childbearing} sub="Priority counselling" color="#dc2626" />
          </div>
          <div className="row">
            <KPI label="Enzyme-Inducer AEDs" value={k.enzyme_inducer_count} sub="OCP interaction risk" color="#f97316" />
            <KPI label="Mean PHQ-9 Score" value={k.mean_phq9_score ?? '—'} sub="Depression screen" color="#0ea5e9" />
            <KPI label="Depression Rate (PHQ-9≥10)" value={`${k.depression_rate_pct}%`} color="#6366f1" />
            <KPI label="Anxiety Rate (GAD-7≥10)" value={`${k.anxiety_rate_pct}%`} color="#14b8a6" />
          </div>

          <div className="row mt-2">
            {/* Teratogenicity distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">AED Teratogenicity Tier Distribution</div>
                <div className="card-body">
                  {ov.teratogenicity_distribution.map(t => (
                    <div key={t.tier} className="mb-3">
                      <div className="d-flex justify-content-between mb-1">
                        <TierBadge tier={t.tier} />
                        <span className="fw-bold">{t.count} patients</span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div className="progress-bar"
                          style={{ width: `${Math.round(100 * t.count / k.total_female_patients)}%`, background: TIER_COLOR[t.tier] }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Age distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Age Distribution</div>
                <div className="card-body">
                  {ov.age_distribution.map(a => (
                    <MiniBar key={a.bucket} label={a.bucket} count={a.count} total={k.total_female_patients} color="#8b5cf6" />
                  ))}
                  <div className="text-muted small mt-2">Childbearing age (18–45) highlighted in purple</div>
                </div>
              </div>
            </div>

            {/* Top AEDs */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Most Used AEDs in Female Cohort</div>
                <div className="card-body">
                  {ov.aed_overview.most_used_aeds.map(a => (
                    <div key={a.aed} className="d-flex justify-content-between align-items-center mb-2">
                      <div className="d-flex align-items-center gap-2">
                        <TierBadge tier={a.tier} />
                        <span className="small fw-semibold">{a.aed}</span>
                      </div>
                      <span className="badge bg-secondary">{a.count}</span>
                    </div>
                  ))}
                  {ov.aed_overview.enzyme_inducers_in_use.length > 0 && (
                    <div className="mt-2 p-2 rounded" style={{ background: '#fff7ed' }}>
                      <div className="text-warning small fw-semibold">⚠ Enzyme-inducers in use:</div>
                      <div className="small">{ov.aed_overview.enzyme_inducers_in_use.join(', ')}</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Epilepsy type distribution */}
          <div className="card shadow-sm mt-1">
            <div className="card-header fw-semibold">Epilepsy Type Distribution (Female Cohort)</div>
            <div className="card-body d-flex flex-wrap gap-3">
              {ov.epilepsy_type_distribution.map(et => (
                <div key={et.epilepsy_type} className="text-center p-2 rounded border" style={{ minWidth: 100 }}>
                  <div className="h5 mb-0 fw-bold text-primary">{et.count}</div>
                  <div className="text-muted small">{et.epilepsy_type}</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* AED SAFETY TAB */}
      {tab === 'aed' && bd && (
        <>
          {/* Risk matrix */}
          <div className="row mb-3">
            {bd.risk_matrix.map((r, i) => (
              <div key={i} className="col-md-4 mb-2">
                <div className={`card border-${r.color} shadow-sm`}>
                  <div className="card-body p-2 d-flex align-items-center gap-3">
                    <div className={`h3 mb-0 text-${r.color} fw-bold`}>{r.count}</div>
                    <div className="small text-muted">{r.label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Tier detail */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">Teratogenicity Tier — Patient Roster</div>
            <div className="card-body">
              {bd.teratogenicity_tier_detail.map(t => (
                <div key={t.tier} className="mb-3 p-2 rounded"
                  style={{ background: TIER_BG[t.tier], borderLeft: `4px solid ${TIER_COLOR[t.tier]}` }}>
                  <div className="d-flex align-items-center gap-2 mb-1">
                    <TierBadge tier={t.tier} />
                    <span className="small fw-semibold">{t.count} patient{t.count !== 1 ? 's' : ''}</span>
                  </div>
                  <div className="small text-muted">{t.patients.join(', ') || 'None'}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Per-patient AED table */}
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Per-Patient AED Profile</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th>
                    <th>Age</th>
                    <th>CBA</th>
                    <th>AEDs</th>
                    <th>Tier</th>
                    <th>Enzyme Inducer</th>
                    <th>High-Risk AEDs</th>
                    <th>Drug Response</th>
                  </tr>
                </thead>
                <tbody>
                  {bd.per_patient.map(p => (
                    <tr key={p.patient_id} style={{ background: p.teratogenicity_tier === 'HIGH' ? '#fef2f2' : undefined }}>
                      <td className="fw-semibold small">{p.patient_id}</td>
                      <td className="small">{p.age}</td>
                      <td>{p.is_childbearing_age ? <span className="text-success fw-bold">Yes</span> : <span className="text-muted">No</span>}</td>
                      <td className="small">{p.aeds.join(', ') || '—'}</td>
                      <td><TierBadge tier={p.teratogenicity_tier} /></td>
                      <td>{p.has_enzyme_inducer ? <span className="text-warning fw-bold">⚠ Yes</span> : <span className="text-muted">No</span>}</td>
                      <td className="small text-danger">{p.high_risk_aeds.join(', ') || '—'}</td>
                      <td className="small">{p.drug_responsiveness}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* MENTAL HEALTH TAB */}
      {tab === 'mentalhealth' && bd && (
        <>
          <div className="row mb-3">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">PHQ-9 Depression Severity</div>
                <div className="card-body">
                  {bd.phq9_distribution.map(d => (
                    <div key={d.label} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{d.label} ({d.range})</span>
                        <span className="fw-bold">{d.count}</span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div className="progress-bar"
                          style={{
                            width: `${Math.round(100 * d.count / (bd.per_patient.filter(p => p.phq9_score !== null).length || 1))}%`,
                            background: d.label === 'Minimal' ? '#22c55e' : d.label === 'Mild' ? '#f97316' : d.label === 'Moderate' ? '#ef4444' : '#7f1d1d',
                          }} />
                      </div>
                    </div>
                  ))}
                  <div className="text-muted small mt-2">PHQ-9 ≥ 10 = likely depression. Epilepsy prevalence: ~30–35%.</div>
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">GAD-7 Anxiety Severity</div>
                <div className="card-body">
                  {bd.gad7_distribution.map(d => (
                    <div key={d.label} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{d.label} ({d.range})</span>
                        <span className="fw-bold">{d.count}</span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div className="progress-bar"
                          style={{
                            width: `${Math.round(100 * d.count / (bd.per_patient.filter(p => p.gad7_score !== null).length || 1))}%`,
                            background: d.label === 'Minimal' ? '#22c55e' : d.label === 'Mild' ? '#f97316' : d.label === 'Moderate' ? '#ef4444' : '#7f1d1d',
                          }} />
                      </div>
                    </div>
                  ))}
                  <div className="text-muted small mt-2">GAD-7 ≥ 10 = likely anxiety. Epilepsy prevalence: ~25–30%.</div>
                </div>
              </div>
            </div>
          </div>

          {/* Mental health per-patient */}
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Per-Patient Mental Health Scores</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th>
                    <th>Age</th>
                    <th>Epilepsy Type</th>
                    <th>PHQ-9</th>
                    <th>PHQ-9 Interp</th>
                    <th>GAD-7</th>
                    <th>GAD-7 Interp</th>
                    <th>NDDI-E</th>
                    <th>QOLIE-31</th>
                  </tr>
                </thead>
                <tbody>
                  {bd.per_patient.map(p => (
                    <tr key={p.patient_id}>
                      <td className="fw-semibold small">{p.patient_id}</td>
                      <td className="small">{p.age}</td>
                      <td className="small">{p.epilepsy_type}</td>
                      <td className="small">
                        {p.phq9_score !== null
                          ? <span style={{ color: p.phq9_score >= 10 ? '#ef4444' : '#22c55e', fontWeight: 700 }}>{p.phq9_score}</span>
                          : '—'}
                      </td>
                      <td className="small">{p.phq9_interpretation || '—'}</td>
                      <td className="small">
                        {p.gad7_score !== null
                          ? <span style={{ color: p.gad7_score >= 10 ? '#ef4444' : '#22c55e', fontWeight: 700 }}>{p.gad7_score}</span>
                          : '—'}
                      </td>
                      <td className="small">{p.gad7_interpretation || '—'}</td>
                      <td className="small">{p.nddi_score ?? '—'}</td>
                      <td className="small">{p.qolie_score ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* PER PATIENT TAB */}
      {tab === 'patients' && bd && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">All Female EPAT Patients — Full Clinical Profile</div>
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Age</th>
                  <th>Epilepsy Type</th>
                  <th>Onset Age</th>
                  <th>Yrs w/ Epilepsy</th>
                  <th>AEDs</th>
                  <th>Tier</th>
                  <th>CBA</th>
                  <th>Enzyme Ind.</th>
                  <th>Drug Response</th>
                  <th>Etiology</th>
                  <th>PHQ-9</th>
                  <th>GAD-7</th>
                  <th>Employment</th>
                  <th>Comorbidities</th>
                </tr>
              </thead>
              <tbody>
                {bd.per_patient.map(p => (
                  <tr key={p.patient_id}
                    style={{ background: p.teratogenicity_tier === 'HIGH' && p.is_childbearing_age ? '#fef2f2' : undefined }}>
                    <td className="fw-semibold small">{p.patient_id}</td>
                    <td className="small">{p.age}</td>
                    <td className="small">{p.epilepsy_type}</td>
                    <td className="small">{p.epilepsy_onset_age ?? '—'}</td>
                    <td className="small">{p.years_with_epilepsy ?? '—'}</td>
                    <td className="small">{p.aeds.join(', ') || '—'}</td>
                    <td><TierBadge tier={p.teratogenicity_tier} /></td>
                    <td>{p.is_childbearing_age ? '✓' : ''}</td>
                    <td>{p.has_enzyme_inducer ? <span className="text-warning">⚠</span> : ''}</td>
                    <td className="small">{p.drug_responsiveness}</td>
                    <td className="small">{p.etiology}</td>
                    <td className="small" style={{ color: p.phq9_score >= 10 ? '#ef4444' : undefined, fontWeight: p.phq9_score >= 10 ? 700 : undefined }}>{p.phq9_score ?? '—'}</td>
                    <td className="small" style={{ color: p.gad7_score >= 10 ? '#ef4444' : undefined, fontWeight: p.gad7_score >= 10 ? 700 : undefined }}>{p.gad7_score ?? '—'}</td>
                    <td className="small">{p.employment_status}</td>
                    <td className="small">{p.comorbidities?.length || 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <>
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">Dashboard Purpose</div>
            <div className="card-body"><p className="mb-0 small">{defs.dashboard_purpose}</p></div>
          </div>

          {/* Teratogenicity tiers */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">AED Teratogenicity Tiers (EURAP 2022)</div>
            <div className="card-body">
              {defs.teratogenicity_tiers.map(t => (
                <div key={t.tier} className="mb-3 p-3 rounded"
                  style={{ background: TIER_BG[t.tier], borderLeft: `4px solid ${TIER_COLOR[t.tier]}` }}>
                  <div className="d-flex align-items-center gap-2 mb-1">
                    <TierBadge tier={t.tier} />
                    <span className="small fw-semibold">{t.mcm_threshold}</span>
                  </div>
                  <div className="small mb-1"><strong>Drugs:</strong> {t.drugs.join(', ')}</div>
                  <div className="small text-muted">{t.guidance}</div>
                </div>
              ))}
            </div>
          </div>

          {/* AED reference table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">AED Safety Reference Table</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Drug</th>
                    <th>Tier</th>
                    <th>MCM Risk</th>
                    <th>Key Risk</th>
                    <th>Enzyme Inducer</th>
                    <th>OCP Interaction</th>
                    <th>Folic Acid (mg)</th>
                  </tr>
                </thead>
                <tbody>
                  {defs.aed_reference.map(a => (
                    <tr key={a.drug}>
                      <td className="fw-semibold small">{a.drug}</td>
                      <td><TierBadge tier={a.tier} /></td>
                      <td className="small">{a.mcm_risk_pct}</td>
                      <td className="small">{a.key_risk}</td>
                      <td className="small">{a.enzyme_inducer ? <span className="text-warning fw-bold">Yes</span> : 'No'}</td>
                      <td className="small">{a.hormonal_interaction}</td>
                      <td className="small">{a.folic_acid_dose_mg} mg</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Catamenial epilepsy */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">Catamenial Epilepsy</div>
            <div className="card-body">
              <p className="small mb-1"><strong>Definition:</strong> {defs.catamenial_epilepsy.definition}</p>
              <p className="small mb-1"><strong>Prevalence:</strong> {defs.catamenial_epilepsy.prevalence}</p>
              <p className="small mb-1"><strong>Mechanism:</strong> {defs.catamenial_epilepsy.hormonal_mechanism}</p>
              <strong className="small">Management options:</strong>
              <ul className="small mb-0 mt-1">
                {defs.catamenial_epilepsy.management_options.map((m, i) => <li key={i}>{m}</li>)}
              </ul>
            </div>
          </div>

          {/* Pregnancy guidance */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">Pregnancy Guidance (ILAE / NICE)</div>
            <div className="card-body">
              <ul className="small mb-0">
                {defs.pregnancy_guidance.map((g, i) => <li key={i} className="mb-1">{g}</li>)}
              </ul>
            </div>
          </div>

          {/* Mental health context */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">Mental Health in Women with Epilepsy</div>
            <div className="card-body">
              <ul className="small mb-0">
                {defs.mental_health_context.map((m, i) => <li key={i} className="mb-1">{m}</li>)}
              </ul>
            </div>
          </div>

          {/* Data sources */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">Data Sources</div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Table</th><th>Rows</th><th>Use</th></tr></thead>
                  <tbody>
                    {defs.data_sources.map((s, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold">{s.table}</td>
                        <td className="small">{s.rows}</td>
                        <td className="small">{s.use}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* References */}
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Clinical References</div>
            <div className="card-body">
              <ol className="small mb-0">
                {defs.clinical_references.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
              </ol>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

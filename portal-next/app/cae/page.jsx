'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Genetics', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function Bar({ label, value, max, color = '#1d4ed8' }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function PctBar({ label, pct, color = '#1d4ed8' }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const maxEtiol = Math.max(...(data.etiology_distribution || []).map(e => e.count), 1);
  const maxAed = Math.max(...(data.aed_use || []).map(a => a.n_patients), 1);
  const maxCtrl = Math.max(...(data.control_distribution || []).map(c => c.count), 1);
  const triggerColors = ['#dc2626', '#f59e0b', '#d97706', '#ef4444', '#0284c7', '#db2777', '#16a34a', '#9333ea'];

  return (
    <div>
      <div className="alert alert-info py-2 small mb-3">
        <strong>Childhood Absence Epilepsy (CAE):</strong> Most common <strong>generalized epilepsy of childhood</strong>
        — 10–15% of childhood epilepsies. Hallmark: <em>typical absences — abrupt-onset staring, 3 Hz spike-wave,
        5–30 sec, no post-ictal confusion, hyperventilation-provoked</em>.
        Female predominance (~60%). Drug resistance only <strong>~10–20%</strong>.
        <strong> Prognosis: 70–80% remit by adolescence (Berg 2001)</strong> — but 15–20% evolve to JME.
        CHILDHOOD 2010 (NEJM): ETX = first-line for pure absence (superior attention vs VPA).
      </div>

      <div className="row mb-4">
        {(data.kpis || []).map(k => <KPI key={k.label} {...k} />)}
      </div>

      {(data.clinical_alerts || []).length > 0 && (
        <div className="alert alert-danger py-2 mb-3">
          <strong>⚠️ Clinical Alerts:</strong>
          <ul className="mb-0 mt-1 small">
            {data.clinical_alerts.map((a, i) => <li key={i}>{a}</li>)}
          </ul>
        </div>
      )}

      <div className="row">
        <div className="col-md-4 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">Etiology Distribution</div>
            <div className="card-body">
              {(data.etiology_distribution || []).map((e, i) => {
                const colors = ['#1d4ed8', '#0891b2', '#16a34a', '#7c3aed', '#f59e0b', '#6b7280'];
                return (
                  <div key={e.etiology} className="mb-1">
                    <div className="d-flex justify-content-between small mb-1">
                      <span style={{ color: colors[i % colors.length] }}>{e.etiology}</span>
                      <span className="text-muted">{e.count} ({e.pct}%)</span>
                    </div>
                    <div className="progress" style={{ height: 10 }}>
                      <div className="progress-bar" style={{ width: `${(e.count / maxEtiol) * 100}%`, backgroundColor: colors[i % colors.length] }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">AED Regimen Distribution</div>
            <div className="card-body">
              {(data.aed_use || []).map(a => (
                <Bar key={a.regimen} label={a.regimen} value={a.n_patients} max={maxAed} color="#0891b2" />
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">Seizure Control Outcomes</div>
            <div className="card-body">
              {(data.control_distribution || []).map(c => {
                const colors = { 'Seizure-free': '#16a34a', 'Improved (>50% reduction)': '#f59e0b', 'Partial response': '#f97316', 'Drug-resistant': '#dc2626' };
                return (
                  <Bar key={c.status} label={c.status} value={c.count} max={maxCtrl} color={colors[c.status] || '#666'} />
                );
              })}
            </div>
          </div>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Onset Semiology Distribution</div>
            <div className="card-body">
              {(data.semiology_distribution || []).map(s => (
                <div key={s.semiology} className="mb-1">
                  <div className="d-flex justify-content-between small">
                    <span>{s.semiology}</span><span className="badge bg-info text-dark">{s.n}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Top Triggers</div>
            <div className="card-body">
              {(data.triggers || []).map((t, i) => (
                <PctBar key={t.trigger} label={t.trigger} pct={t.frequency_pct} color={triggerColors[i % triggerColors.length]} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading patient data…</div>;
  const [sort, setSort] = useState('patient_id');

  const sorted = [...(data.patients || [])].sort((a, b) => {
    if (sort === 'onset_age_years' || sort === 'years_on_aed' || sort === 'current_age') return a[sort] - b[sort];
    return String(a[sort]).localeCompare(String(b[sort]));
  });

  const controlColors = {
    'Seizure-free': '#16a34a',
    'Improved (>50% reduction)': '#f59e0b',
    'Partial response': '#f97316',
    'Drug-resistant': '#dc2626'
  };
  const etiolColors = ['#1d4ed8', '#0891b2', '#16a34a', '#7c3aed', '#f59e0b', '#6b7280'];

  return (
    <div>
      <h6 className="fw-bold mb-3">Genetic Etiology Catalog</h6>
      <div className="row mb-4">
        {(data.etiology_catalog || []).map((e, i) => (
          <div key={e.etiology} className="col-md-6 mb-3">
            <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${etiolColors[i % etiolColors.length]}` }}>
              <div className="card-header fw-bold d-flex justify-content-between small">
                <span>🧬 {e.etiology}</span>
                <span className="badge" style={{ background: etiolColors[i % etiolColors.length] }}>{e.pct}%</span>
              </div>
              <div className="card-body small">
                <div className="mb-1"><span className="badge bg-secondary">{e.category}</span></div>
                <div className="mb-1">{e.mechanism}</div>
                <div className="mb-1"><strong>EEG:</strong> <em>{e.eeg_correlate}</em></div>
                <div className="mb-1"><strong>MRI:</strong> <em>{e.mri_finding}</em></div>
                <div className="text-muted fst-italic">{e.clinical_note}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mb-2">
        Sort by:
        {['patient_id', 'onset_age_years', 'current_age', 'seizure_control', 'etiology', 'aed_regimen'].map(f => (
          <button key={f} onClick={() => setSort(f)}
            className={`btn btn-sm ms-2 ${sort === f ? 'btn-primary' : 'btn-outline-secondary'}`}>
            {f.replace(/_/g, ' ')}
          </button>
        ))}
      </div>

      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-dark">
            <tr>
              <th>Patient</th><th>Sex</th><th>Onset Age</th><th>Current Age</th>
              <th>Seizure Types</th><th>Etiology</th><th>AED Regimen</th><th>Control</th>
              <th>Catamenial</th><th>Remission</th><th>JME Evolution</th><th>Yrs on AED</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(p => (
              <tr key={p.patient_id}>
                <td>{p.patient_id}</td>
                <td>{p.sex}</td>
                <td>{p.onset_age_years}y</td>
                <td>{p.current_age}y</td>
                <td><span className="badge bg-primary small">{p.seizure_types}</span></td>
                <td><small>{p.etiology}</small></td>
                <td><small>{p.aed_regimen}</small></td>
                <td><span className="badge" style={{ background: controlColors[p.seizure_control] || '#666' }}>{p.seizure_control}</span></td>
                <td>{p.catamenial ? '♀️ Yes' : '—'}</td>
                <td>{p.remission ? '✅ Yes' : '—'}</td>
                <td>{p.jme_evolution ? '⚠️ JME' : '—'}</td>
                <td>{p.years_on_aed}y</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SeizureTriggersTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading seizure data…</div>;
  const typeColors = ['#1d4ed8', '#7c3aed', '#dc2626', '#0891b2'];

  return (
    <div>
      <h6 className="fw-bold mb-3">Seizure Types in CAE</h6>
      <div className="row mb-4">
        {(data.seizure_types || []).map((st, i) => (
          <div key={st.type} className="col-md-6 mb-3">
            <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${typeColors[i % typeColors.length]}` }}>
              <div className="card-header fw-bold d-flex justify-content-between small">
                <span>{st.type}</span>
                <span className="badge" style={{ background: typeColors[i % typeColors.length] }}>{st.prevalence_pct}%</span>
              </div>
              <div className="card-body small">
                <div className="mb-1 text-muted"><strong>Duration:</strong> {st.duration_sec}</div>
                <div className="mb-1">{st.description}</div>
                <div className="mb-1"><strong>EEG:</strong> <em>{st.eeg_correlate}</em></div>
                <div className="text-success fst-italic"><strong>Clinical tip:</strong> {st.clinical_tip}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold mb-3">Seizure Triggers & Mitigation</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-bordered">
          <thead className="table-dark">
            <tr><th>Trigger</th><th>Frequency</th><th>Mechanism</th><th>Patient Mitigation</th></tr>
          </thead>
          <tbody>
            {(data.triggers || []).map(t => (
              <tr key={t.trigger}>
                <td><strong>{t.trigger}</strong></td>
                <td>
                  <div className="progress" style={{ height: 16 }}>
                    <div className="progress-bar bg-danger" style={{ width: `${t.frequency_pct}%` }}>
                      {t.frequency_pct}%
                    </div>
                  </div>
                </td>
                <td><small>{t.mechanism}</small></td>
                <td><small>{t.management}</small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="alert alert-warning py-2 small">
        <strong>⚠️ ABSOLUTE CONTRAINDICATIONS — Pro-Absence Drugs (NEVER use in CAE):</strong>
        <ul className="mb-0 mt-1">
          <li><strong>Carbamazepine (CBZ / Tegretol):</strong> Aggravates absence seizures and can precipitate Absence Status Epilepticus — NEVER use in CAE</li>
          <li><strong>Oxcarbazepine (OXC / Trileptal):</strong> Similar pro-absence mechanism to CBZ — CONTRAINDICATED</li>
          <li><strong>Phenytoin (PHT / Dilantin):</strong> Na⁺ channel blockade without absence protection — worsens frequency</li>
          <li><strong>Vigabatrin (VGB / Sabril):</strong> GABA transaminase inhibitor → paradoxically worsens absences</li>
          <li><strong>Tiagabine (TGB) / Gabapentin (GBP) / Pregabalin (PGB):</strong> All documented to worsen absence seizures</li>
          <li><strong>Hyperventilation as diagnostic test (100% sensitivity in untreated CAE):</strong> HV-negative EEG on treatment = treatment success marker</li>
        </ul>
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading treatment data…</div>;

  return (
    <div>
      <h6 className="fw-bold mb-3">AED Therapies for CAE — CHILDHOOD 2010 Evidence Hierarchy</h6>
      <div className="alert alert-success py-2 small mb-3">
        <strong>CHILDHOOD 2010 (Glauser NEJM) — Level A Evidence:</strong>
        ETX = VPA (53% freedom at 16W); LTG inferior (29%) — NOT first-line.
        ETX preferred for pure absence (superior attention profile).
        VPA preferred when GTCS present or JME evolution suspected (broader spectrum).
      </div>
      <div className="row mb-4">
        {(data.treatments || []).map((tx, i) => {
          const colors = ['#16a34a', '#1d4ed8', '#0891b2', '#7c3aed', '#f59e0b', '#dc2626', '#0891b2', '#6b7280'];
          return (
            <div key={tx.drug} className="col-md-6 mb-3">
              <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${colors[i % colors.length]}` }}>
                <div className="card-header fw-bold small d-flex justify-content-between">
                  <span>{tx.drug} <small className="text-muted">({tx.brand})</small></span>
                  <span className="badge bg-secondary small">{tx.evidence_level}</span>
                </div>
                <div className="card-body small">
                  <div className="mb-1"><strong>Indication:</strong> <em>{tx.indication}</em></div>
                  <div className="mb-1"><strong>Dose:</strong> {tx.dose}</div>
                  <div className="mb-1"><strong>MOA:</strong> {tx.mechanism}</div>
                  <div className="mb-1"><strong>Efficacy:</strong> {tx.efficacy}</div>
                  <div className="mb-1 text-danger"><strong>Safety:</strong> {tx.safety}</div>
                  <div className="text-muted fst-italic"><strong>Monitoring:</strong> {tx.monitoring}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <h6 className="fw-bold mb-3">AED Monitoring Requirements</h6>
      {(data.aed_monitoring || []).map(m => (
        <div key={m.drug} className="card mb-3 shadow-sm">
          <div className="card-header fw-bold d-flex justify-content-between">
            <span>💊 {m.drug}</span>
          </div>
          <div className="card-body small">
            <div className="mb-2">
              <strong>Monitoring Parameters:</strong>
              <ul className="mb-0 mt-1">
                {(m.parameters || []).map((p, i) => <li key={i}>{p}</li>)}
              </ul>
            </div>
            <div>
              <strong className="text-danger">Clinical Alerts:</strong>
              <ul className="mb-0 mt-1">
                {(m.alerts || []).map((a, i) => <li key={i} className="text-danger">{a}</li>)}
              </ul>
            </div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mb-3">Lifecycle Management</h6>
      <div className="row">
        {(data.lifecycle || []).map((lc, i) => {
          const colors = ['#1d4ed8', '#dc2626', '#f59e0b', '#db2777', '#16a34a', '#7c3aed'];
          return (
            <div key={lc.phase} className="col-md-6 mb-3">
              <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${colors[i % colors.length]}` }}>
                <div className="card-header fw-bold small" style={{ color: colors[i % colors.length] }}>
                  {lc.phase}
                </div>
                <div className="card-body small">
                  <div className="mb-1">
                    <strong>Key Events:</strong>
                    <ul className="mb-1 mt-1 small">
                      {(lc.key_events || []).map((e, j) => <li key={j}>{e}</li>)}
                    </ul>
                  </div>
                  <div className="mb-1">
                    <strong>Clinical Priorities:</strong>
                    <ul className="mb-1 mt-1 small">
                      {(lc.clinical_priorities || []).slice(0, 3).map((p, j) => <li key={j}>{p}</li>)}
                    </ul>
                  </div>
                  <div className="text-primary fst-italic"><strong>Treatment focus:</strong> {lc.treatment_focus}</div>
                  {lc.warning_signs && (
                    <div className="text-danger mt-1"><strong>⚠️ Warning:</strong> {lc.warning_signs}</div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions…</div>;

  return (
    <div>
      <h6 className="fw-bold mb-3">Core Concepts — CAE (14 Definitions)</h6>
      <div className="row mb-3">
        {(data.concepts || []).map(c => (
          <div key={c.term} className="col-md-6 mb-2">
            <div className="card shadow-sm">
              <div className="card-body py-2">
                <span className="badge bg-primary me-2">{c.term}</span>
                <span className="small">{c.definition}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold mb-2">Clinical Standards</h6>
      <div className="row mb-3">
        {(data.standards || []).map(s => (
          <div key={s.standard} className="col-md-6 mb-2">
            <div className="card shadow-sm h-100">
              <div className="card-body py-2 small">
                <div className="fw-bold mb-1">{s.standard}</div>
                <div className="text-muted mb-1 fst-italic">{s.reference}</div>
                <ul className="mb-0 mt-1">
                  {(s.key_points || []).map((p, i) => <li key={i}>{p}</li>)}
                </ul>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold mb-2">Key Thresholds</h6>
      <div className="table-responsive mb-3">
        <table className="table table-sm table-bordered">
          <thead className="table-dark">
            <tr><th>Threshold</th><th>Value / Criterion</th><th>Rationale</th></tr>
          </thead>
          <tbody>
            {(data.thresholds || []).map(t => (
              <tr key={t.threshold}>
                <td><strong>{t.threshold}</strong></td>
                <td><code>{t.value}</code></td>
                <td><small>{t.rationale}</small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-2">Key References</h6>
      <div className="row">
        {(data.references || []).map(r => (
          <div key={r.citation} className="col-md-6 mb-2">
            <div className="card shadow-sm">
              <div className="card-body py-2 small">
                <div className="fw-bold mb-1">{r.citation}</div>
                <div className="text-muted mb-1">{r.full}</div>
                <div className="text-primary">{r.key_finding}</div>
                <span className="badge bg-success mt-1">{r.evidence_level}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function CAEDashboard() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/cae/overview`).then(r => r.json()).then(setOverview).catch(() => setError('Backend unreachable'));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3 && !breakdown) {
      fetch(`${API}/api/cae/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4 && !definitions) {
      fetch(`${API}/api/cae/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3">
        <span className="fs-3 me-2">🧒</span>
        <div>
          <h4 className="mb-0 fw-bold">Childhood Absence Epilepsy (CAE)</h4>
          <small className="text-muted">
            Most Common Childhood GGE · ICD-10: G40.309 · 10–15% of childhood epilepsies ·
            Drug resistance ~10–20% · 70–80% remit by adolescence (Berg 2001) ·
            ETX = First-line (CHILDHOOD 2010 NEJM · Glauser et al.)
          </small>
        </div>
      </div>

      {error && <div className="alert alert-danger py-2 small">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizureTriggersTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}

'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Surgery', 'Treatments', 'Definitions'];

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
        <strong>Temporal Lobe Epilepsy (TLE):</strong> Most common <strong>focal epilepsy</strong> (~60% of focal epilepsies).
        Hallmark: <em>déjà vu / epigastric rising aura → oral automatisms → postictal confusion</em>.
        EEG: temporal spike-wave + rhythmic theta. Drug resistance ~30–40% (highest of any common syndrome).
        <strong> Surgical cure: 60–70% seizure-free (Wiebe 2001 NEJM)</strong> — refer after ≥2 AED failures.
      </div>

      <div className="row mb-4">
        {(data.kpis || []).map(k => <KPI key={k.label} {...k} />)}
      </div>

      {(data.clinical_alerts || []).length > 0 && (
        <div className="alert alert-danger py-2 mb-3">
          <strong>Clinical Alerts:</strong>
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
              {(data.etiology_distribution || []).map(e => (
                <Bar key={e.etiology} label={`${e.etiology} (${e.pct}%)`} value={e.count} max={maxEtiol} color="#1d4ed8" />
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">AED Regimen Use</div>
            <div className="card-body">
              {(data.aed_use || []).map(a => (
                <Bar key={a.regimen} label={a.regimen} value={a.n_patients} max={maxAed} color="#0284c7" />
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">Seizure Control Status</div>
            <div className="card-body">
              {(data.control_distribution || []).map((c, i) => {
                const colors = ['#16a34a', '#f59e0b', '#f97316', '#dc2626'];
                return <Bar key={c.status} label={c.status} value={c.count} max={maxCtrl} color={colors[i % colors.length]} />;
              })}
            </div>
          </div>
        </div>
      </div>

      <div className="row mt-2">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Seizure Triggers (% of TLE patients affected)</div>
            <div className="card-body">
              {(data.triggers || []).map((t, i) => (
                <PctBar key={t.trigger} label={t.trigger} pct={t.frequency_pct} color={triggerColors[i % triggerColors.length]} />
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Aura Type Distribution</div>
            <div className="card-body">
              {(data.aura_distribution || []).map((a, i) => {
                const maxA = Math.max(...(data.aura_distribution || []).map(x => x.n), 1);
                return <Bar key={a.aura} label={a.aura} value={a.n} max={maxA} color="#7c3aed" />;
              })}
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
    if (sort === 'onset_age_years' || sort === 'years_on_aed') return a[sort] - b[sort];
    return String(a[sort]).localeCompare(String(b[sort]));
  });

  const controlColors = {
    'Seizure-free': '#16a34a',
    'Improved (>50% reduction)': '#f59e0b',
    'Partial response': '#f97316',
    'Drug-resistant': '#dc2626'
  };
  const etiolColors = ['#1d4ed8', '#7c3aed', '#0891b2', '#16a34a', '#f59e0b', '#dc2626'];

  return (
    <div>
      <h6 className="fw-bold mb-3">Etiology Catalog</h6>
      <div className="row mb-4">
        {(data.etiologies || []).map((e, i) => (
          <div key={e.etiology} className="col-md-6 mb-3">
            <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${etiolColors[i % etiolColors.length]}` }}>
              <div className="card-header fw-bold d-flex justify-content-between small">
                <span>🧠 {e.etiology}</span>
                <span className="badge" style={{ background: etiolColors[i % etiolColors.length] }}>{e.pct}%</span>
              </div>
              <div className="card-body small">
                <div className="mb-1"><span className="badge bg-secondary">{e.category}</span></div>
                <div className="mb-1">{e.mechanism}</div>
                <div className="mb-1"><strong>MRI:</strong> <em>{e.mri_finding}</em></div>
                <div className="mb-1 text-success"><strong>Surgical:</strong> {e.surgical_outcome}</div>
                <div className="text-muted fst-italic">{e.clinical_note}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mb-2">
        Sort by:
        {['patient_id', 'onset_age_years', 'seizure_control', 'etiology', 'aed_regimen'].map(f => (
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
              <th>Patient</th><th>Sex</th><th>Onset Age</th><th>Seizure Types</th>
              <th>Etiology</th><th>Aura</th><th>AED Regimen</th><th>Control</th>
              <th>Catamenial</th><th>Post-Surgical</th><th>Yrs on AED</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(p => (
              <tr key={p.patient_id}>
                <td>{p.patient_id}</td>
                <td>{p.sex}</td>
                <td>{p.onset_age_years}y</td>
                <td><span className="badge bg-primary small">{p.seizure_types}</span></td>
                <td><small>{p.etiology}</small></td>
                <td><small className="text-muted">{p.aura}</small></td>
                <td><small>{p.aed_regimen}</small></td>
                <td><span className="badge" style={{ background: controlColors[p.seizure_control] || '#666' }}>{p.seizure_control}</span></td>
                <td>{p.catamenial ? '🌙 Yes' : '—'}</td>
                <td>{p.post_surgical ? '✂️ Yes' : '—'}</td>
                <td>{p.years_on_aed}y</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SeizureSurgeryTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading seizure data…</div>;
  const typeColors = ['#1d4ed8', '#7c3aed', '#dc2626', '#0891b2'];

  return (
    <div>
      <h6 className="fw-bold mb-3">Seizure Types in TLE</h6>
      <div className="row mb-4">
        {(data.seizure_types || []).map((st, i) => (
          <div key={st.type} className="col-md-6 mb-3">
            <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${typeColors[i % typeColors.length]}` }}>
              <div className="card-header fw-bold d-flex justify-content-between small">
                <span>{st.type}</span>
                <span className="badge" style={{ background: typeColors[i % typeColors.length] }}>{st.prevalence_pct}%</span>
              </div>
              <div className="card-body small">
                <div className="mb-1">{st.description}</div>
                <div className="mb-1"><strong>EEG:</strong> <em>{st.eeg}</em></div>
                <div className="mb-1 text-success fst-italic"><strong>Clinical tip:</strong> {st.clinical_tip}</div>
                <div className="mb-1"><strong>Triggers:</strong> {st.triggers}</div>
                <div><strong>First-line AED:</strong> <span className="badge bg-success">{st.first_line_aed}</span></div>
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
                <td><small>{t.mitigation}</small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="alert alert-success py-2 small">
        <strong>✂️ SURGICAL PATHWAY (Level A Evidence — Wiebe 2001 NEJM / ERSET 2012 NEJM)</strong>
        <ul className="mb-0 mt-1">
          <li><strong>Trigger:</strong> ≥2 adequate AED failures (ILAE DRE definition 2010) → mandatory Level 4 epilepsy centre referral</li>
          <li><strong>Pre-surgical workup:</strong> Video-EEG (seizure capture) + 3T MRI epilepsy protocol + FDG-PET + neuropsychology + WADA test / fMRI language lateralisation</li>
          <li><strong>If MRI-negative:</strong> SEEG (stereo-EEG) depth electrodes + MEG + 7T MRI consideration → tailored resection</li>
          <li><strong>Outcome MTLE-HS:</strong> 60–70% Engel Class I (seizure-free) at 2 years post-ATL</li>
          <li><strong>SAH (Selective Amygdalohippocampectomy):</strong> comparable seizure outcome to ATL with better verbal memory preservation</li>
          <li><strong>ERSET trial:</strong> 73% seizure-free with early surgery vs 0% continued medical therapy → do NOT delay referral</li>
        </ul>
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading treatment data…</div>;

  return (
    <div>
      <h6 className="fw-bold mb-3">AED Therapies & Surgical Options for TLE</h6>
      <div className="row mb-4">
        {(data.treatments || []).map((tx, i) => {
          const colors = ['#1d4ed8', '#0284c7', '#16a34a', '#0891b2', '#7c3aed', '#9333ea', '#f59e0b', '#6b7280'];
          return (
            <div key={tx.drug} className="col-md-6 mb-3">
              <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${colors[i % colors.length]}` }}>
                <div className="card-header fw-bold small d-flex justify-content-between">
                  <span>{tx.drug}</span>
                  <span className="badge bg-secondary small">{tx.type}</span>
                </div>
                <div className="card-body small">
                  <div className="mb-1"><strong>Dose:</strong> {tx.dose}</div>
                  <div className="mb-1"><strong>MOA:</strong> {tx.moa}</div>
                  <div className="mb-1"><strong>Efficacy:</strong> {tx.efficacy}</div>
                  <div className="mb-1 text-danger"><strong>Safety:</strong> {tx.safety}</div>
                  <div className="text-muted fst-italic">{tx.fda_status}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <h6 className="fw-bold mb-3">AED Monitoring Requirements</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-bordered">
          <thead className="table-dark">
            <tr><th>Drug</th><th>Category</th><th>Risk</th><th>Monitoring</th><th>Mitigation</th></tr>
          </thead>
          <tbody>
            {(data.aed_monitoring || []).map(m => (
              <tr key={m.drug}>
                <td><strong>{m.drug}</strong></td>
                <td><span className="badge bg-danger small">{m.category}</span></td>
                <td><small>{m.risk}</small></td>
                <td><small>{m.monitoring}</small></td>
                <td><small>{m.mitigation}</small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-3">Lifecycle Management</h6>
      <div className="row">
        {(data.lifecycle || []).map((lc, i) => {
          const colors = ['#1d4ed8', '#dc2626', '#f59e0b', '#db2777', '#16a34a', '#7c3aed'];
          return (
            <div key={lc.age_window} className="col-md-6 mb-3">
              <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${colors[i % colors.length]}` }}>
                <div className="card-header fw-bold small" style={{ color: colors[i % colors.length] }}>
                  {lc.age_window}
                </div>
                <div className="card-body small">
                  <div className="mb-1"><strong>Profile:</strong> {lc.typical_profile}</div>
                  <div className="mb-1"><strong>TLE indicators:</strong> {lc.tle_indicators}</div>
                  <div className="text-primary"><strong>Action:</strong> {lc.action}</div>
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
      <h6 className="fw-bold mb-3">Core Concepts</h6>
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
      <div className="table-responsive mb-3">
        <table className="table table-sm table-bordered">
          <thead className="table-dark"><tr><th>Standard</th><th>Body</th><th>Scope</th></tr></thead>
          <tbody>
            {(data.standards || []).map(s => (
              <tr key={s.name}>
                <td><strong>{s.name}</strong></td>
                <td><span className="badge bg-info text-dark">{s.body}</span></td>
                <td><small>{s.scope}</small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-2">Key Thresholds</h6>
      <div className="table-responsive mb-3">
        <table className="table table-sm table-bordered">
          <thead className="table-dark"><tr><th>Parameter</th><th>Threshold</th><th>Unit</th><th>Significance</th></tr></thead>
          <tbody>
            {(data.thresholds || []).map(t => (
              <tr key={t.parameter}>
                <td><strong>{t.parameter}</strong></td>
                <td><code>{t.threshold}</code></td>
                <td><span className="badge bg-secondary">{t.unit}</span></td>
                <td><small>{t.significance}</small></td>
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
                <div className="text-muted">{r.key_finding}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function TLEDashboard() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/tle/overview`).then(r => r.json()).then(setOverview).catch(() => setError('Backend unreachable'));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3 && !breakdown) {
      fetch(`${API}/api/tle/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4 && !definitions) {
      fetch(`${API}/api/tle/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3">
        <span className="fs-3 me-2">🧠</span>
        <div>
          <h4 className="mb-0 fw-bold">Temporal Lobe Epilepsy (TLE)</h4>
          <small className="text-muted">
            Most Common Focal Epilepsy · ICD-10: G40.209 / G40.219 · ~60% of focal epilepsies ·
            Drug resistance ~30–40% · Surgery: 60–70% seizure-free (Wiebe 2001 NEJM)
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
      {tab === 2 && <SeizureSurgeryTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}

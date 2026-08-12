'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

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

function PctBar({ label, pct, color = '#7c3aed' }) {
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

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 mb-2`} style={{ fontSize: 13 }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = '#7c3aed' }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-header fw-bold" style={{ backgroundColor: '#f5f3ff', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

export default function DooseDashboard() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [patientSearch, setPatientSearch] = useState('');
  const [patientSort, setPatientSort] = useState({ key: 'patient_id', dir: 1 });

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/doose/overview`).then(r => r.json()),
      fetch(`${API}/api/doose/breakdown`).then(r => r.json()),
      fetch(`${API}/api/doose/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading Doose Syndrome Dashboard…</div>;
  if (!overview || !breakdown) return <div className="container py-5 text-center text-danger">Failed to load data.</div>;

  const etDist = overview.etiology_distribution || {};
  const prognosis = overview.prognosis_summary || {};

  const patients = breakdown.patients || [];
  const filtered = patients.filter(p =>
    !patientSearch ||
    String(p.patient_id).toLowerCase().includes(patientSearch.toLowerCase()) ||
    (p.etiology || '').toLowerCase().includes(patientSearch.toLowerCase()) ||
    (p.seizure_control || '').toLowerCase().includes(patientSearch.toLowerCase())
  );
  const sorted = [...filtered].sort((a, b) => {
    const av = a[patientSort.key] ?? ''; const bv = b[patientSort.key] ?? '';
    return (av < bv ? -1 : av > bv ? 1 : 0) * patientSort.dir;
  });
  function toggleSort(k) {
    setPatientSort(s => ({ key: k, dir: s.key === k ? -s.dir : 1 }));
  }
  function Th({ k, label }) {
    const active = patientSort.key === k;
    return <th style={{ cursor: 'pointer', whiteSpace: 'nowrap' }} onClick={() => toggleSort(k)}>
      {label} {active ? (patientSort.dir === 1 ? '▲' : '▼') : '⇅'}
    </th>;
  }

  const CTRL_BADGE = { 'Drug-resistant': 'danger', 'Partial control': 'warning', 'Seizure-free': 'success' };
  const PHASE_BADGE = { 'Active': 'warning', 'Near-remission': 'info', 'Remission': 'success' };

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      <div className="mb-3">
        <h2 className="fw-bold mb-0" style={{ color: '#7c3aed' }}>🧒 Doose Syndrome — Myoclonic-Atonic Epilepsy (MAE)</h2>
        <div className="text-muted small">
          Self-limited GGE · Drop-attack epilepsy · 50% remission · KD Level A first-line ·{' '}
          {overview.total_patients} patients · {overview.generated}
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 0 && (
        <>
          <div className="row g-2 mb-4">
            <KPI label="Total Patients" value={overview.total_patients} color="#7c3aed" />
            <KPI label="Male" value={`${overview.male_n} (${overview.male_pct}%)`} color="#2563eb" />
            <KPI label="Drop Attacks Active" value={`${overview.drop_attacks_active_n} (${overview.drop_attacks_active_pct}%)`} color="#dc2626" />
            <KPI label="On Ketogenic Diet" value={`${overview.kd_n} (${overview.kd_pct}%)`} color="#16a34a" />
            <KPI label="Seizure-Free" value={`${overview.seizure_free_n} (${overview.seizure_free_pct}%)`} color="#16a34a" />
            <KPI label="Intell. Disability" value={`${overview.intellectual_disability_n} (${overview.intellectual_disability_pct}%)`} color="#f59e0b" />
          </div>

          <SectionCard title="⚠️ Clinical Safety Alerts" borderColor="#dc2626">
            {(overview.clinical_alerts || []).map((a, i) => (
              <Alert key={i} text={a} variant={a.startsWith('⛔') ? 'danger' : a.startsWith('✅') ? 'success' : 'warning'} />
            ))}
          </SectionCard>

          <div className="row mb-4">
            <div className="col-md-6">
              <SectionCard title="📊 Etiology Distribution" borderColor="#7c3aed">
                {Object.entries(etDist).sort((a, b) => b[1] - a[1]).map(([et, n]) => (
                  <PctBar key={et} label={et} pct={Math.round(n / overview.total_patients * 100)} />
                ))}
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="✅ Prognosis Summary" borderColor="#16a34a">
                <table className="table table-sm table-bordered mb-0">
                  <tbody>
                    {Object.entries(prognosis).map(([k, v]) => (
                      <tr key={k}>
                        <td className="small fw-bold text-muted" style={{ whiteSpace: 'nowrap' }}>
                          {k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                        </td>
                        <td className="small">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </SectionCard>
            </div>
          </div>

          <SectionCard title="📚 References" borderColor="#6b7280">
            <ol className="mb-0">
              {(overview.references || []).map((r, i) => <li key={i} className="small mb-1">{r}</li>)}
            </ol>
          </SectionCard>
        </>
      )}

      {/* ── Patients & Etiology ── */}
      {tab === 1 && (
        <>
          <div className="accordion mb-4" id="etAcc">
            {(breakdown.etiology_catalog || []).map((et, i) => (
              <div className="accordion-item" key={i}>
                <h2 className="accordion-header">
                  <button className="accordion-button collapsed fw-bold" type="button"
                    data-bs-toggle="collapse" data-bs-target={`#et${i}`}>
                    🧬 {et.etiology} — {et.pct}%
                    <span className="badge ms-2" style={{ backgroundColor: '#7c3aed', fontSize: 11 }}>{et.category}</span>
                  </button>
                </h2>
                <div id={`et${i}`} className="accordion-collapse collapse" data-bs-parent="#etAcc">
                  <div className="accordion-body">
                    {et.mechanism && <><strong>Mechanism:</strong><p className="small">{et.mechanism}</p></>}
                    {et.eeg_correlate && <><strong>EEG Correlate:</strong><p className="small">{et.eeg_correlate}</p></>}
                    {et.mri_finding && <><strong>MRI Finding:</strong><p className="small">{et.mri_finding}</p></>}
                    {et.clinical_note && (
                      <div className="alert alert-warning py-2 mt-2">
                        <strong>Clinical Note:</strong> <span className="small">{et.clinical_note}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <SectionCard title={`👥 Patient Register (${filtered.length}/${patients.length})`} borderColor="#7c3aed">
            <div className="mb-2">
              <input className="form-control form-control-sm" style={{ maxWidth: 300 }}
                placeholder="Search patient ID / etiology / control…"
                value={patientSearch} onChange={e => setPatientSearch(e.target.value)} />
            </div>
            <div className="table-responsive" style={{ maxHeight: 480, overflowY: 'auto' }}>
              <table className="table table-sm table-hover table-bordered">
                <thead className="table-light sticky-top">
                  <tr>
                    <Th k="patient_id" label="Patient" />
                    <Th k="age" label="Age" />
                    <Th k="gender" label="Sex" />
                    <Th k="onset_age" label="Onset" />
                    <Th k="etiology" label="Etiology" />
                    <Th k="primary_seizure_type" label="Seizure Type" />
                    <Th k="current_aed" label="AED" />
                    <Th k="seizure_control" label="Control" />
                    <Th k="disease_phase" label="Phase" />
                    <Th k="drop_attacks_active" label="Drop Attacks" />
                    <Th k="kd_on_ketogenic_diet" label="KD" />
                    <Th k="intellectual_disability" label="ID" />
                  </tr>
                </thead>
                <tbody>
                  {sorted.map(p => (
                    <tr key={p.patient_id}>
                      <td><strong>{p.patient_id}</strong></td>
                      <td>{p.age}</td>
                      <td>{p.gender}</td>
                      <td>{p.onset_age}y</td>
                      <td><span className="badge text-bg-light border small">{p.etiology}</span></td>
                      <td><span className="small">{p.primary_seizure_type}</span></td>
                      <td><code className="small">{p.current_aed}</code></td>
                      <td>
                        <span className={`badge text-bg-${CTRL_BADGE[p.seizure_control] || 'secondary'}`}>
                          {p.seizure_control}
                        </span>
                      </td>
                      <td>
                        <span className={`badge text-bg-${PHASE_BADGE[p.disease_phase] || 'secondary'}`}>
                          {p.disease_phase}
                        </span>
                      </td>
                      <td className="text-center">{p.drop_attacks_active ? '⚠️' : '—'}</td>
                      <td className="text-center">{p.kd_on_ketogenic_diet ? '🥑' : '—'}</td>
                      <td className="text-center">{p.intellectual_disability ? '📚' : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Seizure Types & Triggers ── */}
      {tab === 2 && (
        <>
          <SectionCard title="⚡ Seizure Types in Doose Syndrome (MAE)" borderColor="#7c3aed">
            {(breakdown.seizure_types || []).map((st, i) => (
              <div key={i} className="mb-4 border-bottom pb-3">
                <div className="d-flex align-items-center mb-2">
                  <span className="fw-bold me-3">{st.type}</span>
                  <span className="badge" style={{ backgroundColor: '#7c3aed' }}>{st.freq_pct}% of MAE</span>
                  {st.duration_sec && <span className="badge bg-secondary ms-2">{st.duration_sec}</span>}
                </div>
                <p className="small mb-1">{st.description}</p>
                {st.eeg_correlate && <div className="alert alert-light py-1 small mb-1"><strong>EEG:</strong> {st.eeg_correlate}</div>}
                {st.clinical_tip && <div className="alert alert-info py-1 small mb-0"><strong>Clinical Tip:</strong> {st.clinical_tip}</div>}
              </div>
            ))}
          </SectionCard>

          <SectionCard title="🎯 Seizure Triggers" borderColor="#f59e0b">
            <div className="table-responsive">
              <table className="table table-sm table-bordered">
                <thead className="table-light">
                  <tr><th>Trigger</th><th>Prevalence</th><th>Mechanism</th><th>Management</th></tr>
                </thead>
                <tbody>
                  {(breakdown.triggers || []).sort((a, b) => b.pct - a.pct).map((tr, i) => (
                    <tr key={i}>
                      <td><strong>{tr.trigger}</strong></td>
                      <td>
                        <div className="progress" style={{ height: 10, minWidth: 80 }}>
                          <div className="progress-bar" style={{ width: `${tr.pct}%`, backgroundColor: '#f59e0b' }} />
                        </div>
                        <div className="text-muted small">{tr.pct}%</div>
                      </td>
                      <td className="small">{tr.mechanism}</td>
                      <td className="small">{tr.management}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Treatments ── */}
      {tab === 3 && (
        <>
          <div className="alert alert-danger mb-3">
            <strong>⛔ Absolute Contraindications in ALL MAE:</strong>
            <ul className="mb-0 mt-1">
              <li><strong>Carbamazepine (CBZ)</strong> — aggravates myoclonic-atonic seizures; can cause myoclonic status</li>
              <li><strong>Oxcarbazepine (OXC)</strong> — same Na⁺-channel mechanism as CBZ; worsens drop attacks</li>
              <li><strong>Phenytoin (PHT)</strong> — worsens myoclonic-atonic seizures; avoid absolutely</li>
              <li><strong>Vigabatrin (VGB)</strong> — aggravates myoclonus + irreversible visual field loss</li>
              <li><strong>Lamotrigine (LTG)</strong> — ⚠️ RELATIVE: worsens myoclonus in 15–20%; extreme caution</li>
            </ul>
          </div>

          <div className="alert alert-success mb-3">
            <strong>✅ Ketogenic Diet is Level A Evidence in MAE</strong> — consider early (after 1 failed AED), not as last resort.
            50–60% responder rate is the highest KD efficacy of any GGE syndrome.
          </div>

          <div className="accordion mb-4" id="txAcc">
            {(breakdown.treatments || []).map((tx, i) => (
              <div className="accordion-item" key={i}>
                <h2 className="accordion-header">
                  <button className="accordion-button collapsed fw-bold" type="button"
                    data-bs-toggle="collapse" data-bs-target={`#tx${i}`}>
                    💊 {tx.drug}
                    <span className="badge bg-success ms-2" style={{ fontSize: 11 }}>{tx.evidence}</span>
                    {tx.contraindication_note && (
                      <span className="badge bg-warning text-dark ms-2" style={{ fontSize: 10, maxWidth: 220, whiteSpace: 'normal', textAlign: 'left' }}>
                        ⚠️ {tx.contraindication_note}
                      </span>
                    )}
                  </button>
                </h2>
                <div id={`tx${i}`} className="accordion-collapse collapse" data-bs-parent="#txAcc">
                  <div className="accordion-body">
                    <div className="row g-3">
                      <div className="col-md-6">
                        <strong>Adult Dose:</strong> <span className="small">{tx.dose_adult}</span>
                        {tx.dose_paed && <><br /><strong>Paediatric:</strong> <span className="small">{tx.dose_paed}</span></>}
                      </div>
                      <div className="col-md-6">
                        <strong>Evidence:</strong> <span className="small">{tx.evidence_ref}</span>
                      </div>
                      <div className="col-12">
                        <strong>Mechanism:</strong><p className="small mb-1">{tx.moa}</p>
                        <strong>Efficacy:</strong> <span className="small">{tx.efficacy}</span>
                        <br /><strong>Safety:</strong> <span className="small">{tx.safety}</span>
                        <br /><strong>Monitoring:</strong>
                        <div className="alert alert-warning py-1 mt-1 small">{tx.monitoring}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <SectionCard title="🩺 AED Monitoring Protocol" borderColor="#7c3aed">
            {(breakdown.aed_monitoring || []).map((m, i) => (
              <div key={i} className="mb-3 border-bottom pb-2">
                <div className="fw-bold small">{m.item}</div>
                <div className="text-muted small">Frequency: {m.frequency}</div>
                <div className="small mt-1">{m.rationale}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="📅 MAE Lifecycle Management" borderColor="#2563eb">
            {(breakdown.lifecycle || []).map((lc, i) => (
              <div key={i} className="mb-3 border-bottom pb-2">
                <div className="fw-bold">{lc.window}</div>
                <div className="small text-muted mb-1">Age: {lc.age_range}</div>
                <div className="small mb-1"><strong>Key events:</strong> {lc.key_events}</div>
                <div className="small"><strong>Clinical focus:</strong> {lc.focus}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="📋 Standards & Thresholds" borderColor="#6b7280">
            <div className="row g-3">
              <div className="col-md-6">
                <strong className="small">Clinical Standards</strong>
                {(breakdown.standards || []).map((s, i) => (
                  <div key={i} className="border rounded p-2 mb-2 bg-light">
                    <div className="fw-bold small">{s.standard}</div>
                    <div className="text-muted small">{s.relevance}</div>
                  </div>
                ))}
              </div>
              <div className="col-md-6">
                <strong className="small">Decision Thresholds</strong>
                <table className="table table-sm table-bordered mt-1">
                  <thead className="table-light"><tr><th>Threshold</th><th>Value</th></tr></thead>
                  <tbody>
                    {(breakdown.thresholds || []).map((th, i) => (
                      <tr key={i}>
                        <td className="small">{th.threshold}</td>
                        <td className="small fw-bold">{th.value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Definitions ── */}
      {tab === 4 && (
        <>
          <SectionCard title="⛔ Contraindications in Doose Syndrome / MAE" borderColor="#dc2626">
            <div className="table-responsive">
              <table className="table table-sm table-bordered">
                <thead className="table-light">
                  <tr><th>Drug / Approach</th><th>Contraindicated In</th><th>Consequence</th></tr>
                </thead>
                <tbody>
                  {(definitions?.absolute_contraindications || []).map((ci, i) => (
                    <tr key={i}>
                      <td>
                        <strong className={ci.drug.includes('RELATIVE') ? 'text-warning' : 'text-danger'}>
                          {ci.drug}
                        </strong>
                      </td>
                      <td>{ci.contraindicated_in}</td>
                      <td className="small">{ci.consequence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="📖 Glossary — Doose Syndrome / MAE Concepts" borderColor="#7c3aed">
            <div className="accordion" id="conceptAcc">
              {(definitions?.concepts || []).map((c, i) => (
                <div className="accordion-item" key={i}>
                  <h2 className="accordion-header">
                    <button className="accordion-button collapsed" type="button"
                      data-bs-toggle="collapse" data-bs-target={`#concept${i}`}>
                      <strong>{c.term}</strong>
                    </button>
                  </h2>
                  <div id={`concept${i}`} className="accordion-collapse collapse" data-bs-parent="#conceptAcc">
                    <div className="accordion-body small">{c.definition}</div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="🎯 Decision Thresholds" borderColor="#2563eb">
            <table className="table table-sm table-bordered">
              <thead className="table-light"><tr><th>Threshold</th><th>Value</th></tr></thead>
              <tbody>
                {(definitions?.thresholds || []).map((th, i) => (
                  <tr key={i}>
                    <td className="small">{th.threshold}</td>
                    <td className="small fw-bold">{th.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </SectionCard>

          <SectionCard title="📚 References" borderColor="#6b7280">
            <ol className="mb-0">
              {(definitions?.references || []).map((r, i) => <li key={i} className="small mb-1">{r}</li>)}
            </ol>
          </SectionCard>
        </>
      )}
    </div>
  );
}

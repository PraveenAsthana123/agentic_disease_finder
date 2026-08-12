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

function Alert({ text, variant = 'danger' }) {
  return (
    <div className={`alert alert-${variant} py-2 mb-2`} style={{ fontSize: 13 }}>
      ⚠️ {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = '#7c3aed' }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-header fw-bold" style={{ backgroundColor: '#f9f5ff', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

export default function PMEDashboard() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [patientSearch, setPatientSearch] = useState('');
  const [patientSort, setPatientSort] = useState({ key: 'patient_id', dir: 1 });

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/pme/overview`).then(r => r.json()),
      fetch(`${API}/api/pme/breakdown`).then(r => r.json()),
      fetch(`${API}/api/pme/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading PME Dashboard…</div>;
  if (!overview || !breakdown) return <div className="container py-5 text-center text-danger">Failed to load data.</div>;

  const etDist = overview.etiology_distribution || {};
  const stageDist = overview.disease_stage_distribution || {};
  const STAGE_COLORS = { Mild: '#16a34a', Moderate: '#f59e0b', Advanced: '#dc2626', 'End-stage': '#6b21a8' };

  // Patient table
  const patients = (breakdown.patients || []);
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

  const STAGE_BADGE = { Mild: 'success', Moderate: 'warning', Advanced: 'danger', 'End-stage': 'dark' };
  const CTRL_BADGE = { 'Drug-resistant': 'danger', 'Partial control': 'warning', 'Seizure-free': 'success' };

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      <div className="mb-3">
        <h2 className="fw-bold mb-0" style={{ color: '#7c3aed' }}>🧬 Progressive Myoclonic Epilepsy (PME)</h2>
        <div className="text-muted small">
          Rare inherited epilepsies: cortical myoclonus + GTCS + progressive neurological deterioration
          (ULD/Lafora/MERRF/NCL/Sialidosis) · {overview.total_patients} patients · {overview.generated}
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
          {/* KPIs */}
          <div className="row g-2 mb-4">
            <KPI label="Total Patients" value={overview.total_patients} color="#7c3aed" />
            <KPI label="Drug-Resistant" value={`${overview.drug_resistant_n} (${overview.drug_resistant_pct}%)`} color="#dc2626" />
            <KPI label="Photosensitive (PPR)" value={`${overview.photosensitive_n} (${overview.photosensitive_pct}%)`} color="#f59e0b" />
            <KPI label="Progressive Ataxia" value={`${overview.progressive_ataxia_n} (${overview.progressive_ataxia_pct}%)`} color="#0891b2" />
            <KPI label="Cognitive Decline" value={`${overview.cognitive_decline_n} (${overview.cognitive_decline_pct}%)`} color="#6d28d9" />
            <KPI label="VPA Contraindicated" value={`${overview.vpa_contraindicated_n} (${overview.vpa_contraindicated_pct}%)`} color="#b91c1c" />
          </div>

          {/* Alerts */}
          <SectionCard title="⛔ Clinical Safety Alerts — MANDATORY" borderColor="#b91c1c">
            {(overview.clinical_alerts || []).map((a, i) => <Alert key={i} text={a} />)}
          </SectionCard>

          {/* Etiology + Stage side by side */}
          <div className="row mb-4">
            <div className="col-md-6">
              <SectionCard title="🧬 Etiology Distribution" borderColor="#7c3aed">
                {Object.entries(etDist).sort((a, b) => b[1] - a[1]).map(([et, n]) => (
                  <PctBar key={et} label={et} pct={Math.round(n / overview.total_patients * 100)} />
                ))}
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="📊 Disease Stage Distribution" borderColor="#0891b2">
                {Object.entries(stageDist).sort((a, b) => b[1] - a[1]).map(([stage, n]) => (
                  <PctBar key={stage} label={stage} pct={Math.round(n / overview.total_patients * 100)}
                    color={STAGE_COLORS[stage] || '#7c3aed'} />
                ))}
              </SectionCard>
            </div>
          </div>

          {/* Subtype Prognosis */}
          <SectionCard title="📋 Subtype Prognosis Summary" borderColor="#6d28d9">
            <div className="table-responsive">
              <table className="table table-sm table-bordered">
                <thead className="table-light">
                  <tr><th>Subtype</th><th>Prognosis</th></tr>
                </thead>
                <tbody>
                  {Object.entries(overview.subtype_prognosis || {}).map(([sub, prog]) => (
                    <tr key={sub}>
                      <td><strong>{sub.replace(/_/g, ' ').toUpperCase()}</strong></td>
                      <td>{prog}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          {/* References */}
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
          {/* Etiology Catalog */}
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

          {/* Patient Table */}
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
                    <Th k="primary_seizure_type" label="Primary Seizure" />
                    <Th k="current_aed" label="Current AED" />
                    <Th k="seizure_control" label="Control" />
                    <Th k="disease_stage" label="Stage" />
                    <Th k="photosensitive" label="Photo" />
                    <Th k="progressive_ataxia" label="Ataxia" />
                    <Th k="vpa_contraindicated" label="VPA CI" />
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
                        <span className={`badge text-bg-${STAGE_BADGE[p.disease_stage] || 'secondary'}`}>
                          {p.disease_stage}
                        </span>
                      </td>
                      <td className="text-center">{p.photosensitive ? '✅' : '—'}</td>
                      <td className="text-center">{p.progressive_ataxia ? '✅' : '—'}</td>
                      <td className="text-center">{p.vpa_contraindicated ? '⛔' : '—'}</td>
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
          <SectionCard title="⚡ Seizure Types" borderColor="#7c3aed">
            {(breakdown.seizure_types || []).map((st, i) => (
              <div key={i} className="mb-4 border-bottom pb-3">
                <div className="d-flex align-items-center mb-2">
                  <span className="fw-bold me-3">{st.type}</span>
                  <span className="badge" style={{ backgroundColor: '#7c3aed' }}>{st.freq_pct}% of PME</span>
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
            <strong>⛔ Absolute Contraindications in ALL PME:</strong>
            <ul className="mb-0 mt-1">
              <li><strong>CBZ / OXC / PHT</strong> — exacerbate cortical myoclonus in ALL PME subtypes</li>
              <li><strong>Valproate (VPA)</strong> — CONTRAINDICATED in MERRF (MT-TK mutation) — fatal hepatotoxicity</li>
              <li><strong>Vigabatrin (VGB)</strong> — worsens myoclonus + irreversible visual field loss</li>
              <li><strong>Gabapentin / Pregabalin</strong> — paradoxical myoclonus worsening</li>
            </ul>
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
                      <span className="badge bg-danger ms-2" style={{ fontSize: 10, maxWidth: 200, whiteSpace: 'normal', textAlign: 'left' }}>
                        {tx.contraindication_note}
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

          <SectionCard title="🩺 AED Monitoring Protocol" borderColor="#0891b2">
            {(breakdown.aed_monitoring || []).map((m, i) => (
              <div key={i} className="mb-3 border-bottom pb-2">
                <div className="fw-bold small">{m.item}</div>
                <div className="text-muted small">Frequency: {m.frequency}</div>
                <div className="small mt-1">{m.rationale}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="📅 PME Lifecycle Management" borderColor="#6d28d9">
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
          {/* Absolute Contraindications */}
          <SectionCard title="⛔ Absolute Contraindications" borderColor="#b91c1c">
            <div className="table-responsive">
              <table className="table table-sm table-bordered">
                <thead className="table-light">
                  <tr><th>Drug</th><th>Contraindicated In</th><th>Consequence</th></tr>
                </thead>
                <tbody>
                  {(definitions?.absolute_contraindications || []).map((ci, i) => (
                    <tr key={i}>
                      <td><strong className="text-danger">{ci.drug}</strong></td>
                      <td>{ci.contraindicated_in}</td>
                      <td className="small">{ci.consequence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          {/* Concepts */}
          <SectionCard title="📖 Glossary — PME Concepts" borderColor="#7c3aed">
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

          {/* Thresholds */}
          <SectionCard title="🎯 Decision Thresholds" borderColor="#0891b2">
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

          {/* References */}
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

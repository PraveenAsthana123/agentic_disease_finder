'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a5276';   // deep navy — GRIN2A / NMDA receptor / neuroscience
const ACCENT2 = '#922b21';   // deep red — CBZ absolute CI / danger
const ACCENT3 = '#1e8449';   // dark green — CSWS suppression / treatment success

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

function PctBar({ label, pct, color = ACCENT }) {
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

function SectionCard({ title, children, borderColor = ACCENT }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-header fw-bold" style={{ backgroundColor: '#eaf4fb', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function TabBtn({ label, active, onClick }) {
  return (
    <button
      className={`btn btn-sm me-1 mb-1 ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
      onClick={onClick}
    >{label}</button>
  );
}

function EtiologyCard({ e }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="card mb-3 shadow-sm">
      <div className="card-header d-flex justify-content-between align-items-center"
           style={{ backgroundColor: '#eaf4fb', cursor: 'pointer' }}
           onClick={() => setOpen(o => !o)}>
        <span className="fw-bold" style={{ color: ACCENT }}>{e.etiology}</span>
        <span className="badge" style={{ backgroundColor: ACCENT }}>{e.n} pts ({e.pct}%)</span>
      </div>
      {open && (
        <div className="card-body small">
          <p><strong>Mechanism:</strong> {e.mechanism}</p>
          <p><strong>EEG Signature:</strong> {e.eeg_signature}</p>
          <p><strong>MRI:</strong> {e.mri}</p>
          <p><strong>Clinical Note:</strong> {e.clinical_note}</p>
        </div>
      )}
    </div>
  );
}

function SeizureCard({ s }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="card mb-3 shadow-sm">
      <div className="card-header d-flex justify-content-between align-items-center"
           style={{ backgroundColor: '#fdf2e9', cursor: 'pointer' }}
           onClick={() => setOpen(o => !o)}>
        <span className="fw-bold" style={{ color: '#784212' }}>{s.type}</span>
        <span className="badge bg-warning text-dark">{s.prevalence_pct}%</span>
      </div>
      {open && (
        <div className="card-body small">
          <p><strong>Onset Age:</strong> {s.onset_age}</p>
          <p><strong>EEG Correlate:</strong> {s.eeg_correlate}</p>
          <p><strong>Clinical Tip:</strong> {s.clinical_tip}</p>
        </div>
      )}
    </div>
  );
}

function TriggerCard({ t }) {
  const [open, setOpen] = useState(false);
  const isCI = t.trigger.toLowerCase().includes('cbz') || t.trigger.toLowerCase().includes('oxc');
  return (
    <div className="card mb-2 shadow-sm">
      <div className="card-header d-flex justify-content-between align-items-center"
           style={{ backgroundColor: isCI ? '#fdecea' : '#f0f3f4', cursor: 'pointer' }}
           onClick={() => setOpen(o => !o)}>
        <span className="fw-bold" style={{ color: isCI ? ACCENT2 : '#2c3e50', fontSize: 13 }}>{t.trigger}</span>
        <span className="badge" style={{ backgroundColor: isCI ? ACCENT2 : '#5d6d7e' }}>{t.rate_pct}%</span>
      </div>
      {open && (
        <div className="card-body small">
          <p><strong>Mechanism:</strong> {t.mechanism}</p>
          <p><strong>Management:</strong> {t.management}</p>
        </div>
      )}
    </div>
  );
}

function TreatmentCard({ tx }) {
  const [open, setOpen] = useState(false);
  const isCI = tx.drug.toLowerCase().includes('contraindication') || tx.drug.toLowerCase().includes('absolute');
  return (
    <div className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${isCI ? ACCENT2 : ACCENT3}` }}>
      <div className="card-header d-flex justify-content-between align-items-center"
           style={{ backgroundColor: isCI ? '#fdecea' : '#eafaf1', cursor: 'pointer' }}
           onClick={() => setOpen(o => !o)}>
        <div>
          <span className="fw-bold" style={{ color: isCI ? ACCENT2 : ACCENT3 }}>{tx.drug}</span>
          <span className="ms-2 badge" style={{ backgroundColor: isCI ? ACCENT2 : ACCENT }}>{tx.evidence_level.split(' (')[0]}</span>
        </div>
        <span className="text-muted small">{tx.role}</span>
      </div>
      {open && (
        <div className="card-body small">
          {tx.dose && tx.dose !== 'N/A — contraindicated' && <p><strong>Dose:</strong> {tx.dose}</p>}
          <p><strong>MOA:</strong> {tx.moa}</p>
          <p><strong>Efficacy:</strong> {tx.efficacy}</p>
          {tx.monitoring && tx.monitoring !== 'N/A — do not use.' && <p><strong>Monitoring:</strong> {tx.monitoring}</p>}
          <p><strong>Safety:</strong> {tx.safety}</p>
        </div>
      )}
    </div>
  );
}

function PatientTable({ patients }) {
  if (!patients || !patients.length) return <p className="text-muted">No patient data.</p>;
  return (
    <div className="table-responsive">
      <table className="table table-sm table-striped table-hover small">
        <thead className="table-dark">
          <tr>
            <th>ID</th><th>Onset (Y)</th><th>Sex</th><th>Etiology</th>
            <th>SWI (%)</th><th>CSWS</th><th>LKS</th><th>Treatment</th><th>Language Outcome</th>
          </tr>
        </thead>
        <tbody>
          {patients.map(p => (
            <tr key={p.patient_id}>
              <td>{p.patient_id}</td>
              <td>{p.age_onset_years}</td>
              <td>{p.sex}</td>
              <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                  title={p.etiology_class}>{p.etiology_class}</td>
              <td>
                <span style={{ color: p.swi_pct >= 85 ? ACCENT2 : p.swi_pct >= 50 ? '#d68910' : ACCENT3 }}>
                  {p.swi_pct}%
                </span>
              </td>
              <td>
                <span className={`badge ${p.csws === 'Yes' ? 'bg-danger' : p.csws === 'Moderate' ? 'bg-warning text-dark' : 'bg-success'}`}>
                  {p.csws}
                </span>
              </td>
              <td>{p.lks_phenotype === 'Yes' ? <span className="badge bg-danger">Yes</span> : <span className="text-muted">No</span>}</td>
              <td style={{ maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                  title={p.current_tx}>{p.current_tx}</td>
              <td>
                <span className={`badge ${p.language_outcome === 'Full recovery' ? 'bg-success' : p.language_outcome === 'Partial recovery' ? 'bg-warning text-dark' : 'bg-danger'}`}>
                  {p.language_outcome}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function GRIN2ADashboard() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/grin2a/overview`).then(r => r.json()),
      fetch(`${API}/api/grin2a/breakdown`).then(r => r.json()),
      fetch(`${API}/api/grin2a/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOverview(o); setBreakdown(b); setDefs(d); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;
  if (!overview) return <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}><div className="spinner-border text-primary" /><span className="ms-3">Loading GRIN2A dashboard…</span></div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="card mb-4 shadow" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, #154360 100%)`, color: '#fff' }}>
        <div className="card-body py-3">
          <h3 className="mb-1 fw-bold">&#129516; GRIN2A Epilepsy-Aphasia Spectrum (GRIN2A-EAS)</h3>
          <div className="small opacity-75">
            <strong>Gene:</strong> {overview.gene} &nbsp;|&nbsp;
            <strong>Locus:</strong> {overview.locus} &nbsp;|&nbsp;
            <strong>Protein:</strong> {overview.protein} &nbsp;|&nbsp;
            <strong>Cohort:</strong> N={overview.cohort_n}
          </div>
          <div className="small opacity-75 mt-1">{overview.condition}</div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert text="&#128721; CBZ / OXC ABSOLUTE CONTRAINDICATION — Na⁺ channel blockers paradoxically worsen CSWS, accelerating language regression. STOP CBZ immediately if CSWS identified." variant="danger" />
      <Alert text="&#128270; Sleep EEG (overnight PSG) MANDATORY — waking EEG alone grossly underestimates CSWS burden. Any child with centrotemporal spikes + language concerns MUST have PSG." variant="warning" />
      <Alert text="&#128680; Anti-NMDAr antibody (CSF + serum) MANDATORY in acute-onset LKS — autoimmune encephalitis mimics LKS and requires immunotherapy, not AEDs." variant="warning" />

      {/* KPIs */}
      <div className="row mb-3">
        {overview.kpis.map(k => (
          <KPI key={k.label} label={k.label} value={k.value}
               color={k.label.includes('CSWS') || k.label.includes('LKS') ? ACCENT2 : ACCENT} />
        ))}
      </div>

      {/* Tabs */}
      <div className="mb-3">
        {TABS.map(t => <TabBtn key={t} label={t} active={tab === t} onClick={() => setTab(t)} />)}
      </div>

      {/* ── Tab: Overview ── */}
      {tab === 'Overview' && (
        <div className="row">
          <div className="col-md-6">
            <SectionCard title="Etiology Distribution (N=41)">
              {overview.etiology_summary.map(e => (
                <PctBar key={e.label} label={e.label} pct={e.pct} color={ACCENT} />
              ))}
            </SectionCard>
            <SectionCard title="Seizure Type Prevalence">
              {overview.seizure_prevalence.map(s => (
                <PctBar key={s.type} label={s.type} pct={s.pct} color="#d68910" />
              ))}
            </SectionCard>
          </div>
          <div className="col-md-6">
            <SectionCard title="Trigger Rates">
              {overview.trigger_prevalence.map(t => (
                <PctBar key={t.trigger} label={t.trigger} pct={t.pct}
                        color={t.trigger.toLowerCase().includes('cbz') ? ACCENT2 : '#5d6d7e'} />
              ))}
            </SectionCard>
            <SectionCard title="Key Clinical Concept" borderColor={ACCENT3}>
              <p className="small">{overview.key_concept}</p>
              <hr />
              <strong className="small">Clinical Alerts:</strong>
              {overview.top_clinical_alerts.map((a, i) => (
                <Alert key={i} text={a} variant={a.toLowerCase().includes('contraindic') || a.toLowerCase().includes('absolute') ? 'danger' : 'warning'} />
              ))}
            </SectionCard>
          </div>
        </div>
      )}

      {/* ── Tab: Patients & Etiology ── */}
      {tab === 'Patients & Etiology' && breakdown && (
        <div>
          <SectionCard title="Etiology Catalog — 5 Classes (click to expand)">
            {breakdown.etiology_catalog.map(e => <EtiologyCard key={e.category} e={e} />)}
          </SectionCard>
          <SectionCard title="Patient Cohort (N=41)">
            <p className="small text-muted mb-2">SWI ≥85% = CSWS (red) · SWI 50–85% = Moderate (amber) · SWI &lt;50% = No CSWS (green)</p>
            <PatientTable patients={breakdown.patients} />
          </SectionCard>
        </div>
      )}

      {/* ── Tab: Seizure Types & Triggers ── */}
      {tab === 'Seizure Types & Triggers' && breakdown && (
        <div className="row">
          <div className="col-md-6">
            <SectionCard title="Seizure Types (4) — click to expand">
              {breakdown.seizure_types.map(s => <SeizureCard key={s.type} s={s} />)}
            </SectionCard>
          </div>
          <div className="col-md-6">
            <SectionCard title="Triggers & Management (8) — click to expand">
              <Alert text="CBZ / OXC initiation is listed as an iatrogenic TRIGGER of CSWS worsening — it is absolutely contraindicated." variant="danger" />
              {breakdown.triggers.map(t => <TriggerCard key={t.trigger} t={t} />)}
            </SectionCard>
          </div>
        </div>
      )}

      {/* ── Tab: Treatments ── */}
      {tab === 'Treatments' && breakdown && (
        <div>
          <Alert text="TREATMENT STRATEGY: VPA + nocturnal CLB (70–80% dose at bedtime) are first-line for CSWS. Add ACTH (4–6 weeks) for Landau-Kleffner with active aphasia. CBZ/OXC ABSOLUTELY CONTRAINDICATED." variant="danger" />
          <div className="row">
            <div className="col-md-8">
              <SectionCard title="Treatment Ladder (8) — click to expand">
                {breakdown.treatments.map(tx => <TreatmentCard key={tx.drug} tx={tx} />)}
              </SectionCard>
            </div>
            <div className="col-md-4">
              <SectionCard title="Contraindications" borderColor={ACCENT2}>
                {breakdown.contraindications.map(c => (
                  <div key={c.drug} className="mb-3">
                    <div className="fw-bold small" style={{ color: ACCENT2 }}>{c.severity}</div>
                    <div className="small fw-bold">{c.drug}</div>
                    <div className="small text-muted">{c.reason?.slice(0, 220)}…</div>
                  </div>
                ))}
              </SectionCard>
              <SectionCard title="Monitoring Checklist" borderColor={ACCENT3}>
                {breakdown.monitoring.map(m => (
                  <div key={m.item} className="mb-2 small">
                    <div className="fw-bold" style={{ color: ACCENT3 }}>{m.item}</div>
                    <div className="text-muted">{m.frequency}</div>
                  </div>
                ))}
              </SectionCard>
              <SectionCard title="Lifecycle Windows (6)" borderColor="#7d3c98">
                {breakdown.lifecycle.map(l => (
                  <div key={l.window} className="mb-2 small">
                    <div className="fw-bold" style={{ color: '#7d3c98' }}>{l.window}</div>
                    <div className="text-muted">{l.management_focus?.slice(0, 140)}…</div>
                  </div>
                ))}
              </SectionCard>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab: Definitions ── */}
      {tab === 'Definitions' && defs && (
        <div className="row">
          <div className="col-md-6">
            <SectionCard title="Key Concepts (14)">
              {defs.concepts.map(c => (
                <div key={c.term} className="mb-3">
                  <div className="fw-bold small" style={{ color: ACCENT }}>{c.term.replace(/-/g, ' ')}</div>
                  <div className="small text-muted">{c.definition}</div>
                </div>
              ))}
            </SectionCard>
          </div>
          <div className="col-md-6">
            <SectionCard title="Clinical Thresholds (10)" borderColor={ACCENT2}>
              {defs.thresholds.map(t => (
                <div key={t.threshold} className="mb-2 small d-flex">
                  <span className="badge me-2" style={{ backgroundColor: ACCENT2, minWidth: 60, whiteSpace: 'normal', height: 'fit-content' }}>{t.value}</span>
                  <div>
                    <div className="fw-bold">{t.threshold}</div>
                    <div className="text-muted">{t.action}</div>
                  </div>
                </div>
              ))}
            </SectionCard>
            <SectionCard title="Clinical Standards (8)" borderColor={ACCENT3}>
              {defs.standards.map(s => (
                <div key={s.code} className="mb-2 small">
                  <span className="badge me-1" style={{ backgroundColor: ACCENT3 }}>{s.code}</span>
                  <span className="text-muted">{s.scope}</span>
                </div>
              ))}
            </SectionCard>
            <SectionCard title="Key References (6)">
              {defs.references.map(r => (
                <div key={r.ref} className="mb-2 small">
                  <div className="fw-bold" style={{ color: ACCENT }}>{r.ref}</div>
                  <div className="text-muted">{r.citation}</div>
                  <div className="text-muted fst-italic">{r.impact}</div>
                </div>
              ))}
            </SectionCard>
          </div>
        </div>
      )}
    </div>
  );
}

'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a3a6e';   // deep navy — LGI1 / auditory / lateral temporal
const ACCENT2 = '#7b1c1c';   // deep crimson — contraindications / danger
const ACCENT3 = '#1a5c2e';   // deep forest green — good prognosis / seizure-free
const ACCENT4 = '#5c3a00';   // amber — autoimmune distinction / caution

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#eaf0f7', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: 11 }}>
      {text}
    </span>
  );
}

export default function LGI1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/lgi1/overview`).then(r => r.json()),
      fetch(`${API}/api/lgi1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/lgi1/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error) return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="card mb-4 shadow" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, #2a5298 100%)`, color: '#fff' }}>
        <div className="card-body py-3">
          <h2 className="mb-1 fw-bold">&#x1f9e0; LGI1 Epilepsy / ADLTE</h2>
          <div className="small opacity-90">
            <strong>Gene:</strong> LGI1 · <strong>Locus:</strong> 10q23.33 · <strong>OMIM:</strong> #600512 ·
            <strong> Syndrome:</strong> Autosomal Dominant Lateral Temporal Epilepsy (ADLTE / ADPEAF / EPILEPSY-AUDITORY)
          </div>
          <div className="small opacity-80 mt-1">
            <strong>Protein:</strong> Leucine-Rich Glioma Inactivated 1 — secreted synaptic adhesion protein;
            LGI1–ADAM22–ADAM23 trans-synaptic complex; stabilises AMPA receptors + Kv1.1 in lateral temporal cortex
          </div>
          <div className="small opacity-80 mt-1">
            <strong>Inheritance:</strong> Autosomal Dominant · ~70% penetrance (30% non-penetrant) ·
            <strong> Prognosis: </strong>
            <span style={{ color: '#a8f0b0', fontWeight: 'bold' }}>EXCELLENT — 65–70% seizure-free on CBZ/OXC (best of genetic focal epilepsies)</span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <div className="row mb-3">
        <div className="col-12">
          <Alert variant="danger" text={`⛔ TIAGABINE ABSOLUTE CI — NCSE risk (GAT-1 → tonic GABA-A). Class-effect in all focal cortical epilepsies including ADLTE.`} />
          <Alert variant="warning" text={`🔬 HLA-B*15:02 MANDATORY before CBZ/OXC in Asian-ancestry patients — SJS/TEN risk 10×. CPIC 2023.`} />
          <Alert variant="warning" text={`⚠️ CRITICAL DISTINCTION: LGI1 AUTOIMMUNE ENCEPHALITIS (anti-LGI1 Ab + FBDS + limbic encephalitis) ≠ GENETIC ADLTE (auditory aura + AD family history). If subacute onset/FBDS: check CSF anti-LGI1 Ab; treat with immunotherapy NOT AEDs alone.`} />
          <Alert variant="success" text={`✅ GOOD PROGNOSIS: ADLTE is NOT a DEE. Most patients achieve seizure freedom on CBZ/OXC. MRI typically normal. Standard employment + quality of life achievable.`} />
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ── */}
      {tab === 0 && overview && (
        <>
          <div className="row g-3 mb-4">
            <KPI label="Patients" value={overview.n_patients} color={ACCENT} />
            <KPI label="Seizure-Free" value={`${overview.seizure_free_pct}%`} color={ACCENT3} />
            <KPI label="Drug-Resistant" value={`${overview.drug_resistant_pct}%`} color={ACCENT2} />
            <KPI label="Auditory Aura" value={`${overview.auditory_aura_pct}%`} color={ACCENT} />
            <KPI label="FBTCS" value={`${overview.fbtcs_pct}%`} color={ACCENT4} />
            <KPI label="MRI Normal" value={`${overview.mri_normal_pct}%`} color={ACCENT3} />
            <KPI label="HLA Tested" value={`${overview.hla_b1502_tested_pct}%`} color={ACCENT} />
            <KPI label="POLG Done" value={`${overview.polg_done_pct}%`} color={ACCENT} />
            <KPI label="SIADH Events" value={overview.siadh_events_n} color={ACCENT4} />
            <KPI label="Non-Penetrant" value={`${overview.non_penetrant_pct}%`} color='#888' />
          </div>

          <SectionCard title="Gene & Disease Summary" borderColor={ACCENT}>
            <div className="row">
              <div className="col-md-6">
                <table className="table table-sm table-bordered">
                  <tbody>
                    <tr><th>Gene</th><td>{overview.gene}</td></tr>
                    <tr><th>Locus</th><td>{overview.locus}</td></tr>
                    <tr><th>OMIM</th><td>{overview.omim}</td></tr>
                    <tr><th>Inheritance</th><td>{overview.inheritance}</td></tr>
                    <tr><th>Incidence</th><td>{overview.incidence}</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="col-md-6">
                <p className="small mb-0">{overview.summary}</p>
              </div>
            </div>
          </SectionCard>

          <SectionCard title="Clinical Alerts — Safety Checklist" borderColor={ACCENT2}>
            <Alert variant="danger" text={`⛔ TIAGABINE: ${overview.tiagabine_alert}`} />
            <Alert variant="warning" text={`🔬 HLA-B*15:02: ${overview.hla_alert}`} />
            <Alert variant="warning" text={`🧬 POLG: ${overview.polg_alert}`} />
            <Alert variant="warning" text={`🤰 VPPP: ${overview.vppp_alert}`} />
            <Alert variant="info" text={`🔵 AUTOIMMUNE DISTINCTION: ${overview.autoimmune_alert}`} />
            <Alert variant="success" text={`✅ PROGNOSIS: ${overview.prognosis_note}`} />
          </SectionCard>

          <SectionCard title="Contraindications Summary" borderColor={ACCENT2}>
            <ul className="mb-0">
              {overview.contraindications_summary.map((ci, i) => (
                <li key={i} className="mb-1 small">{ci}</li>
              ))}
            </ul>
          </SectionCard>

          <SectionCard title="Key Thresholds" borderColor={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-striped mb-0">
                <thead><tr><th>Metric</th><th>Target</th><th>Alert</th></tr></thead>
                <tbody>
                  {overview.thresholds.map((t, i) => (
                    <tr key={i}>
                      <td className="small fw-bold">{t.metric}</td>
                      <td className="small">{t.target}</td>
                      <td className="small text-danger">{t.alert}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Tab 1: Patients & Etiology ── */}
      {tab === 1 && breakdown && (
        <>
          <SectionCard title="Cohort Summary" borderColor={ACCENT}>
            <div className="row">
              <div className="col-md-6">
                <PctBar label="Seizure-Free" pct={breakdown.summary.seizure_free_pct} color={ACCENT3} />
                <PctBar label="Drug-Resistant" pct={breakdown.summary.drug_resistant_pct} color={ACCENT2} />
                <PctBar label="Auditory Aura" pct={breakdown.summary.auditory_aura_pct} color={ACCENT} />
                <PctBar label="FBTCS Events" pct={breakdown.summary.fbtcs_pct} color={ACCENT4} />
              </div>
              <div className="col-md-6">
                <PctBar label="HLA-B*15:02 Tested" pct={breakdown.summary.hla_b1502_tested_pct} color={ACCENT} />
                <PctBar label="POLG Done" pct={breakdown.summary.polg_done_pct} color={ACCENT} />
                <PctBar label="MRI Normal" pct={breakdown.summary.mri_normal_pct} color={ACCENT3} />
                <PctBar label="Non-Penetrant" pct={breakdown.summary.non_penetrant_pct} color='#888' />
              </div>
            </div>
          </SectionCard>

          <SectionCard title="Etiology Distribution (5 Classes)" borderColor={ACCENT}>
            {breakdown.etiology_distribution.map((e, i) => (
              <div key={i} className="mb-3 pb-2 border-bottom">
                <div className="d-flex justify-content-between align-items-start mb-1">
                  <span className="fw-bold small">{e.etiology}</span>
                  <span className="badge" style={{ backgroundColor: ACCENT }}>{e.pct}% (n={e.n})</span>
                </div>
                <div className="progress mb-1" style={{ height: 8 }}>
                  <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ACCENT }} />
                </div>
                <div className="text-muted small">{e.mechanism_short}…</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Etiology Catalog — Full Detail" borderColor={ACCENT}>
            {breakdown.etiology_catalog.map((ec, i) => (
              <div key={i} className="mb-4 pb-3 border-bottom">
                <h6 className="fw-bold" style={{ color: ACCENT }}>{ec.etiology} — {ec.pct}% (n={ec.n})</h6>
                <div className="row">
                  <div className="col-md-6">
                    <p className="small mb-1"><strong>Mechanism:</strong> {ec.mechanism}</p>
                    <p className="small mb-0"><strong>EEG Correlate:</strong> {ec.eeg_correlate}</p>
                  </div>
                  <div className="col-md-6">
                    <p className="small mb-1"><strong>Key Treatments:</strong></p>
                    <div className="mb-2">
                      {ec.key_treatments.map((t, j) => <Badge key={j} text={t} color={ACCENT} />)}
                    </div>
                    <p className="small mb-0"><strong>Clinical Tip:</strong> {ec.clinical_tip}</p>
                  </div>
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Patient Sample (first 15)" borderColor={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-striped mb-0" style={{ fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>ID</th><th>Etiology</th><th>Age Onset</th><th>Sex</th>
                    <th>Seizure-Free</th><th>Seizure Types</th><th>Current AED</th>
                    <th>HLA</th><th>POLG</th><th>SIADH</th><th>Family Hx</th><th>MRI Normal</th>
                  </tr>
                </thead>
                <tbody>
                  {breakdown.patients_sample.map((p, i) => (
                    <tr key={i}>
                      <td>{p.id}</td>
                      <td style={{ maxWidth: 120, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={p.etiology}>{p.etiology}</td>
                      <td>{p.age_onset ?? 'NP'}</td>
                      <td>{p.sex}</td>
                      <td style={{ color: p.seizure_free ? ACCENT3 : ACCENT2 }}>
                        {p.seizure_free ? '✅ Yes' : (p.drug_resistant ? '🔴 DR' : '⚠️ No')}
                      </td>
                      <td style={{ maxWidth: 120, fontSize: 11 }}>{p.seizure_types.join(', ') || (p.non_penetrant ? 'Non-penetrant' : '—')}</td>
                      <td>{typeof p.aed_current === 'object' ? p.aed_current.join('+') : p.aed_current}</td>
                      <td>{p.hla_b1502_tested ? (p.hla_b1502_pos ? '⚠️ POS' : '✅ neg') : '❓'}</td>
                      <td>{p.polg_done ? '✅' : '❌'}</td>
                      <td>{p.siadh_event ? '⚠️ Yes' : 'No'}</td>
                      <td>{p.family_history ? '✅' : 'No'}</td>
                      <td>{p.mri_normal ? '✅ Nml' : '⚠️'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Tab 2: Seizures & Triggers ── */}
      {tab === 2 && breakdown && (
        <>
          <SectionCard title="Seizure Types (5 types) — Prevalence" borderColor={ACCENT}>
            {breakdown.seizure_types.map((s, i) => (
              <PctBar key={i} label={s.type} pct={s.prevalence_pct} color={ACCENT} />
            ))}
          </SectionCard>

          <SectionCard title="Seizure Type Detail" borderColor={ACCENT}>
            {breakdown.seizure_detail.map((s, i) => (
              <div key={i} className="mb-4 pb-3 border-bottom">
                <h6 className="fw-bold" style={{ color: ACCENT }}>{s.type} — {s.prevalence_pct}%</h6>
                <div className="row">
                  <div className="col-md-4">
                    <p className="small mb-1"><strong>EEG Signature:</strong></p>
                    <p className="small text-muted">{s.eeg_signature}</p>
                  </div>
                  <div className="col-md-4">
                    <p className="small mb-1"><strong>Semiology:</strong></p>
                    <p className="small text-muted">{s.semiology}</p>
                  </div>
                  <div className="col-md-4">
                    <p className="small mb-1"><strong>Clinical Tip:</strong></p>
                    <p className="small text-primary">{s.clinical_tip}</p>
                  </div>
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Seizure Triggers (8 triggers) — Prevalence" borderColor={ACCENT4}>
            {breakdown.triggers.map((t, i) => (
              <PctBar key={i} label={t.trigger} pct={t.prevalence_pct} color={ACCENT4} />
            ))}
          </SectionCard>

          <SectionCard title="Trigger Detail" borderColor={ACCENT4}>
            {breakdown.trigger_detail.map((t, i) => (
              <div key={i} className="mb-3 pb-2 border-bottom">
                <h6 className="fw-bold" style={{ color: ACCENT4 }}>{t.trigger} — {t.prevalence_pct}%</h6>
                <div className="row">
                  <div className="col-md-6">
                    <p className="small mb-0"><strong>Mechanism:</strong> {t.mechanism}</p>
                  </div>
                  <div className="col-md-6">
                    <p className="small mb-0"><strong>Management:</strong> {t.management}</p>
                  </div>
                </div>
              </div>
            ))}
          </SectionCard>
        </>
      )}

      {/* ── Tab 3: Treatments ── */}
      {tab === 3 && breakdown && (
        <>
          <Alert variant="success" text="✅ ADLTE PROGNOSIS: 65–70% seizure-free on CBZ/OXC (first AED trial). One of the best prognoses in genetic epilepsy." />
          <Alert variant="info" text="💡 First-line: CBZ XR or OXC XR (Level B). LTG preferred if female of reproductive potential (lower teratogenicity). LEV as adjunct or alternative." />
          <Alert variant="warning" text="⚠️ HLA-B*15:02 mandatory before CBZ/OXC (Asian ancestry). POLG mandatory before VPA. VPPP mandatory for VPA in females." />

          <SectionCard title="Contraindications" borderColor={ACCENT2}>
            {breakdown.contraindication_detail.map((ci, i) => (
              <div key={i} className="mb-3 pb-2 border-bottom">
                <div className="d-flex justify-content-between align-items-center mb-1">
                  <h6 className="fw-bold mb-0" style={{ color: ACCENT2 }}>{ci.drug}</h6>
                  <span className="badge" style={{ backgroundColor: ACCENT2 }}>{ci.severity}</span>
                </div>
                <p className="small mb-1"><strong>Reason:</strong> {ci.reason}</p>
                <p className="small mb-0 text-success"><strong>Alternative:</strong> {ci.alternative}</p>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Treatment Detail (6 medications)" borderColor={ACCENT3}>
            {breakdown.treatment_detail.map((tx, i) => (
              <div key={i} className="mb-4 pb-3 border-bottom">
                <div className="d-flex justify-content-between align-items-start mb-2">
                  <h6 className="fw-bold mb-0" style={{ color: ACCENT3 }}>{tx.drug}</h6>
                  <span className="badge" style={{ backgroundColor: ACCENT3 }}>{tx.evidence}</span>
                </div>
                <div className="row">
                  <div className="col-md-3">
                    <p className="small mb-1"><strong>MOA:</strong></p>
                    <p className="small text-muted">{tx.moa}</p>
                  </div>
                  <div className="col-md-2">
                    <p className="small mb-1"><strong>Dose:</strong></p>
                    <p className="small text-muted">{tx.dose}</p>
                  </div>
                  <div className="col-md-2">
                    <p className="small mb-1"><strong>Efficacy:</strong></p>
                    <p className="small text-muted">{tx.efficacy}</p>
                  </div>
                  <div className="col-md-2">
                    <p className="small mb-1"><strong>Safety:</strong></p>
                    <p className="small text-danger">{tx.safety}</p>
                  </div>
                  <div className="col-md-3">
                    <p className="small mb-1"><strong>LGI1-Specific Note:</strong></p>
                    <p className="small text-primary">{tx.lgi1_note}</p>
                    <p className="small mb-0"><strong>Monitor:</strong> <em>{tx.monitoring}</em></p>
                  </div>
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Monitoring Checklist" borderColor={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-striped mb-0">
                <thead><tr><th>Monitoring Item</th><th>Frequency</th><th>Notes</th></tr></thead>
                <tbody>
                  {breakdown.monitoring.map((m, i) => (
                    <tr key={i}>
                      <td className="small fw-bold">{m.item}</td>
                      <td className="small">{m.frequency}</td>
                      <td className="small text-muted">{m.notes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="Patient Lifecycle (6 windows)" borderColor={ACCENT}>
            {breakdown.lifecycle.map((lc, i) => (
              <div key={i} className="mb-3 pb-2 border-bottom">
                <h6 className="fw-bold" style={{ color: ACCENT }}>{lc.window} — {lc.prevalence_pct}%</h6>
                <p className="small mb-0 text-muted">{lc.description}</p>
              </div>
            ))}
          </SectionCard>
        </>
      )}

      {/* ── Tab 4: Definitions ── */}
      {tab === 4 && definitions && (
        <>
          <SectionCard title="Key Concepts & Definitions (15)" borderColor={ACCENT}>
            {definitions.concepts.map((c, i) => (
              <div key={i} className="mb-3 pb-2 border-bottom">
                <h6 className="fw-bold" style={{ color: ACCENT }}>{c.term}</h6>
                <p className="small mb-0 text-muted">{c.definition}</p>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Standards & Guidelines" borderColor={ACCENT3}>
            <ul className="mb-0">
              {definitions.standards.map((s, i) => (
                <li key={i} className="small mb-1">{s}</li>
              ))}
            </ul>
          </SectionCard>

          <SectionCard title="Key References" borderColor={ACCENT}>
            <ol className="mb-0">
              {definitions.references.map((r, i) => (
                <li key={i} className="small mb-1">{r}</li>
              ))}
            </ol>
          </SectionCard>
        </>
      )}
    </div>
  );
}

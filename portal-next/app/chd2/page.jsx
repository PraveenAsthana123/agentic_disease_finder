'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#7a3a00';   // deep amber-brown — CHD2 / chromatin / photosensitive
const ACCENT2 = '#8a1c1c';   // deep crimson — contraindications / danger
const ACCENT3 = '#1a5a1a';   // forest green — seizure-free / monitoring OK
const ACCENT4 = '#c77d00';   // amber-gold — photosensitivity / PPR warning

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#fdf5e6', color: borderColor }}>
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

export default function CHD2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/chd2/overview`).then(r => r.json()),
      fetch(`${API}/api/chd2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/chd2/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading CHD2 data…</p></div>;
  if (error) return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="row mb-3">
        <div className="col">
          <h2 className="fw-bold mb-0" style={{ color: ACCENT }}>
            &#x1f9ec; CHD2 Epilepsy — GGE-Photosensitive / Myoclonic Encephalopathy
          </h2>
          <p className="text-muted mb-1" style={{ fontSize: 13 }}>
            <strong>CHD2</strong> · Chromodomain Helicase DNA Binding Protein 2 · 15q26.1 ·
            OMIM <a href="#" onClick={e => e.preventDefault()}>#615369</a> ·
            AD (>90% de novo) · H3.3 chromatin remodeling · GABAergic interneuron LOF ·
            PPR 75-80% (among highest in genetic epilepsies)
          </p>
          <Alert
            text="⚠ CBZ / OXC / PHT ABSOLUTE CONTRAINDICATION — GGE aggravation (absence status + myoclonic worsening). Tiagabine ABSOLUTE CI (NCSE). LTG: EEG-gated (myoclonic worsening 15-20%). VPPP + POLG mandatory before VPA."
            variant="danger"
          />
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && overview && (
        <>
          <div className="row mb-3">
            {(overview.key_kpis || []).map((k, i) => <KPI key={i} label={k.label} value={k.value} color={k.color} />)}
          </div>

          <div className="row">
            <div className="col-md-6">
              <SectionCard title="🧬 Gene Summary" borderColor={ACCENT}>
                <table className="table table-sm table-borderless mb-0" style={{ fontSize: 13 }}>
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>{overview.gene} ({overview.locus})</td></tr>
                    <tr><td className="fw-bold">OMIM</td><td>{overview.omim}</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>{overview.protein}</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>{overview.inheritance}</td></tr>
                    <tr><td className="fw-bold">Syndrome</td><td>{overview.syndrome}</td></tr>
                    <tr><td className="fw-bold">Cohort</td><td>{overview.cohort_size} patients</td></tr>
                    <tr><td className="fw-bold">pLI</td><td>0.99 (very high LOF intolerance)</td></tr>
                    <tr><td className="fw-bold">Incidence</td><td>~1:50,000–100,000</td></tr>
                  </tbody>
                </table>
              </SectionCard>

              <SectionCard title="🌟 Mechanism" borderColor={ACCENT}>
                <p style={{ fontSize: 13 }}>{overview.mechanism}</p>
              </SectionCard>
            </div>

            <div className="col-md-6">
              <SectionCard title="⚡ Photosensitivity & Outcomes" borderColor={ACCENT4}>
                <PctBar label="Photosensitive (PPR)" pct={overview.photosensitive_pct} color={ACCENT4} />
                <PctBar label="Seizure-free" pct={overview.seizure_free_pct} color={ACCENT3} />
                <PctBar label="Drug-resistant" pct={overview.drug_resistant_pct} color={ACCENT2} />
                <PctBar label="GTCS (any)" pct={55} color="#1a4080" />
                <PctBar label="Myoclonic (pathognomonic)" pct={85} color={ACCENT} />
                <PctBar label="Typical Absence" pct={70} color="#5a0080" />
                <PctBar label="ADHD comorbidity" pct={50} color="#805000" />
                <PctBar label="Mild-Moderate ID" pct={65} color="#3a5080" />
              </SectionCard>

              <SectionCard title="🔝 Top Triggers" borderColor="#c77d00">
                {(overview.top_triggers || []).map((t, i) => (
                  <div key={i} className="mb-1">
                    <PctBar label={t.trigger} pct={t.prevalence_pct} color={ACCENT4} />
                  </div>
                ))}
              </SectionCard>
            </div>
          </div>

          <SectionCard title="📋 Dashboard Counts" borderColor="#5a5a5a">
            <div className="row text-center">
              {[
                { label: 'Etiology Classes', val: overview.etiology_count },
                { label: 'Seizure Types', val: overview.seizure_type_count },
                { label: 'Triggers', val: overview.trigger_count },
                { label: 'Treatments', val: overview.treatment_count },
                { label: 'Contraindications', val: overview.contraindication_count },
                { label: 'Monitoring Items', val: overview.monitoring_count },
                { label: 'Concepts', val: overview.concept_count },
                { label: 'Standards', val: overview.standard_count },
                { label: 'References', val: overview.reference_count },
              ].map((x, i) => (
                <div key={i} className="col-4 col-md-3 mb-2">
                  <div className="fw-bold fs-5" style={{ color: ACCENT }}>{x.val}</div>
                  <div className="text-muted small">{x.label}</div>
                </div>
              ))}
            </div>
          </SectionCard>
        </>
      )}

      {/* ── TAB 1: Patients & Etiology ── */}
      {tab === 1 && breakdown && (
        <>
          <SectionCard title="🧬 Etiology Catalog (5 Classes)" borderColor={ACCENT}>
            {(breakdown.etiology_catalog || []).map((e, i) => (
              <div key={i} className="mb-3 pb-3 border-bottom">
                <div className="d-flex justify-content-between align-items-start mb-1">
                  <strong style={{ color: ACCENT }}>{e.etiology}</strong>
                  <span className="badge" style={{ backgroundColor: ACCENT }}>{e.pct}% (n={e.n})</span>
                </div>
                <div className="mb-1" style={{ fontSize: 12 }}>
                  <strong>Mechanism:</strong> {e.mechanism}
                </div>
                <div className="mb-1" style={{ fontSize: 12 }}>
                  <strong>EEG:</strong> {e.eeg_correlate}
                </div>
                <div className="mb-1" style={{ fontSize: 12 }}>
                  <strong>Semiology:</strong> {e.semiology}
                </div>
                <div className="mb-1" style={{ fontSize: 12 }}>
                  <strong>Treatment:</strong> {e.treatment}
                </div>
                <div style={{ fontSize: 12, color: e.prognosis.includes('Poor') || e.prognosis.includes('poor') ? ACCENT2 : ACCENT3 }}>
                  <strong>Prognosis:</strong> {e.prognosis}
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="👥 Patient Sample (first 15)" borderColor="#3a5080">
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>ID</th><th>Name</th><th>Age</th><th>Etiology</th>
                    <th>Onset (mo)</th><th>AED</th><th>Outcome</th><th>Photo</th><th>ID</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.patient_sample || []).map((p, i) => (
                    <tr key={i}>
                      <td><code>{p.id}</code></td>
                      <td>{p.name}</td>
                      <td>{p.age}y</td>
                      <td style={{ maxWidth: 160, fontSize: 11 }}>{p.etiology}</td>
                      <td>{p.onset_age_months}mo</td>
                      <td><Badge text={p.current_aed} color={ACCENT} /></td>
                      <td>
                        <Badge
                          text={p.outcome}
                          color={
                            p.outcome === 'Seizure-free' ? ACCENT3 :
                            p.outcome === 'Drug-resistant' ? ACCENT2 :
                            p.outcome === '≥50% reduction' ? '#1a4080' : '#805000'
                          }
                        />
                      </td>
                      <td>{p.photosensitive ? '✅' : '—'}</td>
                      <td><Badge text={p.id_severity} color="#5a5a5a" /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      )}

      {/* ── TAB 2: Seizures & Triggers ── */}
      {tab === 2 && breakdown && (
        <>
          <SectionCard title="⚡ Seizure Types (5)" borderColor={ACCENT}>
            {(breakdown.seizure_types || []).map((s, i) => (
              <div key={i} className="mb-4 pb-3 border-bottom">
                <div className="d-flex justify-content-between align-items-center mb-2">
                  <h6 className="fw-bold mb-0" style={{ color: ACCENT }}>{s.type}</h6>
                  <span className="badge" style={{ backgroundColor: ACCENT }}>{s.prevalence_pct}%</span>
                </div>
                <PctBar label="Prevalence" pct={s.prevalence_pct} color={ACCENT} />
                <div style={{ fontSize: 12 }} className="mb-1"><strong>EEG:</strong> {s.eeg}</div>
                <div style={{ fontSize: 12 }} className="mb-1"><strong>Semiology:</strong> {s.semiology}</div>
                <Alert text={`💡 ${s.clinical_tip}`} variant="info" />
              </div>
            ))}
          </SectionCard>

          <SectionCard title="🔥 Triggers (8)" borderColor={ACCENT4}>
            {(breakdown.trigger_detail || []).map((t, i) => (
              <div key={i} className="mb-3 pb-3 border-bottom">
                <div className="d-flex justify-content-between align-items-center mb-1">
                  <strong style={{ color: ACCENT4 }}>{t.trigger}</strong>
                  <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.prevalence_pct}%</span>
                </div>
                <PctBar label="Prevalence" pct={t.prevalence_pct} color={ACCENT4} />
                <div style={{ fontSize: 12 }} className="mb-1"><strong>Mechanism:</strong> {t.mechanism}</div>
                <div style={{ fontSize: 12 }}><strong>Management:</strong> {t.management}</div>
              </div>
            ))}
          </SectionCard>
        </>
      )}

      {/* ── TAB 3: Treatments ── */}
      {tab === 3 && breakdown && (
        <>
          <Alert
            text="⚠ MANDATORY BEFORE VPA: (1) POLG screen — fatal hepatic failure risk in POLG mutation carriers. (2) VPPP — annual contraception review for all females ≥12 years (MHRA 2021). (3) LTG: EEG-gated — myoclonic EEG assessment mandatory before initiating (15-20% worsening risk)."
            variant="danger"
          />

          {(breakdown.treatment_detail || []).map((tx, i) => (
            <SectionCard
              key={i}
              title={`${tx.drug} — ${tx.evidence}`}
              borderColor={tx.evidence.includes('A') ? ACCENT3 : tx.evidence.includes('CAUTION') ? ACCENT2 : ACCENT}
            >
              <div className="row">
                <div className="col-md-6">
                  <p style={{ fontSize: 12 }}><strong>Indication:</strong> {tx.indication}</p>
                  <p style={{ fontSize: 12 }}><strong>Dose:</strong> {tx.dose}</p>
                  <p style={{ fontSize: 12 }}><strong>MOA:</strong> {tx.moa}</p>
                  <p style={{ fontSize: 12 }}><strong>Efficacy:</strong> {tx.efficacy}</p>
                </div>
                <div className="col-md-6">
                  <p style={{ fontSize: 12 }}><strong>Safety:</strong> {tx.safety}</p>
                  <p style={{ fontSize: 12 }}><strong>Monitoring:</strong> {tx.monitoring}</p>
                  <Alert text={`🧬 CHD2 note: ${tx.chd2_note}`} variant="info" />
                </div>
              </div>
            </SectionCard>
          ))}

          <SectionCard title="🚫 Contraindications" borderColor={ACCENT2}>
            {(breakdown.contraindication_detail || []).map((c, i) => (
              <div key={i} className="mb-3 pb-2 border-bottom">
                <div className="d-flex justify-content-between align-items-start mb-1">
                  <strong style={{ color: ACCENT2 }}>{c.drug}</strong>
                  <Badge text={c.risk_level} color={c.risk_level.includes('ABSOLUTE') ? ACCENT2 : '#805000'} />
                </div>
                <p style={{ fontSize: 12 }} className="mb-1"><strong>Reason:</strong> {c.reason}</p>
                <p style={{ fontSize: 12 }} className="mb-0"><strong>Alternative:</strong> {c.alternative}</p>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="🩺 Monitoring (14 items)" borderColor={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
                <thead><tr><th>Item</th><th>Frequency</th><th>Rationale</th></tr></thead>
                <tbody>
                  {(breakdown.monitoring || []).map((m, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{m.item}</td>
                      <td>{m.frequency}</td>
                      <td>{m.rationale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      )}

      {/* ── TAB 4: Definitions ── */}
      {tab === 4 && definitions && (
        <>
          <SectionCard title="📖 Key Concepts (15)" borderColor={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
                <thead><tr><th style={{ minWidth: 200 }}>Term</th><th>Definition</th></tr></thead>
                <tbody>
                  {(definitions.concepts || []).map((c, i) => (
                    <tr key={i}>
                      <td className="fw-bold" style={{ color: ACCENT }}>{c.term}</td>
                      <td>{c.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="📏 Thresholds" borderColor="#5a1080">
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
                <thead><tr><th>Parameter</th><th>Value</th><th>Unit</th></tr></thead>
                <tbody>
                  {(definitions.thresholds || []).map((t, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{t.parameter}</td>
                      <td style={{ color: ACCENT }}>{t.value}</td>
                      <td className="text-muted">{t.unit}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="📚 Standards (12)" borderColor="#1a4080">
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
                <thead><tr><th>Standard</th><th>Relevance</th></tr></thead>
                <tbody>
                  {(definitions.standards || []).map((s, i) => (
                    <tr key={i}>
                      <td className="fw-bold" style={{ color: '#1a4080' }}>{s.standard}</td>
                      <td>{s.relevance}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="🔬 References (6)" borderColor="#2a6040">
            {(definitions.references || []).map((r, i) => (
              <div key={i} className="mb-2 pb-2 border-bottom" style={{ fontSize: 12 }}>
                <strong style={{ color: '#2a6040' }}>[{r.ref}]</strong> {r.citation}
              </div>
            ))}
          </SectionCard>

          <SectionCard title="🕐 Lifecycle (6 windows)" borderColor={ACCENT}>
            {(breakdown?.lifecycle || []).map((lc, i) => (
              <div key={i} className="mb-3 pb-2 border-bottom">
                <h6 className="fw-bold mb-1" style={{ color: ACCENT }}>
                  {lc.window} — {lc.label}
                </h6>
                <ul className="mb-0" style={{ fontSize: 12 }}>
                  {(lc.key_actions || []).map((a, j) => <li key={j}>{a}</li>)}
                </ul>
              </div>
            ))}
          </SectionCard>
        </>
      )}
    </div>
  );
}

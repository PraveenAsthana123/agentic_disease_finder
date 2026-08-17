'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4a1060';   // deep purple — calmodulin / cardiac-neuro
const ACCENT2 = '#8a0000';   // deep crimson — ICD / SCD / absolute CI
const ACCENT3 = '#005040';   // teal-green — cardiac-safe AEDs / monitoring OK
const ACCENT4 = '#c05000';   // amber-red — LQTS / QTc warning

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#f5e8ff', color: borderColor }}>
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

export default function CALMPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/calm/overview`).then(r => r.json()),
      fetch(`${API}/api/calm/breakdown`).then(r => r.json()),
      fetch(`${API}/api/calm/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading CALM Calmodulinopathy data…</p></div>;
  if (error) return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="row mb-3">
        <div className="col">
          <h2 className="fw-bold mb-0" style={{ color: ACCENT }}>
            &#x1f9ec; CALM1/2/3 Calmodulinopathy — DEE + Long-QT Syndrome / CPVT
          </h2>
          <p className="text-muted mb-1" style={{ fontSize: 13 }}>
            <strong>CALM1</strong> (14q32.11) · <strong>CALM2</strong> (2p21) · <strong>CALM3</strong> (19q13.32) ·
            Three genes — identical 148 aa Calmodulin (CaM) protein ·
            OMIM LQTS14 <strong>#616036</strong> / LQTS15 <strong>#616037</strong> / LQTS16 <strong>#616038</strong> ·
            AD (>99% de novo) · GOF EF-hand missense · CDI failure → QTc → TdP + CPVT5 + DEE
          </p>
          <Alert
            text="⚠️ CALMODULINOPATHY: Only epilepsy syndrome requiring mandatory CARDIAC co-management. QTc-prolonging co-medications ABSOLUTELY CONTRAINDICATED. ICD for QTc >500ms + VT. Seizure → catecholamine → CPVT/TdP cascade = dual SCD+SUDEP risk."
            variant="danger"
          />
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && overview && (
        <>
          <div className="row mb-2">
            {(overview.key_kpis || []).map((k, i) => (
              <KPI key={i} label={k.label} value={k.value} color={k.color} />
            ))}
          </div>

          <div className="row">
            <div className="col-md-6">
              <SectionCard title="Gene & Protein" borderColor={ACCENT}>
                <table className="table table-sm table-borderless mb-0" style={{ fontSize: 13 }}>
                  <tbody>
                    <tr><th>Genes</th><td>{overview.gene_group}</td></tr>
                    <tr><th>Loci</th><td>{overview.loci}</td></tr>
                    <tr><th>OMIM</th><td>{overview.omim}</td></tr>
                    <tr><th>Protein</th><td>{overview.protein}</td></tr>
                    <tr><th>Inheritance</th><td>{overview.inheritance}</td></tr>
                    <tr><th>Syndrome</th><td>{overview.syndrome}</td></tr>
                    <tr><th>Cohort</th><td>{overview.cohort_size} patients</td></tr>
                  </tbody>
                </table>
              </SectionCard>

              <SectionCard title="Calmodulinopathy Mechanism" borderColor={ACCENT4}>
                <p style={{ fontSize: 13 }}>{overview.mechanism}</p>
                <Alert
                  text="🫀 THREE GENES → ONE PROTEIN: CALM1, CALM2, CALM3 all encode identical calmodulin. A single de novo GOF variant in any one gene is sufficient to cause calmodulinopathy — because mutant CaM competes with WT-CaM for the same binding sites (dominant mechanism)."
                  variant="info"
                />
              </SectionCard>
            </div>

            <div className="col-md-6">
              <SectionCard title="Cohort KPIs" borderColor={ACCENT2}>
                <PctBar label="ICD Implanted" pct={overview.icd_implanted_pct} color={ACCENT2} />
                <PctBar label="Drug-Resistant DEE" pct={overview.drug_resistant_pct} color={ACCENT4} />
                <PctBar label="CALM2 (Most Severe LQTS15)" pct={overview.calm2_severe_pct} color="#4a0080" />
                <PctBar label="CPVT5 Overlap" pct={72} color={ACCENT3} />
                <PctBar label="Febrile Trigger" pct={88} color="#c05000" />
              </SectionCard>

              <SectionCard title="Top Triggers" borderColor={ACCENT4}>
                {(overview.top_triggers || []).map((t, i) => (
                  <PctBar key={i} label={t.trigger} pct={t.prevalence_pct} color={ACCENT4} />
                ))}
              </SectionCard>

              <SectionCard title="Key Counts" borderColor={ACCENT3}>
                <div className="row g-2" style={{ fontSize: 13 }}>
                  {[
                    ['Etiology Classes', overview.etiology_count],
                    ['Seizure Types', overview.seizure_type_count],
                    ['Triggers', overview.trigger_count],
                    ['Treatments', overview.treatment_count],
                    ['Contraindications', overview.contraindication_count],
                    ['Monitoring Items', overview.monitoring_count],
                    ['Concepts', overview.concept_count],
                    ['Standards', overview.standard_count],
                    ['References', overview.reference_count],
                  ].map(([label, val], i) => (
                    <div className="col-6" key={i}>
                      <span className="fw-bold" style={{ color: ACCENT }}>{val}</span> {label}
                    </div>
                  ))}
                </div>
              </SectionCard>
            </div>
          </div>
        </>
      )}

      {/* ── TAB 1: Patients & Etiology ── */}
      {tab === 1 && breakdown && (
        <>
          <SectionCard title="5-Class Etiology Catalog" borderColor={ACCENT}>
            {(breakdown.etiology_catalog || []).map((et, i) => (
              <div key={i} className="mb-4 pb-3" style={{ borderBottom: '1px solid #eee' }}>
                <div className="d-flex justify-content-between align-items-start mb-1">
                  <strong style={{ color: ACCENT, fontSize: 14 }}>{et.etiology}</strong>
                  <Badge text={`${et.pct}% · n=${et.n}`} color={ACCENT} />
                </div>
                <PctBar label="" pct={et.pct} color={ACCENT} />
                <div className="mb-1" style={{ fontSize: 13 }}>
                  <strong>Mechanism:</strong> {et.mechanism}
                </div>
                <div className="mb-1" style={{ fontSize: 12, color: '#555' }}>
                  <strong>EEG:</strong> {et.eeg_correlate}
                </div>
                <div className="mb-1" style={{ fontSize: 12 }}>
                  <strong>Semiology:</strong> {et.semiology}
                </div>
                <div className="mb-1" style={{ fontSize: 12, color: ACCENT3 }}>
                  <strong>Treatment:</strong> {et.treatment}
                </div>
                <div style={{ fontSize: 12, color: '#888' }}>
                  <strong>Prognosis:</strong> {et.prognosis}
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Patient Sample (n=15 of 40)" borderColor={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-light">
                  <tr>
                    <th>ID</th><th>Gene</th><th>Variant</th><th>Onset (m)</th>
                    <th>Phenotype</th><th>QTc (ms)</th><th>ICD</th><th>Outcome</th>
                    <th>AEDs</th><th>Cardiac Rx</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.patient_sample || []).map((p, i) => (
                    <tr key={i}>
                      <td>{p.patient_id}</td>
                      <td><Badge text={p.gene} color={ACCENT} /></td>
                      <td><code>{p.variant}</code></td>
                      <td>{p.onset_months}m</td>
                      <td style={{ maxWidth: 120 }}>{p.phenotype}</td>
                      <td>
                        <span style={{ color: p.qtc_ms > 500 ? ACCENT2 : '#333', fontWeight: p.qtc_ms > 500 ? 'bold' : 'normal' }}>
                          {p.qtc_ms}
                        </span>
                      </td>
                      <td>{p.icd_implanted ? <Badge text="Yes" color={ACCENT2} /> : <span style={{ color: '#888' }}>No</span>}</td>
                      <td>
                        <Badge
                          text={p.outcome}
                          color={p.outcome === 'Drug-resistant' ? ACCENT2 : p.outcome === 'Seizure-free' ? ACCENT3 : ACCENT4}
                        />
                      </td>
                      <td style={{ maxWidth: 110 }}>{p.current_aeds}</td>
                      <td style={{ maxWidth: 140, fontSize: 11 }}>{p.cardiac_rx}</td>
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
          <SectionCard title="5 Seizure Types" borderColor={ACCENT}>
            {(breakdown.seizure_types || []).map((st, i) => (
              <div key={i} className="mb-4 pb-3" style={{ borderBottom: '1px solid #eee' }}>
                <div className="d-flex justify-content-between mb-1">
                  <strong style={{ color: ACCENT, fontSize: 14 }}>{st.type}</strong>
                  <Badge text={`${st.prevalence_pct}%`} color={ACCENT} />
                </div>
                <PctBar label="" pct={st.prevalence_pct} color={ACCENT} />
                <div style={{ fontSize: 13 }} className="mb-1">
                  <strong>EEG:</strong> {st.eeg_pattern}
                </div>
                <div style={{ fontSize: 13 }} className="mb-1">
                  <strong>Semiology:</strong> {st.semiology}
                </div>
                <Alert text={`⚠️ Cardiac Tip: ${st.clinical_tip}`} variant="warning" />
              </div>
            ))}
          </SectionCard>

          <SectionCard title="8 Triggers (with Cardiac Risk)" borderColor={ACCENT4}>
            {(breakdown.trigger_detail || []).map((t, i) => (
              <div key={i} className="mb-3 pb-2" style={{ borderBottom: '1px solid #eee' }}>
                <div className="d-flex justify-content-between mb-1">
                  <strong style={{ fontSize: 14 }}>{t.trigger}</strong>
                  <Badge text={`${t.prevalence_pct}%`} color={ACCENT4} />
                </div>
                <PctBar label="" pct={t.prevalence_pct} color={ACCENT4} />
                <div style={{ fontSize: 12 }}>
                  <Badge text={t.cardiac_risk} color={t.cardiac_risk?.startsWith('VERY HIGH') ? ACCENT2 : t.cardiac_risk?.startsWith('HIGH') ? '#c05000' : '#6a6a00'} />
                </div>
                <div style={{ fontSize: 12, color: '#555' }} className="mt-1">
                  <strong>Mechanism:</strong> {t.mechanism}
                </div>
                <div style={{ fontSize: 12, color: ACCENT3 }}>
                  <strong>Management:</strong> {t.management}
                </div>
              </div>
            ))}
          </SectionCard>
        </>
      )}

      {/* ── TAB 3: Treatments ── */}
      {tab === 3 && breakdown && (
        <>
          <Alert
            text="🫀 CALMODULINOPATHY AED SAFETY RULE: Use VPA, LEV, CLB — NO QTc effect. Avoid phenytoin (QTc concern, IV bradycardia). MANDATORY CredibleMeds audit for ALL co-medications. Cardiac drugs (nadolol, flecainide, ICD) are co-managed by electrophysiology."
            variant="info"
          />
          <SectionCard title="8 Treatments (Neurological + Cardiac)" borderColor={ACCENT3}>
            {(breakdown.treatment_detail || []).map((tx, i) => (
              <div key={i} className="mb-4 pb-3" style={{ borderBottom: '1px solid #eee' }}>
                <div className="d-flex justify-content-between mb-1">
                  <strong style={{ color: ACCENT3, fontSize: 14 }}>{tx.drug}</strong>
                  <Badge text={tx.evidence} color={ACCENT3} />
                </div>
                <div style={{ fontSize: 12 }} className="mb-1">
                  <strong>Dose:</strong> {tx.dose}
                </div>
                <div style={{ fontSize: 12 }} className="mb-1">
                  <strong>MOA:</strong> {tx.moa}
                </div>
                <div style={{ fontSize: 12 }} className="mb-1">
                  <strong>Efficacy:</strong> {tx.efficacy}
                </div>
                <div style={{ fontSize: 12 }} className="mb-1">
                  <strong>Safety:</strong> {tx.safety}
                </div>
                <div style={{ fontSize: 12 }} className="mb-1">
                  <strong>Monitoring:</strong> {tx.monitoring}
                </div>
                <Alert text={`🧬 CALM note: ${tx.calm_note}`} variant="info" />
              </div>
            ))}
          </SectionCard>

          <SectionCard title="5 Contraindications" borderColor={ACCENT2}>
            {(breakdown.contraindication_detail || []).map((ci, i) => (
              <div key={i} className="mb-3 pb-2" style={{ borderBottom: '1px solid #eee' }}>
                <div className="d-flex justify-content-between mb-1">
                  <strong style={{ color: ACCENT2, fontSize: 14 }}>{ci.drug}</strong>
                  <Badge text={ci.risk_level} color={ci.risk_level?.startsWith('ABSOLUTE') ? ACCENT2 : ACCENT4} />
                </div>
                <div style={{ fontSize: 12 }} className="mb-1">{ci.reason}</div>
                <div style={{ fontSize: 12, color: ACCENT3 }}>
                  <strong>Alternative:</strong> {ci.alternative}
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="14 Monitoring Items" borderColor={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-light">
                  <tr><th>Item</th><th>Frequency</th><th>Rationale</th></tr>
                </thead>
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
          <SectionCard title="15 Key Concepts" borderColor={ACCENT}>
            {(definitions.concepts || []).map((c, i) => (
              <div key={i} className="mb-2 pb-2" style={{ borderBottom: '1px solid #f0e8ff' }}>
                <strong style={{ color: ACCENT }}>{c.term}</strong>
                <div style={{ fontSize: 13, color: '#444' }}>{c.definition}</div>
              </div>
            ))}
          </SectionCard>

          <div className="row">
            <div className="col-md-6">
              <SectionCard title="12 Thresholds" borderColor={ACCENT4}>
                <table className="table table-sm" style={{ fontSize: 12 }}>
                  <thead className="table-light"><tr><th>Parameter</th><th>Value</th><th>Unit</th></tr></thead>
                  <tbody>
                    {(definitions.thresholds || []).map((t, i) => (
                      <tr key={i}>
                        <td>{t.parameter}</td>
                        <td className="fw-bold" style={{ color: ACCENT4 }}>{t.value}</td>
                        <td className="text-muted">{t.unit}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="12 Standards & Guidelines" borderColor={ACCENT3}>
                {(definitions.standards || []).map((s, i) => (
                  <div key={i} className="mb-2" style={{ fontSize: 12 }}>
                    <strong style={{ color: ACCENT3 }}>{s.standard}</strong>
                    <div className="text-muted">{s.relevance}</div>
                  </div>
                ))}
              </SectionCard>
            </div>
          </div>

          <SectionCard title="6 References" borderColor={ACCENT}>
            {(definitions.references || []).map((r, i) => (
              <div key={i} className="mb-2" style={{ fontSize: 12 }}>
                <strong style={{ color: ACCENT }}>{r.ref}</strong> — {r.citation}
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Lifecycle (6 Windows)" borderColor={ACCENT3}>
            {(breakdown?.lifecycle || []).map((lc, i) => (
              <div key={i} className="mb-3 pb-2" style={{ borderBottom: '1px solid #eee' }}>
                <strong style={{ color: ACCENT }}>{lc.window}</strong>
                <span className="ms-2 text-muted" style={{ fontSize: 12 }}>— {lc.label}</span>
                <ul className="mb-0 mt-1" style={{ fontSize: 12 }}>
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

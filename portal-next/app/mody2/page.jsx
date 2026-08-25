'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variant & Cohort', 'Treatment & Detection', 'Definitions'];

// MODY2 colour scheme — green/teal (stable, benign, no treatment)
const ACCENT  = '#1b5e20';   // deep green — stable mild hyperglycaemia; diet only; no progression
const ACCENT2 = '#006064';   // dark teal — limited OGTT increment; glucostat reset
const ACCENT3 = '#b71c1c';   // deep red — misdiagnosis; pregnancy exception
const ACCENT4 = '#e65100';   // deep orange — pregnancy macrosomia; fetal GCK decision
const ACCENT5 = '#4a148c';   // purple — GCK gene; molecular genetics
const ACCENT6 = '#1565c0';   // dark blue — HbA1c; glucose metrics
const ACCENT7 = '#37474f';   // dark slate — epidemiology; detection
const ACCENT8 = '#004d40';   // dark teal — variants; GCK enzyme

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

function Alert({ color, children }) {
  return (
    <div className="alert mb-2" style={{ background: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 6 }}>
      {children}
    </div>
  );
}

function Section({ title, color, children }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

function Badge({ text, color }) {
  return <span className="badge me-1" style={{ background: color, fontSize: '0.72em' }}>{text}</span>;
}

export default function MODY2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/mody2/overview`).then(r => r.json()),
      fetch(`${API}/api/mody2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mody2/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;

  const kpis = overview?.kpis || {};
  const patients = overview?.patients || [];
  const keyFacts = overview?.key_facts || [];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT2}11)`, border: `1px solid ${ACCENT}44` }}>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span style={{ fontSize: '1.6rem' }}>🧬</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>MODY2 — GCK-MODY</h4>
            <div className="text-muted small">Maturity-Onset Diabetes of the Young Type 2 · Chr 7p13 · OMIM #125851 · ~25–35% of all MODY · Glucose sensor defect · Stable mild hyperglycaemia · NO treatment (usually) · Autosomal Dominant</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="GCK *138079" color={ACCENT5} />
            <Badge text="AD 50% risk" color={ACCENT7} />
            <Badge text="Stable HbA1c" color={ACCENT} />
            <Badge text="No Rx needed" color={ACCENT2} />
            <Badge text="Antibody negative" color={ACCENT7} />
            <Badge text="Pregnancy exception" color={ACCENT4} />
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Cohort" value={kpis.cohort_size || 40} color={ACCENT} />
        <KPI label="Mean Age (yr)" value={kpis.mean_age_years?.toFixed(1)} color={ACCENT6} />
        <KPI label="Dx Age (yr)" value={kpis.mean_age_at_diagnosis_years?.toFixed(1)} color={ACCENT7} />
        <KPI label="Mean HbA1c (%)" value={kpis.mean_hba1c_percent?.toFixed(1)} color={ACCENT6} />
        <KPI label="Mean FG (mmol/L)" value={kpis.mean_fasting_glucose_mmol?.toFixed(1)} color={ACCENT2} />
        <KPI label="OGTT Inc (mmol/L)" value={kpis.mean_ogtt_increment_mmol?.toFixed(1)} color={ACCENT2} />
        <KPI label="Diet/Monitoring (%)" value={`${kpis.pct_diet_only?.toFixed(0)}%`} color={ACCENT} />
        <KPI label="HbA1c <7% (%)" value={`${kpis.pct_hba1c_lt_7?.toFixed(0)}%`} color={ACCENT} />
        <KPI label="Misdiagnosed (%)" value={`${kpis.pct_prior_misdiagnosis?.toFixed(0)}%`} color={ACCENT3} />
        <KPI label="Family Hx (%)" value={`${kpis.pct_family_hx_positive?.toFixed(0)}%`} color={ACCENT5} />
        <KPI label="Antibody Neg" value="100%" color={ACCENT7} />
        <KPI label="Mean C-pep (nmol/L)" value={kpis.mean_c_peptide_nmol_L?.toFixed(2)} color={ACCENT} />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}>{t}</button>
          </li>
        ))}
      </ul>

      {/* Tab 0: Overview */}
      {tab === 0 && (
        <div className="row g-3">
          <div className="col-lg-6">
            <Section title="🧬 Disease Overview" color={ACCENT}>
              <Alert color={ACCENT}>
                <strong>MODY2-UNIQUE:</strong> GCK LOF resets the pancreatic glucose set-point upward by ~1–2 mmol/L — a SENSING defect, not secretory failure. HbA1c is mildly elevated but STABLE over decades. No progression. No complications at typical MODY2 HbA1c levels.
              </Alert>
              <Alert color={ACCENT2}>
                <strong>NO TREATMENT (usual):</strong> Diet and monitoring only. Sulfonylure causes hypoglycaemia without benefit (lowers below the new set-point). Insulin equally unhelpful except in specific pregnancy situations.
              </Alert>
              <Alert color={ACCENT3}>
                <strong>PREGNANCY EXCEPTION:</strong> Treatment depends on fetal GCK genotype. GCK-negative fetus → insulin for mother to prevent macrosomia. GCK-positive fetus → no treatment (maternal insulin causes maternal hypo + fetal growth restriction).
              </Alert>
              <table className="table table-sm table-bordered">
                <tbody>
                  <tr><td className="fw-bold">Gene</td><td>GCK (Glucokinase / Hexokinase IV) · Chr 7p13 · *138079</td></tr>
                  <tr><td className="fw-bold">Disease OMIM</td><td>#125851 (MODY2)</td></tr>
                  <tr><td className="fw-bold">Inheritance</td><td>Autosomal Dominant · 50% recurrence per child</td></tr>
                  <tr><td className="fw-bold">Mechanism</td><td>GCK LOF → raised glucose-sensing threshold → stable hyperglycaemia at new set-point; NOT progressive secretory failure</td></tr>
                  <tr><td className="fw-bold">Prevalence</td><td>~1:1,000–1:4,000 (25–35% of all MODY; most underdiagnosed — often labelled prediabetes)</td></tr>
                  <tr><td className="fw-bold">Onset age</td><td>Any age — children (school screening), adults (incidental), pregnancy (GDM screen)</td></tr>
                  <tr><td className="fw-bold">HbA1c</td><td>Stable 5.6–7.6% — does NOT worsen over decades (vs T2D/MODY3 progression)</td></tr>
                  <tr><td className="fw-bold">C-peptide</td><td>NORMAL — sensing defect, not secretory failure or autoimmune destruction</td></tr>
                  <tr><td className="fw-bold">Autoantibodies</td><td>NEGATIVE (GADA, ZnT8, IA-2) — key T1D differentiator</td></tr>
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-6">
            <Section title="📋 Key Clinical Facts" color={ACCENT5}>
              <ul className="list-group list-group-flush">
                {keyFacts.map((f, i) => (
                  <li key={i} className="list-group-item py-1 small">{f}</li>
                ))}
              </ul>
            </Section>
            <Section title="🩺 Diagnostic Criteria" color={ACCENT2}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(overview?.diagnostic_criteria || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold small">{k}</td><td className="small">{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
        </div>
      )}

      {/* Tab 1: Variant & Cohort */}
      {tab === 1 && (
        <div className="row g-3">
          <div className="col-lg-5">
            <Section title="🔬 Variant Distribution" color={ACCENT8}>
              <table className="table table-sm table-bordered table-hover">
                <thead><tr><th>Variant</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.variant_distribution || {})
                    .sort((a, b) => b[1] - a[1])
                    .map(([v, n]) => (
                      <tr key={v}>
                        <td><span className="badge" style={{ background: ACCENT8, fontSize: '0.72em' }}>{v}</span></td>
                        <td>{n}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 HbA1c Tiers" color={ACCENT6}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Tier</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.hba1c_tiers || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🩸 Fasting Glucose Tiers" color={ACCENT2}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Range</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.fasting_glucose_tiers || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="👥 Age Groups" color={ACCENT7}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Age group</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.age_groups || {}).map(([g, n]) => (
                    <tr key={g}><td>{g}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-7">
            <Section title="👩‍⚕️ Patient Cohort (40 patients, seed 307)" color={ACCENT}>
              <div style={{ maxHeight: 540, overflowY: 'auto' }}>
                <table className="table table-sm table-bordered table-hover" style={{ fontSize: '0.73em' }}>
                  <thead className="table-dark">
                    <tr>
                      <th>ID</th><th>Age</th><th>Sex</th><th>Dx Age</th><th>HbA1c%</th>
                      <th>FG (mmol)</th><th>OGTT Inc</th><th>C-pep</th><th>Variant</th><th>Treatment</th><th>Detected</th>
                    </tr>
                  </thead>
                  <tbody>
                    {patients.map(p => (
                      <tr key={p.patient_id}>
                        <td>{p.patient_id}</td>
                        <td>{p.age}</td>
                        <td>{p.sex}</td>
                        <td>{p.age_at_diagnosis}</td>
                        <td>
                          <span style={{ color: p.hba1c_percent > 7.6 ? ACCENT3 : p.hba1c_percent > 7.0 ? ACCENT4 : ACCENT, fontWeight: 600 }}>
                            {p.hba1c_percent?.toFixed(1)}
                          </span>
                        </td>
                        <td>{p.fasting_glucose_mmol?.toFixed(1)}</td>
                        <td>
                          <span style={{ color: p.ogtt_increment_mmol < 3.5 ? ACCENT2 : ACCENT3, fontWeight: 600 }}>
                            +{p.ogtt_increment_mmol?.toFixed(1)}
                          </span>
                        </td>
                        <td>{p.c_peptide_nmol_L?.toFixed(2)}</td>
                        <td style={{ fontSize: '0.68em' }}>{p.variant}</td>
                        <td style={{ fontSize: '0.68em' }}>{p.current_treatment}</td>
                        <td style={{ fontSize: '0.65em' }}>{p.detection_mode?.replace(/_/g, ' ')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* Tab 2: Treatment & Detection */}
      {tab === 2 && (
        <div className="row g-3">
          <div className="col-lg-6">
            <Section title="💊 Treatment Distribution" color={ACCENT}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Treatment</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.treatment_distribution || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
              <Alert color={ACCENT}>
                <strong>Key:</strong> Diet and monitoring only in the majority. Sulfonylurea is inappropriate — it lowers glucose below the new set-point causing hypoglycaemia without benefit. Insulin is reserved for pregnancy (when fetal GCK is negative).
              </Alert>
            </Section>
            <Section title="⚠️ Sulfonylure Consequence" color={ACCENT3}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Outcome</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.sulfo_consequence_distribution || {}).map(([r, n]) => (
                    <tr key={r}>
                      <td>
                        <span className="badge me-1" style={{
                          background: r === 'Hypoglycaemia' ? ACCENT3 : r === 'No_benefit' ? ACCENT4 : r === 'Not_started' ? ACCENT : ACCENT7,
                          fontSize: '0.7em'
                        }}>{r.replace(/_/g, ' ')}</span>
                      </td>
                      <td>{n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🔁 Prior Misdiagnosis" color={ACCENT3}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Prior Diagnosis</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.misdiagnosis_distribution || {}).map(([d, n]) => (
                    <tr key={d}>
                      <td>
                        <span className="badge me-1" style={{
                          background: d === 'Prediabetes' ? ACCENT7 : d === 'GDM' ? ACCENT4 : d === 'T2D' ? ACCENT3 : ACCENT,
                          fontSize: '0.7em'
                        }}>{d}</span>
                      </td>
                      <td>{n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <Alert color={ACCENT3}>
                <strong>Action:</strong> Confirmed MODY2 → STOP unnecessary sulfonylure/insulin → diet and monitoring only. Remove 'GDM' label in affected pregnancies if fetus is GCK-positive. Cascade test all first-degree relatives.
              </Alert>
            </Section>
          </div>
          <div className="col-lg-6">
            <Section title="🔍 Detection Mode" color={ACCENT7}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>How Detected</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.detection_mode_distribution || {}).map(([d, n]) => (
                    <tr key={d}><td>{d.replace(/_/g, ' ')}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="⚠️ Complications" color={ACCENT7}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Complication</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.complication_distribution || {}).map(([c, n]) => (
                    <tr key={c}><td>{c.replace(/_/g, ' ')}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
              <Alert color={ACCENT}>
                <strong>Low complication rate:</strong> Typical MODY2 HbA1c 5.6–7.6% does not reach the threshold for microvascular complications. This is strong supporting evidence against treatment and for reassurance.
              </Alert>
            </Section>
            <Section title="📅 Duration Tiers" color={ACCENT6}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Duration</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.duration_tiers || {}).map(([d, n]) => (
                    <tr key={d}><td>{d}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🧬 MODY2 vs MODY3 vs Prediabetes" color={ACCENT5}>
              <table className="table table-sm table-bordered" style={{ fontSize: '0.8em' }}>
                <thead><tr><th>Feature</th><th>MODY2 (GCK)</th><th>MODY3 (HNF1A)</th><th>Prediabetes</th></tr></thead>
                <tbody>
                  <tr><td>HbA1c</td><td style={{ color: ACCENT }}>5.6–7.6% stable</td><td>Progressive ↑</td><td>Progressive ↑</td></tr>
                  <tr><td>OGTT 2h inc</td><td style={{ color: ACCENT2 }}>&lt;3.5 mmol/L</td><td>Large excursion</td><td>Large excursion</td></tr>
                  <tr><td>Treatment</td><td style={{ color: ACCENT }}>None (usual)</td><td style={{ color: '#1565c0' }}>Sulfonylurea</td><td>Lifestyle/Metformin</td></tr>
                  <tr><td>Complications</td><td style={{ color: ACCENT }}>Rare/absent</td><td>Progressive risk</td><td>Progressive risk</td></tr>
                  <tr><td>Family hx</td><td>~75–80%</td><td>~90%</td><td>Variable</td></tr>
                  <tr><td>Antibodies</td><td style={{ color: ACCENT }}>NEGATIVE</td><td style={{ color: ACCENT }}>NEGATIVE</td><td>N/A</td></tr>
                  <tr><td>C-peptide</td><td style={{ color: ACCENT }}>Normal</td><td>Preserved early</td><td>Normal/high</td></tr>
                  <tr><td>Renal glycosuria</td><td style={{ color: ACCENT }}>ABSENT (0%)</td><td style={{ color: ACCENT4 }}>50% present</td><td>ABSENT</td></tr>
                  <tr><td>Sulfo response</td><td style={{ color: ACCENT3 }}>Hypoglycaemia</td><td style={{ color: ACCENT }}>Excellent</td><td>Moderate</td></tr>
                </tbody>
              </table>
            </Section>
          </div>
        </div>
      )}

      {/* Tab 3: Definitions */}
      {tab === 3 && (
        <div className="row g-3">
          <div className="col-12">
            <Section title="📖 Glossary — MODY2 / GCK-MODY" color={ACCENT}>
              <div className="row g-2">
                {(definitions?.terms || []).map((term, i) => (
                  <div key={i} className="col-md-6 col-lg-4">
                    <div className="card h-100 shadow-sm">
                      <div className="card-body py-2 px-3">
                        <div className="fw-bold small mb-1" style={{ color: ACCENT }}>{term.term}</div>
                        <div className="text-muted" style={{ fontSize: '0.78em' }}>{term.definition}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          </div>
        </div>
      )}
    </div>
  );
}

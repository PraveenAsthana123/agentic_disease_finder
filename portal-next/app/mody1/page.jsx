'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variant & Cohort', 'Treatment & Neonatal', 'Definitions'];

// MODY1 colour scheme — indigo/amber (HNF4A; macrosomia neonatal clue; sulfo-responsive)
const ACCENT  = '#283593';   // deep indigo — HNF4A; 2nd most common MODY
const ACCENT2 = '#1b5e20';   // deep green — sulfo SUCCESS; 85-90% response
const ACCENT3 = '#b71c1c';   // deep red — misdiagnosis; neonatal risk
const ACCENT4 = '#e65100';   // deep orange — macrosomia; neonatal hypoglycaemia; distinct feature
const ACCENT5 = '#4a148c';   // purple — haploinsufficiency; HNF4A→HNF1A axis
const ACCENT6 = '#0d47a1';   // deep blue — HbA1c; clinical metrics
const ACCENT7 = '#37474f';   // dark slate — epidemiology
const ACCENT8 = '#6a1b9a';   // dark violet — variants

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

export default function MODY1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/mody1/overview`).then(r => r.json()),
      fetch(`${API}/api/mody1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mody1/definitions`).then(r => r.json()),
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
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT4}11)`, border: `1px solid ${ACCENT}44` }}>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span style={{ fontSize: '1.6rem' }}>🧬</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>MODY1 — HNF4A-MODY</h4>
            <div className="text-muted small">Maturity-Onset Diabetes of the Young Type 1 · Chr 20q13.12 · OMIM #125850 · 2nd most common MODY (~10%) · Neonatal macrosomia + TNH · Autosomal Dominant · Sulfo-responsive</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="HNF4A *600281" color={ACCENT} />
            <Badge text="AD 50% risk" color={ACCENT5} />
            <Badge text="Sulfo first-line" color={ACCENT2} />
            <Badge text="Macrosomia 50–60%" color={ACCENT4} />
            <Badge text="Antibody negative" color={ACCENT7} />
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Cohort" value={kpis.cohort_size || 40} color={ACCENT} />
        <KPI label="Mean Age (yr)" value={kpis.mean_age_years?.toFixed(1)} color={ACCENT6} />
        <KPI label="Dx Age (yr)" value={kpis.mean_age_at_diagnosis_years?.toFixed(1)} color={ACCENT4} />
        <KPI label="Mean HbA1c (%)" value={kpis.mean_hba1c_percent?.toFixed(1)} color={ACCENT6} />
        <KPI label="On Sulfo (%)" value={`${kpis.pct_on_sulfonylurea?.toFixed(0)}%`} color={ACCENT2} />
        <KPI label="Sulfo Excellent" value={`${kpis.pct_sulfo_excellent_response?.toFixed(0)}%`} color={ACCENT2} />
        <KPI label="Antibody Neg" value="100%" color={ACCENT7} />
        <KPI label="Family Hx (%)" value={`${kpis.pct_family_hx_positive?.toFixed(0)}%`} color={ACCENT5} />
        <KPI label="Macrosomia" value={`${kpis.pct_neonatal_macrosomia?.toFixed(0)}%`} color={ACCENT4} />
        <KPI label="Neonatal Hypo" value={`${kpis.pct_neonatal_hypoglycaemia?.toFixed(0)}%`} color={ACCENT3} />
        <KPI label="Mean C-pep (nmol/L)" value={kpis.mean_c_peptide_nmol_L?.toFixed(2)} color={ACCENT} />
        <KPI label="Misdiagnosed" value={`${kpis.pct_prior_misdiagnosis?.toFixed(0)}%`} color={ACCENT3} />
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
              <Alert color={ACCENT4}>
                <strong>MODY1-UNIQUE:</strong> ~50–60% of HNF4A carriers are macrosomic at birth (≥4 kg) due to fetal hyperinsulinism. Transient neonatal hypoglycaemia resolves by 3 months — then adult-onset diabetes decades later.
              </Alert>
              <Alert color={ACCENT2}>
                <strong>SULFO-FIRST:</strong> Sulfonylurea gives ~85–90% excellent/good response. Less extreme sensitivity than MODY3 but still markedly more than T2D — start at 2.5 mg glibenclamide.
              </Alert>
              <Alert color={ACCENT3}>
                <strong>NO RENAL GLYCOSURIA:</strong> Urine dipstick NEGATIVE — key differentiator from MODY3 (50% glycosuria). HNF4A does NOT regulate SGLT2. Normal renal glucose threshold.
              </Alert>
              <table className="table table-sm table-bordered">
                <tbody>
                  <tr><td className="fw-bold">Gene</td><td>HNF4A (Hepatocyte Nuclear Factor 4α) · Chr 20q13.12 · *600281</td></tr>
                  <tr><td className="fw-bold">Disease OMIM</td><td>#125850 (MODY1)</td></tr>
                  <tr><td className="fw-bold">Inheritance</td><td>Autosomal Dominant · 50% recurrence per child</td></tr>
                  <tr><td className="fw-bold">Mechanism</td><td>HNF4A LOF → ↓HNF1A → impaired GSIS → progressive beta-cell failure; fetal phase → paradoxical hyperinsulinism</td></tr>
                  <tr><td className="fw-bold">Prevalence</td><td>~1:50,000–1:100,000 (~10% of all MODY, 2nd most common)</td></tr>
                  <tr><td className="fw-bold">MODY fraction</td><td>~10% of all MODY (after MODY3 ~35%)</td></tr>
                  <tr><td className="fw-bold">Onset age</td><td>Teens to early 40s (mean ~27 yr, slightly later than MODY3)</td></tr>
                  <tr><td className="fw-bold">C-peptide</td><td>Preserved at diagnosis; declines with duration</td></tr>
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
            <Section title="🩺 Diagnostic Criteria" color={ACCENT4}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(overview?.diagnostic_criteria || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold small">{k}</td><td className="small">{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="💊 Treatment Summary" color={ACCENT2}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(overview?.treatment_summary || {}).map(([k, v]) => (
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
            <Section title="👶 Neonatal Summary" color={ACCENT4}>
              <table className="table table-sm table-bordered">
                <tbody>
                  <tr><td className="fw-bold">Macrosomia (birth ≥4 kg)</td><td>{breakdown?.neonatal_summary?.macrosomia_count} / 40 ({breakdown?.neonatal_summary?.macrosomia_pct}%)</td></tr>
                  <tr><td className="fw-bold">Neonatal hypoglycaemia</td><td>{breakdown?.neonatal_summary?.neonatal_hypoglycaemia_count} / 40 ({breakdown?.neonatal_summary?.neonatal_hypoglycaemia_pct}%)</td></tr>
                  <tr><td className="fw-bold">Both macrosomia + hypo</td><td>{breakdown?.neonatal_summary?.both_macrosomia_and_hypo} / 40</td></tr>
                  <tr><td className="fw-bold">Renal glycosuria</td><td className="text-success fw-bold">0 / 40 (0%) — ABSENT in MODY1</td></tr>
                </tbody>
              </table>
              <Alert color={ACCENT4}>
                <strong>Neonatal clue:</strong> Macrosomia or neonatal hypoglycaemia in a child whose parent later develops young-onset diabetes is pathognomonic for MODY1 family. Screen all first-degree relatives.
              </Alert>
            </Section>
          </div>
          <div className="col-lg-7">
            <Section title="👩‍⚕️ Patient Cohort (40 patients, seed 305)" color={ACCENT}>
              <div style={{ maxHeight: 520, overflowY: 'auto' }}>
                <table className="table table-sm table-bordered table-hover" style={{ fontSize: '0.73em' }}>
                  <thead className="table-dark">
                    <tr>
                      <th>ID</th><th>Age</th><th>Sex</th><th>Dx Age</th><th>HbA1c%</th>
                      <th>C-pep</th><th>Variant</th><th>Treatment</th><th>Response</th><th>Macrosomia</th><th>Neon.Hypo</th>
                    </tr>
                  </thead>
                  <tbody>
                    {patients.map(p => (
                      <tr key={p.patient_id}>
                        <td>{p.patient_id}</td>
                        <td>{p.age}</td>
                        <td>{p.sex}</td>
                        <td>{p.age_at_diagnosis}</td>
                        <td><span style={{ color: p.hba1c_percent > 8 ? ACCENT3 : p.hba1c_percent > 7 ? ACCENT4 : ACCENT2, fontWeight: 600 }}>{p.hba1c_percent?.toFixed(1)}</span></td>
                        <td>{p.c_peptide_nmol_L?.toFixed(2)}</td>
                        <td style={{ fontSize: '0.68em' }}>{p.variant}</td>
                        <td style={{ fontSize: '0.68em' }}>{p.current_treatment}</td>
                        <td>
                          <span className="badge" style={{ background: p.sulfo_response === 'Excellent' ? ACCENT2 : p.sulfo_response === 'Good' ? ACCENT : p.sulfo_response === 'Partial' ? ACCENT4 : ACCENT7, fontSize: '0.65em' }}>
                            {p.sulfo_response}
                          </span>
                        </td>
                        <td>{p.neonatal_macrosomia ? '✅' : '—'}</td>
                        <td>{p.neonatal_hypoglycaemia ? '⚠️' : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* Tab 2: Treatment & Neonatal */}
      {tab === 2 && (
        <div className="row g-3">
          <div className="col-lg-6">
            <Section title="💊 Treatment Distribution" color={ACCENT2}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Treatment</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.treatment_distribution || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="✅ Sulfo Response" color={ACCENT}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Response</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.sulfo_response_distribution || {}).map(([r, n]) => (
                    <tr key={r}>
                      <td><span className="badge me-1" style={{ background: r === 'Excellent' ? ACCENT2 : r === 'Good' ? ACCENT : r === 'Partial' ? ACCENT4 : ACCENT7, fontSize: '0.7em' }}>{r}</span></td>
                      <td>{n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <Alert color={ACCENT2}>
                <strong>Key:</strong> Start sulfonylure at 2.5 mg glibenclamide. MODY1 sensitivity less extreme than MODY3 but still markedly greater than T2D. Titrate slowly; monitor for hypoglycaemia.
              </Alert>
            </Section>
            <Section title="🔁 Prior Misdiagnosis" color={ACCENT3}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Prior Diagnosis</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.misdiagnosis_distribution || {}).map(([d, n]) => (
                    <tr key={d}>
                      <td><span className="badge me-1" style={{ background: d === 'T1D' ? ACCENT3 : d === 'T2D' ? ACCENT6 : d === 'None' ? ACCENT2 : ACCENT4, fontSize: '0.7em' }}>{d}</span></td>
                      <td>{n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <Alert color={ACCENT3}>
                <strong>Action:</strong> Confirmed MODY1 → STOP unnecessary insulin → switch to low-dose sulfonylure. Check neonatal records (macrosomia?) and screen first-degree relatives.
              </Alert>
            </Section>
          </div>
          <div className="col-lg-6">
            <Section title="⚠️ Complications" color={ACCENT7}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Complication</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.complication_distribution || {}).map(([c, n]) => (
                    <tr key={c}><td>{c}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
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
            <Section title="🧬 MODY1 vs MODY3 — Key Differentiators" color={ACCENT5}>
              <table className="table table-sm table-bordered" style={{ fontSize: '0.8em' }}>
                <thead><tr><th>Feature</th><th>MODY1 (HNF4A)</th><th>MODY3 (HNF1A)</th></tr></thead>
                <tbody>
                  <tr><td>MODY fraction</td><td>~10%</td><td style={{ color: ACCENT2 }}>~35% (most common)</td></tr>
                  <tr><td>Chromosome</td><td>20q13.12</td><td>12q24.31</td></tr>
                  <tr><td>Renal glycosuria</td><td style={{ color: ACCENT2 }}>ABSENT (0%)</td><td style={{ color: ACCENT4 }}>~50% present</td></tr>
                  <tr><td>Neonatal macrosomia</td><td style={{ color: ACCENT4 }}>~50–60% YES</td><td style={{ color: ACCENT7 }}>Absent</td></tr>
                  <tr><td>Neonatal hyperinsulinism</td><td style={{ color: ACCENT4 }}>~50% (transient)</td><td style={{ color: ACCENT7 }}>Absent</td></tr>
                  <tr><td>Sulfo sensitivity</td><td>Marked (less extreme)</td><td style={{ color: ACCENT2 }}>Extreme (100–1000×)</td></tr>
                  <tr><td>Mean onset age</td><td>~27 yr</td><td>~24–25 yr</td></tr>
                  <tr><td>HNF relationship</td><td style={{ color: ACCENT5 }}>HNF4A → regulates HNF1A</td><td>HNF1A downstream target</td></tr>
                  <tr><td>Antibodies</td><td style={{ color: ACCENT2 }}>NEGATIVE</td><td style={{ color: ACCENT2 }}>NEGATIVE</td></tr>
                  <tr><td>C-peptide</td><td>Preserved early</td><td>Preserved early</td></tr>
                  <tr><td>Family history</td><td>~85–90%</td><td>~90%</td></tr>
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
            <Section title="📖 Glossary — MODY1 / HNF4A-MODY" color={ACCENT}>
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

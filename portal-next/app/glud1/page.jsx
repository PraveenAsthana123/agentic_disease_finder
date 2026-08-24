'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// GLUD1 color scheme — Gain-of-Function / Hyperinsulinism + Hyperammonemia / Absence Epilepsy
const ACCENT  = '#1a237e';   // deep indigo — GLUD1 GoF / dominant inheritance
const ACCENT2 = '#b71c1c';   // deep red — hypoglycemia CRITICALLY LOW / danger
const ACCENT3 = '#1b5e20';   // deep green — diazoxide treatment (primary, excellent response)
const ACCENT4 = '#e65100';   // burnt orange — hyperammonemia (persistent, always present)
const ACCENT5 = '#880e4f';   // dark pink — absence seizures (most characteristic)
const ACCENT6 = '#006064';   // teal — key negatives (normal biomarkers)
const ACCENT7 = '#0277bd';   // blue — treatments / management
const ACCENT8 = '#4a148c';   // purple — gain-of-function / dominant (opposite of LOF)

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
        <span>{label}</span>
        <span className="fw-bold">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function InfoBox({ title, children, color = ACCENT }) {
  return (
    <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body py-2">
        <div className="fw-bold small mb-1" style={{ color }}>{title}</div>
        <div className="small text-muted">{children}</div>
      </div>
    </div>
  );
}

export default function GLUD1Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/glud1/overview`).then(r => r.json()),
      fetch(`${API}/api/glud1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/glud1/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading GLUD1 dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <div>
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            ⚡ GLUD1 Epilepsy Dashboard
          </h4>
          <div className="text-muted small">
            {ov?.subtitle}
          </div>
          <div className="mt-1">
            <span className="badge me-1" style={{ background: ACCENT }}>10q23.3</span>
            <span className="badge me-1" style={{ background: ACCENT8 }}>AD GoF</span>
            <span className="badge me-1" style={{ background: ACCENT2 }}>Hypoglycemia</span>
            <span className="badge me-1" style={{ background: ACCENT4 }}>Hyperammonemia</span>
            <span className="badge me-1" style={{ background: ACCENT5 }}>Absence Epilepsy</span>
            <span className="badge me-1" style={{ background: ACCENT6 }}>OMIM #606762</span>
            <Link href="/gamt" className="badge text-decoration-none ms-1" style={{ background: ACCENT7 }}>← GAMT (creatine/GAA)</Link>
          </div>
        </div>
      </div>

      {/* Pathway position alert */}
      <div className="alert alert-warning py-2 small mb-3">
        <strong>GLUD1 — Glutamate Catabolism Node (Gain-of-Function — OPPOSITE to all LOF metabolic epilepsies):</strong>&nbsp;
        <strong style={{ color: ACCENT8 }}>GLUD1 GoF (Glu→α-KG, hyperactive)</strong>
        &nbsp;→&nbsp;
        <span style={{ color: ACCENT2 }}>β-cell excess α-KG → K-ATP CLOSE → insulin EXCESS → HYPOGLYCEMIA</span>
        &nbsp;|&nbsp;
        <span style={{ color: ACCENT4 }}>Liver excess NH₄⁺ → HYPERAMMONEMIA</span>
        &nbsp;|&nbsp;
        Related proline synthesis: <Link href="/aldh18a1" style={{ color: ACCENT }}>ALDH18A1</Link> (Glu→P5C, LOF)
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <>
          {/* KPIs */}
          <div className="row g-2 mb-3">
            <KPI label="Glucose nadir (mmol/L)" value={kpi.avg_glucose_mmol_l} color={ACCENT2} />
            <KPI label="Insulin (µU/mL)" value={kpi.avg_insulin_uU_ml} color={ACCENT2} />
            <KPI label="Ammonia (µmol/L)" value={kpi.avg_ammonia_umol_l} color={ACCENT4} />
            <KPI label="Glutamate (µmol/L)" value={kpi.avg_glutamate_umol_l} color={ACCENT} />
            <KPI label="Glutamine (µmol/L)" value={kpi.avg_glutamine_umol_l} color={ACCENT} />
            <KPI label="Hypoglycemia %" value={`${kpi.pct_hypoglycemia}%`} color={ACCENT2} />
            <KPI label="Seizures %" value={`${kpi.pct_seizures}%`} color={ACCENT5} />
            <KPI label="Absence %" value={`${kpi.pct_absence}%`} color={ACCENT5} />
            <KPI label="DRE %" value={`${kpi.pct_dre}%`} color={ACCENT5} />
            <KPI label="Diazoxide Response" value={`${kpi.pct_diazoxide_resp}%`} color={ACCENT3} />
            <KPI label="Leucine Sensitive" value={`${kpi.pct_leucine_sens}%`} color={ACCENT4} />
            <KPI label="IDD %" value={`${kpi.pct_idd}%`} color={ACCENT} />
          </div>

          {/* Key Biochemistry — two panels */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT2}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT2 }}>🔴 Dual Pathology — Glucose LOW + Ammonia HIGH</h6>
                  <p className="small text-muted mb-2">{ov?.key_positive_features}</p>
                  <div className="alert alert-danger py-1 small mb-1">
                    <strong>Glucose {kpi.avg_glucose_mmol_l} mmol/L</strong> (episode nadir; normal 3.5–6.0) + <strong>Insulin HIGH {kpi.avg_insulin_uU_ml} µU/mL</strong> — inappropriate.
                  </div>
                  <div className="alert alert-warning py-1 small mb-0">
                    <strong>Ammonia {kpi.avg_ammonia_umol_l} µmol/L</strong> (normal &lt;50) — PERSISTENTLY elevated regardless of meals.
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT6}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT6 }}>✅ Key Negatives (What Is NORMAL)</h6>
                  <p className="small text-muted mb-2">{ov?.key_negative_features}</p>
                  <div className="alert alert-success py-1 small mb-0">
                    <strong>alpha-AASA NORMAL</strong> (vs PDE/ALDH7A1), <strong>MMA NORMAL</strong> (vs MMUT/cblC),<br/>
                    <strong>tHcy NORMAL</strong> (vs CBS/MTHFR), <strong>GAA NORMAL</strong> (vs GAMT).
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Mechanism */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT8 }}>⚙️ GoF Mechanism — GTP-Regulatory Domain & Dual-Organ Pathology</h6>
              <p className="small text-muted">{ov?.function}</p>
              <hr className="my-2"/>
              <p className="small text-muted mb-0">{ov?.mechanism}</p>
            </div>
          </div>

          {/* Pathway position */}
          {ov?.pathway_position && (
            <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT4 }}>🗺️ Metabolic Pathway Position</h6>
                <div className="small text-muted">
                  <strong>GLUD1 reaction:</strong> L-Glutamate + NAD(P)⁺ → α-Ketoglutarate + NH₄⁺ + NAD(P)H<br/>
                  <strong>Step:</strong> {ov.pathway_position.step}<br/>
                  <strong>Upstream substrate:</strong> {ov.pathway_position.upstream}<br/>
                  <strong>Downstream:</strong> {ov.pathway_position.downstream}<br/>
                  <strong>Position:</strong> {ov.pathway_position.position_summary}
                </div>
              </div>
            </div>
          )}

          {/* Phenotype Distribution */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT }}>📊 Phenotype Distribution (n={ov?.cohort_n})</h6>
              <div className="row g-2 mt-1">
                {ov?.phenotype_distribution && Object.entries(ov.phenotype_distribution).map(([k, v]) => (
                  <div className="col-md-4" key={k}>
                    <div className="border rounded p-2 text-center">
                      <div className="fw-bold" style={{ color: ACCENT }}>{v.pct}%</div>
                      <div className="small text-muted">{k.replace(/-/g, ' ')}</div>
                      <div className="small text-muted">n = {v.n}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* NBS */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <InfoBox title="🧪 Newborn Screening (NBS)" color={ACCENT7}>
                <strong>Primary:</strong> {ov?.nbs_primary}<br/>
                <strong>Secondary:</strong> {ov?.nbs_secondary}
              </InfoBox>
            </div>
            <div className="col-md-6">
              <InfoBox title="📋 Disease Overview" color={ACCENT}>
                <strong>Gene:</strong> {ov?.gene} | <strong>Chr:</strong> {ov?.chromosome}<br/>
                <strong>Protein:</strong> {ov?.protein_size}<br/>
                <strong>OMIM Gene:</strong> {ov?.omim_gene} | <strong>Disease:</strong> {ov?.omim_disease}<br/>
                <strong>Prevalence:</strong> {ov?.prevalence}<br/>
                <strong>Inheritance:</strong> {ov?.inheritance}
              </InfoBox>
            </div>
          </div>

          {/* vs GAMT comparison */}
          {ov?.vs_gamt && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT7 }}>🔬 GLUD1 vs GAMT (both ammonia-connected epilepsies)</h6>
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr><td className="fw-bold">Shared</td><td>{ov.vs_gamt.shared}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT }}>GLUD1 (HHS)</td><td>{ov.vs_gamt.GLUD1}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT7 }}>GAMT (CCDS2)</td><td>{ov.vs_gamt.GAMT}</td></tr>
                    <tr><td className="fw-bold">Epilepsy</td><td>{ov.vs_gamt.epilepsy}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {ov?.vs_aldh18a1 && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT4 }}>🔬 GLUD1 vs ALDH18A1 (both glutamate-connected; opposite inheritance/direction)</h6>
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr><td className="fw-bold">Shared</td><td>{ov.vs_aldh18a1.shared}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT }}>GLUD1 (GoF, AD)</td><td>{ov.vs_aldh18a1.GLUD1}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT4 }}>ALDH18A1 (LOF, AR)</td><td>{ov.vs_aldh18a1.ALDH18A1}</td></tr>
                    <tr><td className="fw-bold">Epilepsy</td><td>{ov.vs_aldh18a1.epilepsy}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* ── TAB 1: PATIENTS & BIOMARKERS ── */}
      {tab === 1 && bd && (
        <>
          {/* Biomarkers */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT }}>🧪 Biomarker Profile (n={ov?.cohort_n})</h6>
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead className="table-light">
                    <tr>
                      <th>Biomarker</th>
                      <th>Observed</th>
                      <th>Normal Range</th>
                      <th>Significance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd.biomarkers?.map((b, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{b.name}</td>
                        <td>{b.mean !== null ? `${b.mean} ${b.unit}` : b.unit}</td>
                        <td className="text-muted">{b.normal_range}</td>
                        <td className="text-muted">{b.significance}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Clinical features */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT4 }}>🧠 Clinical Features (% of cohort)</h6>
              {bd.clinical_features?.map((cf, i) => (
                <div key={i} className="mb-2">
                  <PctBar label={`${cf.feature} — ${cf.note}`} pct={cf.pct}
                    color={cf.pct === 100 ? ACCENT4 : cf.pct >= 80 ? ACCENT : cf.pct >= 60 ? ACCENT5 : ACCENT7} />
                </div>
              ))}
            </div>
          </div>

          {/* Variants */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT }}>🔬 Pathogenic Variants (GoF — GTP-regulatory / antenna domain)</h6>
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead className="table-light">
                    <tr><th>Variant</th><th>Domain</th><th>Freq%</th><th>Phenotype</th><th>Notes</th></tr>
                  </thead>
                  <tbody>
                    {bd.variants?.map((v, i) => (
                      <tr key={i}>
                        <td className="fw-semibold" style={{ color: ACCENT8 }}>{v.variant}</td>
                        <td>{v.domain}</td>
                        <td className="fw-bold">{v.freq_pct}%</td>
                        <td><span className="badge" style={{ background: ACCENT5 }}>{v.phenotype}</span></td>
                        <td className="text-muted small">{v.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── TAB 2: SEIZURES & TREATMENTS ── */}
      {tab === 2 && bd && (
        <>
          {/* Seizure types */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT5 }}>⚡ Seizure Types ({kpi.pct_seizures}% have seizures; absence MOST CHARACTERISTIC)</h6>
              <div className="alert alert-secondary py-1 small mb-3">
                <strong>Absence seizures {kpi.pct_absence}%</strong> — 3 Hz spike-wave, ammonia-driven GABA-A modulation — most characteristic.<br/>
                DRE: <strong>{kpi.pct_dre}%</strong> (less than most metabolic epilepsies — metabolic control with diazoxide reduces seizures significantly).
              </div>
              {bd.seizure_types?.map((s, i) => (
                <div key={i} className="mb-3 border-start ps-3" style={{ borderColor: ACCENT5 }}>
                  <div className="fw-semibold small" style={{ color: ACCENT5 }}>{s.type}</div>
                  <div className="d-flex align-items-center gap-2 my-1">
                    <div className="progress flex-grow-1" style={{ height: 8 }}>
                      <div className="progress-bar" style={{ width: `${s.pct_in_seizure_pts}%`, backgroundColor: ACCENT5 }} />
                    </div>
                    <span className="small fw-bold">{s.pct_in_seizure_pts}%</span>
                  </div>
                  <div className="small text-muted">{s.note}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Treatments */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT7 }}>💊 Treatments</h6>
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead className="table-light">
                    <tr><th>Treatment</th><th>Level</th><th>Dose</th><th>Mechanism</th><th>Contraindication</th></tr>
                  </thead>
                  <tbody>
                    {bd.treatments?.map((t, i) => {
                      const isCI   = t.level.includes('CONTRAINDICATION');
                      const isRisk = t.level.includes('HIGH RISK') || t.level.includes('MODERATE RISK');
                      const isA    = t.level.includes('Level A');
                      return (
                        <tr key={i} className={isCI ? 'table-danger' : isRisk ? 'table-warning' : isA ? 'table-success' : ''}>
                          <td className="fw-semibold">{t.treatment}</td>
                          <td><span className={`badge ${isCI ? 'bg-danger' : isRisk ? 'bg-warning text-dark' : isA ? 'bg-success' : 'bg-primary'}`}>{t.level}</span></td>
                          <td>{t.dose}</td>
                          <td className="text-muted">{t.mechanism}</td>
                          <td className="text-muted">{t.contraindication}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── TAB 3: DEFINITIONS ── */}
      {tab === 3 && def && (
        <>
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT }}>📖 Gene & Disease Definitions</h6>
              <table className="table table-sm small">
                <tbody>
                  <tr><td className="fw-bold">Gene</td><td>{def.gene_full_name}</td></tr>
                  <tr><td className="fw-bold">Chromosome</td><td>{def.chromosome}</td></tr>
                  <tr><td className="fw-bold">OMIM Gene</td><td>{def.gene_omim}</td></tr>
                  <tr><td className="fw-bold">OMIM Disease</td><td>{def.disease_omim}</td></tr>
                  <tr><td className="fw-bold">Disease Name</td><td>{def.disease_name}</td></tr>
                  <tr><td className="fw-bold">Inheritance</td><td>{def.inheritance}</td></tr>
                  <tr><td className="fw-bold">Protein</td><td>{def.protein}</td></tr>
                  <tr><td className="fw-bold">Reaction</td><td><code>{def.reaction}</code></td></tr>
                  <tr><td className="fw-bold">Pathway</td><td>{def.pathway}</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          {def.key_terms && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT4 }}>🔑 Key Terms</h6>
                {Object.entries(def.key_terms).map(([term, desc]) => (
                  <div key={term} className="mb-2 pb-2 border-bottom">
                    <div className="fw-semibold small" style={{ color: ACCENT }}>{term}</div>
                    <div className="small text-muted">{desc}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {def.differential_diagnosis && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT5 }}>⚖️ Differential Diagnosis</h6>
                {Object.entries(def.differential_diagnosis).map(([disease, desc]) => (
                  <div key={disease} className="mb-2 pb-2 border-bottom">
                    <div className="fw-semibold small" style={{ color: ACCENT5 }}>{disease}</div>
                    <div className="small text-muted">{desc}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {def.treatment_summary && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT7 }}>💊 Treatment Summary</h6>
                <table className="table table-sm small mb-0">
                  <tbody>
                    {Object.entries(def.treatment_summary).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ minWidth: 140, textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}</td>
                        <td className="text-muted">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="alert alert-secondary py-2 small">
            <strong>Cohort Note:</strong> {def.cohort_note}
          </div>
        </>
      )}
    </div>
  );
}

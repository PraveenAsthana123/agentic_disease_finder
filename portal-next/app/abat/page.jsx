'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// ABAT color scheme — GABA-T deficiency / GABA excess / severe neonatal epileptic encephalopathy
const ACCENT  = '#1a237e';   // deep indigo — ABAT LOF / autosomal recessive
const ACCENT2 = '#b71c1c';   // deep red — GABA critically high / absolute CI vigabatrin
const ACCENT3 = '#1b5e20';   // deep green — ACTH treatment (IS first-line)
const ACCENT4 = '#e65100';   // burnt orange — infantile spasms / DRE / refractory
const ACCENT5 = '#880e4f';   // dark pink — GABA excess / paradoxical hyperexcitability
const ACCENT6 = '#006064';   // teal — key negatives / distinguishing features
const ACCENT7 = '#0277bd';   // blue — treatments / pyridoxine B6
const ACCENT8 = '#4a148c';   // purple — vigabatrin absolute CI / mechanism warning

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

export default function ABATPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/abat/overview`).then(r => r.json()),
      fetch(`${API}/api/abat/breakdown`).then(r => r.json()),
      fetch(`${API}/api/abat/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading ABAT dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <div>
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            🧬 ABAT Epilepsy Dashboard
          </h4>
          <div className="text-muted small">
            {ov?.subtitle}
          </div>
          <div className="mt-1">
            <span className="badge me-1" style={{ background: ACCENT }}>16q22.2</span>
            <span className="badge me-1" style={{ background: ACCENT }}>AR LOF</span>
            <span className="badge me-1" style={{ background: ACCENT5 }}>GABA↑↑↑ CSF</span>
            <span className="badge me-1" style={{ background: ACCENT4 }}>Infantile Spasms</span>
            <span className="badge me-1" style={{ background: ACCENT2 }}>Vigabatrin ABSOLUTE CI</span>
            <span className="badge me-1" style={{ background: ACCENT6 }}>OMIM #613163</span>
            <Link href="/glud1" className="badge text-decoration-none ms-1" style={{ background: ACCENT7 }}>← GLUD1 (Glu→α-KG GoF)</Link>
          </div>
        </div>
      </div>

      {/* Pathway alert — critical vigabatrin warning */}
      <div className="alert alert-danger py-2 small mb-3">
        <strong>⚠️ ABAT — GABA Catabolism Block (Ultrarare LOF):</strong>&nbsp;
        <strong style={{ color: ACCENT5 }}>ABAT LOF (GABA → SSA blocked)</strong>
        &nbsp;→&nbsp;
        <span style={{ color: ACCENT2 }}>CSF GABA 15–30× normal → GABA-A/B receptor downregulation → paradoxical hyperexcitability → infantile spasms</span>
        &nbsp;|&nbsp;
        <strong style={{ color: ACCENT8 }}>VIGABATRIN = ABSOLUTE CONTRAINDICATION</strong> (vigabatrin inhibits ABAT — the deficient enzyme; worsens catastrophically)
        &nbsp;|&nbsp;
        For infantile spasms: use ACTH only.
        Related GABA pathway: <Link href="/glud1" style={{ color: ACCENT }}>GLUD1</Link> (glutamate GoF, upstream)
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
            <KPI label="CSF GABA (nmol/mL)" value={kpi.avg_csf_gaba_nmol_ml} color={ACCENT5} />
            <KPI label="Urine GABA (mmol/mol)" value={kpi.avg_urine_gaba_mmol_mol_cr} color={ACCENT5} />
            <KPI label="Plasma GABA (µmol/L)" value={kpi.avg_plasma_gaba_umol_l} color={ACCENT5} />
            <KPI label="β-Alanine (µmol/L)" value={kpi.avg_beta_alanine_umol_l} color={ACCENT4} />
            <KPI label="Homocarnosine CSF" value={kpi.avg_homocarnosine_csf} color={ACCENT4} />
            <KPI label="Seizures %" value={`${kpi.pct_seizures}%`} color={ACCENT4} />
            <KPI label="Infantile Spasms %" value={`${kpi.pct_infantile_spasms}%`} color={ACCENT4} />
            <KPI label="DRE %" value={`${kpi.pct_dre}%`} color={ACCENT2} />
            <KPI label="IDD %" value={`${kpi.pct_idd}%`} color={ACCENT} />
            <KPI label="Hypotonia %" value={`${kpi.pct_hypotonia}%`} color={ACCENT} />
            <KPI label="Hyperkinesia %" value={`${kpi.pct_hyperkinesia}%`} color={ACCENT} />
            <KPI label="Somnolence %" value={`${kpi.pct_somnolence}%`} color={ACCENT7} />
          </div>

          {/* Key Biochemistry */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT5}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT5 }}>🔴 GABA Dramatically Elevated — Pathognomonic</h6>
                  <p className="small text-muted mb-2">{ov?.key_positive_features}</p>
                  <div className="alert alert-danger py-1 small mb-1">
                    <strong>CSF GABA avg {kpi.avg_csf_gaba_nmol_ml} nmol/mL</strong> (normal &lt;50) — 15–30× normal. PRIMARY diagnostic marker.
                  </div>
                  <div className="alert alert-warning py-1 small mb-0">
                    <strong>β-Alanine {kpi.avg_beta_alanine_umol_l} µmol/L</strong> (ABAT shared substrate, also accumulates) +
                    <strong> Homocarnosine {kpi.avg_homocarnosine_csf} nmol/mL CSF</strong> (GABA dipeptide, supportive).
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT6}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT6 }}>✅ Key Negatives — vs SSADH + PDE</h6>
                  <p className="small text-muted mb-2">{ov?.key_negative_features}</p>
                  <div className="alert alert-success py-1 small mb-0">
                    <strong>GHB only mildly elevated</strong> (NOT dramatic — distinguishes from SSADH/ALDH5A1).<br/>
                    <strong>alpha-AASA NORMAL</strong> (vs PDE/ALDH7A1), <strong>SSA LOW-NORMAL</strong>, <strong>Pipecolic NORMAL</strong>.
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Vigabatrin warning box */}
          <div className="alert alert-danger mb-3">
            <h6 className="fw-bold mb-1">⛔ VIGABATRIN — ABSOLUTE CONTRAINDICATION IN ABAT DEFICIENCY</h6>
            <p className="small mb-0">
              Vigabatrin (Sabril) is a <strong>suicide inhibitor of GABA-transaminase (ABAT)</strong> — this is its pharmacological mechanism.
              In ABAT deficiency, ABAT is already non-functional. Vigabatrin irreversibly inhibits any residual ABAT activity
              → GABA accumulation worsens catastrophically → encephalopathy and seizures worsen.
              <br/><strong>For infantile spasms in confirmed ABAT deficiency: use ACTH monotherapy only.</strong>
            </p>
          </div>

          {/* Mechanism */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT }}>⚙️ ABAT LOF Mechanism — GABA Catabolism Block</h6>
              <p className="small text-muted">{ov?.function}</p>
              <hr className="my-2"/>
              <p className="small text-muted mb-0">{ov?.mechanism}</p>
            </div>
          </div>

          {/* Pathway position */}
          {ov?.pathway_position && (
            <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT4 }}>🗺️ GABA Catabolism Pathway Position</h6>
                <div className="small text-muted">
                  <strong>ABAT reaction:</strong> GABA + α-Ketoglutarate → SSA + Glutamate (PLP-dependent)<br/>
                  <strong>Step:</strong> {ov.pathway_position.step}<br/>
                  <strong>Upstream substrate:</strong> {ov.pathway_position.upstream}<br/>
                  <strong>Downstream:</strong> {ov.pathway_position.downstream}<br/>
                  <strong>Pathway summary:</strong> {ov.pathway_position.position_summary}
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

          {/* NBS + Disease info */}
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

          {/* vs SSADH comparison */}
          {ov?.vs_ssadh && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT7 }}>🔬 ABAT vs SSADH/ALDH5A1 (same GABA catabolic pathway)</h6>
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr><td className="fw-bold">Shared</td><td>{ov.vs_ssadh.shared}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT }}>ABAT (step 1)</td><td>{ov.vs_ssadh.ABAT}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT7 }}>SSADH (step 2)</td><td>{ov.vs_ssadh.SSADH}</td></tr>
                    <tr><td className="fw-bold">Epilepsy</td><td>{ov.vs_ssadh.epilepsy}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* vs GLUD1 comparison */}
          {ov?.vs_glud1 && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT4 }}>🔬 ABAT vs GLUD1 (both GABA/glutamate axis; opposite inheritance + direction)</h6>
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr><td className="fw-bold">Shared</td><td>{ov.vs_glud1.shared}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT }}>ABAT (LOF, AR)</td><td>{ov.vs_glud1.ABAT}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT4 }}>GLUD1 (GoF, AD)</td><td>{ov.vs_glud1.GLUD1}</td></tr>
                    <tr><td className="fw-bold">Epilepsy</td><td>{ov.vs_glud1.epilepsy}</td></tr>
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
                      <th>Observed Mean</th>
                      <th>Normal Range</th>
                      <th>Significance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd.biomarkers?.map((b, i) => (
                      <tr key={i} className={b.significance?.includes('PATHOGNOMONIC') ? 'table-danger' : b.significance?.includes('KEY NEGATIVE') ? 'table-success' : ''}>
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
                    color={cf.pct === 100 ? ACCENT5 : cf.pct >= 90 ? ACCENT4 : cf.pct >= 70 ? ACCENT : ACCENT7} />
                </div>
              ))}
            </div>
          </div>

          {/* Variants */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT }}>🔬 Pathogenic Variants (LOF — PLP-binding / catalytic / dimerisation)</h6>
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead className="table-light">
                    <tr><th>Variant</th><th>Domain</th><th>Freq%</th><th>Phenotype</th><th>Notes</th></tr>
                  </thead>
                  <tbody>
                    {bd.variants?.map((v, i) => (
                      <tr key={i}>
                        <td className="fw-semibold" style={{ color: ACCENT }}>{v.variant}</td>
                        <td>{v.domain}</td>
                        <td className="fw-bold">{v.freq_pct}%</td>
                        <td><span className="badge" style={{ background: v.phenotype === 'Severe-Neonatal' ? ACCENT2 : v.phenotype === 'Classic-Infantile' ? ACCENT4 : ACCENT7 }}>{v.phenotype}</span></td>
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
              <h6 className="fw-bold" style={{ color: ACCENT4 }}>⚡ Seizure Types ({kpi.pct_seizures}% have seizures; infantile spasms MOST CHARACTERISTIC)</h6>
              <div className="alert alert-danger py-1 small mb-3">
                <strong>Infantile spasms {kpi.pct_infantile_spasms}%</strong> — hypsarrhythmia EEG; GABA-A receptor downregulation → paradoxical hyperexcitability.<br/>
                DRE: <strong>{kpi.pct_dre}%</strong> — extremely refractory; no AED restores GABA catabolism.&nbsp;
                <strong style={{ color: ACCENT8 }}>VIGABATRIN ABSOLUTELY CONTRAINDICATED — use ACTH for IS.</strong>
              </div>
              {bd.seizure_types?.map((s, i) => (
                <div key={i} className="mb-3 border-start ps-3" style={{ borderColor: ACCENT4 }}>
                  <div className="fw-semibold small" style={{ color: ACCENT4 }}>{s.type}</div>
                  <div className="d-flex align-items-center gap-2 my-1">
                    <div className="progress flex-grow-1" style={{ height: 8 }}>
                      <div className="progress-bar" style={{ width: `${s.pct_in_seizure_pts}%`, backgroundColor: ACCENT4 }} />
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
              <h6 className="fw-bold" style={{ color: ACCENT7 }}>💊 Treatments (vigabatrin ABSOLUTE CI — first drug to exclude)</h6>
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead className="table-light">
                    <tr><th>Treatment</th><th>Level</th><th>Dose</th><th>Mechanism</th><th>Contraindication / Note</th></tr>
                  </thead>
                  <tbody>
                    {bd.treatments?.map((t, i) => {
                      const isCI   = t.level.includes('CONTRAINDICATION');
                      const isRisk = t.level.includes('MODERATE RISK');
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

'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// PYCR2 color scheme — Brain-enriched PYCR isoform / Hypomyelination (HLD10) / Proline CRITICALLY LOW
const ACCENT  = '#1a237e';   // deep indigo — brain-enriched isoform / CNS specificity
const ACCENT2 = '#b71c1c';   // deep red — proline critically LOW (synthesis failure)
const ACCENT3 = '#1b5e20';   // deep green — proline supplements / treatment
const ACCENT4 = '#4a148c';   // purple — hypomyelination / white matter disease
const ACCENT5 = '#880e4f';   // dark pink — seizures / DRE (higher than PYCR1)
const ACCENT6 = '#006064';   // teal — normal biomarkers (PLP, alpha-AASA — KEY NEGATIVES)
const ACCENT7 = '#0277bd';   // blue — treatments / spasticity
const ACCENT8 = '#e65100';   // burnt orange — vs PYCR1 distinction / cutis laxa ABSENT

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

export default function PYCR2Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/pycr2/overview`).then(r => r.json()),
      fetch(`${API}/api/pycr2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/pycr2/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading PYCR2 dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <div>
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            🧬 PYCR2 Epilepsy Dashboard
          </h4>
          <div className="text-muted small">
            {ov?.subtitle}
          </div>
          <div className="mt-1">
            <span className="badge me-1" style={{ background: ACCENT }}>1q42.12</span>
            <span className="badge me-1 bg-secondary">AR</span>
            <span className="badge me-1" style={{ background: ACCENT2 }}>Proline CRITICALLY LOW</span>
            <span className="badge me-1" style={{ background: ACCENT4 }}>Hypomyelination (HLD10)</span>
            <span className="badge me-1" style={{ background: ACCENT6 }}>OMIM #616138</span>
            <Link href="/pycr1" className="badge text-decoration-none ms-1" style={{ background: ACCENT8 }}>← PYCR1 (ARCL2B, cutis laxa)</Link>
          </div>
        </div>
      </div>

      {/* Pathway position alert */}
      <div className="alert alert-secondary py-2 small mb-3">
        <strong>Proline Synthesis Pathway (same step as PYCR1, brain-specific):</strong>&nbsp;
        <Link href="/aldh18a1" style={{ color: ACCENT4 }}>ALDH18A1/P5CS (Glu→P5C, Step 1)</Link>
        &nbsp;→&nbsp;
        <strong style={{ color: ACCENT }}>PYCR2 (P5C→Proline, brain — FINAL step)</strong>
        &nbsp;|&nbsp;
        Catabolism (reverse): <Link href="/prodh" style={{ color: ACCENT2 }}>PRODH</Link> → <Link href="/aldh4a1" style={{ color: ACCENT2 }}>ALDH4A1</Link>
        &nbsp;|&nbsp;
        <span style={{ color: ACCENT8 }}>Cutis Laxa sibling: <Link href="/pycr1" style={{ color: ACCENT8 }}>PYCR1</Link></span>
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
            <KPI label="Avg Proline (µmol/L)" value={kpi.avg_proline_umol_l} color={ACCENT2} />
            <KPI label="Avg P5C (µmol/L)" value={kpi.avg_p5c_umol_l} color={ACCENT4} />
            <KPI label="Avg PLP (nmol/L)" value={kpi.avg_plp_nmol_l} color={ACCENT6} />
            <KPI label="Avg Ornithine (µmol/L)" value={kpi.avg_ornithine_umol_l} color={ACCENT} />
            <KPI label="Seizures %" value={`${kpi.pct_seizures}%`} color={ACCENT5} />
            <KPI label="DRE %" value={`${kpi.pct_dre}%`} color={ACCENT5} />
            <KPI label="B6 Response %" value={`${kpi.pct_b6_responded}%`} color={ACCENT6} />
            <KPI label="IDD %" value={`${kpi.pct_idd}%`} color={ACCENT} />
            <KPI label="Hypomyelination %" value={`${kpi.pct_hypomyelination}%`} color={ACCENT4} />
            <KPI label="Spastic Paresis %" value={`${kpi.pct_spastic}%`} color={ACCENT7} />
            <KPI label="On Proline Supp." value={`${kpi.pct_proline_supplemented}%`} color={ACCENT3} />
            <KPI label="Cutis Laxa %" value={`${kpi.pct_cutis_laxa}%`} color={ACCENT8} />
          </div>

          {/* Key Biochemistry — two panels */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT2}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT2 }}>🔴 Proline CRITICALLY LOW — Primary Biomarker</h6>
                  <p className="small text-muted mb-2">{ov?.key_positive_features}</p>
                  <div className="alert alert-danger py-1 small mb-0">
                    <strong>Proline {kpi.avg_proline_umol_l} µmol/L</strong> (Normal: 100–260 µmol/L) — CNS synthesis failure.<br/>
                    P5C mildly ↑ ({kpi.avg_p5c_umol_l} µmol/L) — substrate backup (ALDH18A1 intact; PYCR2 absent in brain).
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
                    <strong>PLP NORMAL</strong> ({kpi.avg_plp_nmol_l} nmol/L) — No B6 deficiency; B6 NOT indicated.<br/>
                    <strong>Cutis Laxa: 0%</strong> — Skin uses PYCR1 (intact); KEY vs PYCR1 (90% cutis laxa).
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Mechanism */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT }}>⚙️ Enzymatic Function & Epileptogenic Mechanism</h6>
              <p className="small text-muted">{ov?.function}</p>
              <hr className="my-2"/>
              <p className="small text-muted mb-0">{ov?.mechanism}</p>
            </div>
          </div>

          {/* Pathway position */}
          {ov?.pathway_position && (
            <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT4 }}>🗺️ Proline Pathway Position (Brain)</h6>
                <div className="small text-muted">
                  <strong>PYCR2 reaction:</strong> P5C + NADPH → L-Proline + NADP⁺ (brain-enriched)<br/>
                  <strong>Step:</strong> {ov.pathway_position.step}<br/>
                  <strong>Upstream:</strong> {ov.pathway_position.upstream}<br/>
                  <strong>Downstream (target):</strong> {ov.pathway_position.downstream}<br/>
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

          {/* PYCR2 vs PYCR1 comparison */}
          {ov?.vs_pycr1 && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT8 }}>🔬 PYCR2 vs PYCR1 (sibling isoform — brain vs peripheral)</h6>
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr><td className="fw-bold">Shared</td><td>{ov.vs_pycr1.shared}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT }}>PYCR2 (HLD10)</td><td>{ov.vs_pycr1.PYCR2}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT8 }}>PYCR1 (ARCL2B)</td><td>{ov.vs_pycr1.PYCR1}</td></tr>
                    <tr><td className="fw-bold">Epilepsy</td><td>{ov.vs_pycr1.epilepsy}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {ov?.vs_aldh18a1 && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT4 }}>🔬 PYCR2 vs ALDH18A1 (upstream P5CS — entry step vs exit step)</h6>
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr><td className="fw-bold">Shared</td><td>{ov.vs_aldh18a1.shared}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT }}>PYCR2</td><td>{ov.vs_aldh18a1.PYCR2}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT4 }}>ALDH18A1</td><td>{ov.vs_aldh18a1.ALDH18A1}</td></tr>
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
                    color={cf.pct === 0 ? '#9e9e9e' : cf.pct >= 80 ? ACCENT : cf.pct >= 50 ? ACCENT4 : ACCENT7} />
                </div>
              ))}
            </div>
          </div>

          {/* Variants */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT }}>🔬 Pathogenic Variants</h6>
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
                        <td><span className="badge" style={{ background: ACCENT4 }}>{v.phenotype}</span></td>
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
              <h6 className="fw-bold" style={{ color: ACCENT5 }}>⚡ Seizure Types (% of seizure patients; {kpi.pct_seizures}% have seizures)</h6>
              <div className="alert alert-warning py-1 small mb-3">
                <strong>60–70% overall seizure rate</strong> — higher than PYCR1 (40–55%) due to hypomyelination-driven cortical hyperexcitability.
                DRE: <strong>{kpi.pct_dre}%</strong>. B6 response: <strong>0%</strong> (PLP NORMAL — no indication for pyridoxine).
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
                      const isCI   = t.level.includes('CONTRAINDICATION') || t.level.includes('NOT INDICATED');
                      const isRisk = t.level.includes('RISK');
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

          {/* Key terms */}
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

          {/* Differential Diagnosis */}
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

          {/* Treatment Summary */}
          {def.treatment_summary && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT7 }}>💊 Treatment Summary</h6>
                <table className="table table-sm small mb-0">
                  <tbody>
                    {Object.entries(def.treatment_summary).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ minWidth: 120, textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}</td>
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

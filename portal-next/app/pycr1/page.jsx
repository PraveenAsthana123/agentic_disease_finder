'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// PYCR1 color scheme — Proline SYNTHESIS exit step / Proline CRITICALLY LOW / Cutis Laxa 2B
const ACCENT  = '#37474f';   // slate blue-grey — P5C→Proline reduction (mitochondrial redox)
const ACCENT2 = '#b71c1c';   // deep red — proline critically LOW (synthesis failure)
const ACCENT3 = '#1b5e20';   // deep green — proline supplements / L-Proline treatment
const ACCENT4 = '#e65100';   // burnt orange — cutis laxa / connective tissue failure
const ACCENT5 = '#880e4f';   // dark pink — seizures / DRE
const ACCENT6 = '#006064';   // teal — normal biomarkers (PLP, alpha-AASA, tHcy — KEY NEGATIVES)
const ACCENT7 = '#0277bd';   // blue — treatments
const ACCENT8 = '#4a148c';   // purple — key distinctions vs ALDH18A1/ALDH4A1

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

export default function PYCR1Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/pycr1/overview`).then(r => r.json()),
      fetch(`${API}/api/pycr1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/pycr1/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading PYCR1 dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <div>
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            🧬 PYCR1 Epilepsy Dashboard
          </h4>
          <div className="text-muted small">
            {ov?.subtitle}
          </div>
          <div className="mt-1">
            <span className="badge me-1" style={{ background: ACCENT }}>17q25.3</span>
            <span className="badge me-1 bg-secondary">AR</span>
            <span className="badge me-1" style={{ background: ACCENT2 }}>Proline CRITICALLY LOW</span>
            <span className="badge me-1" style={{ background: ACCENT4 }}>Cutis Laxa 2B</span>
            <span className="badge me-1" style={{ background: ACCENT6 }}>OMIM #612940</span>
            <Link href="/aldh18a1" className="badge text-decoration-none ms-1" style={{ background: ACCENT8 }}>← ALDH18A1 (upstream P5CS)</Link>
          </div>
        </div>
      </div>

      {/* Pathway position alert */}
      <div className="alert alert-secondary py-2 small mb-3">
        <strong>Proline Synthesis Pathway:</strong>&nbsp;
        <Link href="/aldh18a1" style={{ color: ACCENT8 }}>ALDH18A1/P5CS (Glu→P5C, Step 1)</Link>
        &nbsp;→&nbsp;
        <strong style={{ color: ACCENT }}>PYCR1 (P5C→Proline, Step 2 — FINAL)</strong>
        &nbsp;|&nbsp;
        Catabolism (reverse): <Link href="/prodh" style={{ color: ACCENT2 }}>PRODH</Link> → <Link href="/aldh4a1" style={{ color: ACCENT2 }}>ALDH4A1</Link>
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
            <KPI label="Cutis Laxa %" value={`${kpi.pct_cutis_laxa}%`} color={ACCENT4} />
            <KPI label="On Proline Supp." value={`${kpi.pct_proline_supplemented}%`} color={ACCENT3} />
            <KPI label="PLP Normal %" value={`${kpi.pct_plp_normal}%`} color={ACCENT6} />
            <KPI label="NBS Detected %" value={`${kpi.pct_nbs_detected}%`} color={ACCENT7} />
          </div>

          {/* Key Biochemistry */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT2}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT2 }}>🔴 Proline CRITICALLY LOW — Primary Biomarker</h6>
                  <p className="small text-muted mb-2">{ov?.key_positive_features}</p>
                  <div className="alert alert-danger py-1 small mb-0">
                    <strong>Proline {kpi.avg_proline_umol_l} µmol/L</strong> (Normal: 100–260 µmol/L) — synthesis exit failure.<br/>
                    P5C mildly ↑ ({kpi.avg_p5c_umol_l} µmol/L) — substrate backup (ALDH18A1 intact, PYCR1 absent).
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
                    <strong>B6 Response: 0%</strong> — PLP intact; no P5C-PLP inactivation (vs ALDH4A1).
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
            <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT8}` }}>
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT8 }}>🗺️ Proline Pathway Position</h6>
                <div className="small text-muted">
                  <strong>PYCR1 reaction:</strong> {ov.pathway_position.reaction || 'P5C + NADPH → L-Proline + NADP⁺'}<br/>
                  <strong>Step:</strong> {ov.pathway_position.step}<br/>
                  <strong>Upstream:</strong> {ov.pathway_position.upstream}<br/>
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

          {/* Cross-disease comparisons */}
          {ov?.vs_aldh18a1 && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT8 }}>🔬 PYCR1 vs ALDH18A1 (upstream P5CS)</h6>
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr><td className="fw-bold">Shared</td><td>{ov.vs_aldh18a1.shared}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT }}>PYCR1</td><td>{ov.vs_aldh18a1.PYCR1}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT8 }}>ALDH18A1</td><td>{ov.vs_aldh18a1.ALDH18A1}</td></tr>
                    <tr><td className="fw-bold">Epilepsy</td><td>{ov.vs_aldh18a1.epilepsy}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {ov?.vs_aldh4a1 && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT2 }}>🔬 PYCR1 vs ALDH4A1 (catabolism step 2)</h6>
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr><td className="fw-bold">Shared</td><td>{ov.vs_aldh4a1.shared}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT }}>PYCR1</td><td>{ov.vs_aldh4a1.PYCR1}</td></tr>
                    <tr><td className="fw-bold" style={{ color: ACCENT2 }}>ALDH4A1</td><td>{ov.vs_aldh4a1.ALDH4A1}</td></tr>
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
                      <th>Mean / Status</th>
                      <th>Normal Range</th>
                      <th>Significance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd.biomarkers?.map((b, i) => (
                      <tr key={i}>
                        <td className="fw-bold">{b.name}</td>
                        <td>
                          {b.mean != null
                            ? <span className="fw-bold" style={{ color: ACCENT2 }}>{b.mean} {b.unit}</span>
                            : <span style={{ color: ACCENT4 }}>{b.unit}</span>
                          }
                        </td>
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
              <h6 className="fw-bold" style={{ color: ACCENT4 }}>🏥 Clinical Features</h6>
              {bd.clinical_features?.map((f, i) => (
                <PctBar key={i} label={`${f.feature} — ${f.note}`} pct={f.pct}
                  color={f.feature.includes('Seizure') ? ACCENT5 : f.feature.includes('Cutis') ? ACCENT4 : ACCENT} />
              ))}
            </div>
          </div>

          {/* Variants */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT }}>🧬 Known Pathogenic Variants</h6>
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead className="table-light">
                    <tr>
                      <th>Variant</th>
                      <th>Domain</th>
                      <th>Freq %</th>
                      <th>Phenotype</th>
                      <th>Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd.variants?.map((v, i) => (
                      <tr key={i}>
                        <td className="fw-bold font-monospace">{v.variant}</td>
                        <td>{v.domain}</td>
                        <td><span className="badge" style={{ background: ACCENT }}>{v.freq_pct}%</span></td>
                        <td>{v.phenotype}</td>
                        <td className="text-muted">{v.note}</td>
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
              <h6 className="fw-bold" style={{ color: ACCENT5 }}>⚡ Seizure Types in PYCR1 (among seizure patients)</h6>
              {bd.seizure_types?.map((s, i) => (
                <div key={i} className="mb-3">
                  <PctBar label={s.type} pct={s.pct_in_seizure_pts} color={ACCENT5} />
                  <div className="small text-muted ms-2">{s.note}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Seizure summary boxes */}
          <div className="row g-3 mb-3">
            <div className="col-md-4">
              <div className="card shadow-sm text-center h-100" style={{ borderTop: `3px solid ${ACCENT5}` }}>
                <div className="card-body">
                  <div className="fw-bold fs-4" style={{ color: ACCENT5 }}>{kpi.pct_seizures}%</div>
                  <div className="small text-muted">Seizures overall</div>
                  <div className="small text-muted mt-1">(40–55% range; 48% in cohort)</div>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card shadow-sm text-center h-100" style={{ borderTop: `3px solid ${ACCENT5}` }}>
                <div className="card-body">
                  <div className="fw-bold fs-4" style={{ color: ACCENT5 }}>{kpi.pct_dre}%</div>
                  <div className="small text-muted">Drug-resistant epilepsy</div>
                  <div className="small text-muted mt-1">(15–25% range)</div>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card shadow-sm text-center h-100" style={{ borderTop: `3px solid ${ACCENT6}` }}>
                <div className="card-body">
                  <div className="fw-bold fs-4" style={{ color: ACCENT6 }}>0%</div>
                  <div className="small text-muted">B6/Pyridoxine response</div>
                  <div className="small text-muted mt-1">PLP NORMAL — B6 NOT indicated</div>
                </div>
              </div>
            </div>
          </div>

          {/* Treatments */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT7 }}>💊 Treatment Protocol</h6>
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead className="table-light">
                    <tr>
                      <th>Treatment</th>
                      <th>Level</th>
                      <th>Dose</th>
                      <th>Mechanism</th>
                      <th>Monitoring</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd.treatments?.map((t, i) => {
                      const lvlColor =
                        t.level === 'Level A' ? '#1b5e20' :
                        t.level === 'Level B' ? '#0277bd' :
                        t.level === 'NOT INDICATED' ? '#006064' :
                        t.level === 'ABSOLUTE CONTRAINDICATION' ? '#b71c1c' :
                        t.level === 'MODERATE RISK' ? '#e65100' : '#37474f';
                      return (
                        <tr key={i}>
                          <td className="fw-bold">{t.treatment}</td>
                          <td><span className="badge" style={{ background: lvlColor }}>{t.level}</span></td>
                          <td>{t.dose}</td>
                          <td className="text-muted">{t.mechanism}</td>
                          <td className="text-muted">{t.monitoring}</td>
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
                  <tr><td className="fw-bold">Gene Full Name</td><td>{def.gene_full_name}</td></tr>
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

          {/* Key Terms */}
          {def.key_terms && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT }}>🔑 Key Terms</h6>
                {Object.entries(def.key_terms).map(([term, desc]) => (
                  <div key={term} className="mb-2">
                    <span className="badge me-2" style={{ background: ACCENT }}>{term}</span>
                    <span className="small text-muted">{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Differential diagnosis */}
          {def.differential_diagnosis && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT8 }}>🔬 Differential Diagnosis</h6>
                <div className="table-responsive">
                  <table className="table table-sm small">
                    <thead className="table-light">
                      <tr><th>Condition</th><th>Key Distinctions from PYCR1</th></tr>
                    </thead>
                    <tbody>
                      {Object.entries(def.differential_diagnosis).map(([cond, detail]) => (
                        <tr key={cond}>
                          <td className="fw-bold">{cond}</td>
                          <td className="text-muted">{detail}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Treatment summary */}
          {def.treatment_summary && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT7 }}>💊 Treatment Summary</h6>
                <table className="table table-sm small">
                  <tbody>
                    {Object.entries(def.treatment_summary).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}</td>
                        <td className="text-muted">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="alert alert-secondary small">
            <strong>Cohort note:</strong> {def.cohort_note}
          </div>
        </>
      )}
    </div>
  );
}

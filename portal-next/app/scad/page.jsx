'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Treatments & Controversy', 'Definitions'];

// SCAD / ACADS colour scheme — amber-gold (FAO disorder; controversial NBS; usually benign)
const ACCENT  = '#e65100';   // deep orange — SCAD / short-chain FAO
const ACCENT2 = '#bf360c';   // dark burnt-orange — controversy / NBS warning
const ACCENT3 = '#1b5e20';   // deep green — KEY NEGATIVES (C8 normal, C14:1 normal)
const ACCENT4 = '#01579b';   // deep blue — comparison with MCAD/VLCAD
const ACCENT5 = '#4a148c';   // dark purple — common variants (625G>A / 511C>T)
const ACCENT6 = '#880e4f';   // dark rose — EMA / MSA urine markers
const ACCENT7 = '#37474f';   // dark slate — NBS / epidemiology
const ACCENT8 = '#006064';   // dark teal — riboflavin (B2) treatment

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
  const numPct = typeof pct === 'string' ? parseInt(pct) : pct;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${numPct}%`, backgroundColor: color }} />
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

export default function SCADPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/scad/overview`).then(r => r.json()),
      fetch(`${API}/api/scad/breakdown`).then(r => r.json()),
      fetch(`${API}/api/scad/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center text-muted">Loading SCAD Dashboard&hellip;</div>;
  if (err)     return <div className="p-4 text-center text-danger">Error: {err}</div>;

  const kpis   = ov?.kpis || {};
  const phDist = ov?.phenotype_distribution || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          &#x1f9ec; SCAD Deficiency Dashboard
        </h4>
        <div className="text-muted small">
          Short-Chain Acyl-CoA Dehydrogenase Deficiency &mdash; ACADS &middot; 12q24.31 &middot; AR &middot; OMIM #201470
        </div>
        <div className="text-muted small">
          <span className="badge me-1" style={{ background: ACCENT2 }}>FAO Triad: VLCAD &rarr; MCAD &rarr; SCAD</span>
          <span className="badge me-1" style={{ background: ACCENT5 }}>C4–C6 Chain Length</span>
          <span className="badge" style={{ background: ACCENT7 }}>Most Controversial NBS Disorder</span>
        </div>
      </div>

      {/* NBS Controversy Alert */}
      <div className="alert py-2 mb-3" style={{ background: '#fff3e0', borderLeft: `4px solid ${ACCENT2}`, fontSize: 13 }}>
        <strong>&#x26a0;&#xfe0f; NBS Controversy:</strong> SCAD is the <strong>most controversial</strong> disorder on NBS panels.
        Common variants <strong>625G>A (p.Gly209Ser)</strong> [7% population] and <strong>511C>T (p.Arg171Trp)</strong> [14% population]
        cause C4 elevation WITHOUT clinical disease. The UK removed SCAD from the NBS panel in 2012.
        Majority of NBS positives are asymptomatic common-variant carriers.
      </div>

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ─── TAB 0: Overview ─── */}
      {tab === 0 && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Total Patients"         value={kpis.total_patients}    color={ACCENT}  />
            <KPI label="Asymptomatic NBS"        value={kpis.asymptomatic_nbs}  color={ACCENT7} />
            <KPI label="Biochemical SCAD"        value={kpis.biochemical_n}     color={ACCENT2} />
            <KPI label="Classic SCAD"            value={kpis.classic_n}         color={ACCENT}  />
            <KPI label="With Seizures"           value={kpis.seizures_n}        color={ACCENT2} />
            <KPI label="Hypotonia"               value={kpis.hypotonia_n}       color={ACCENT4} />
            <KPI label="Dev Delay"               value={kpis.dev_delay_n}       color={ACCENT4} />
            <KPI label="Riboflavin Response"     value={kpis.riboflavin_resp_n} color={ACCENT8} />
            <KPI label="Avg C4 (µmol/L)"         value={kpis.avg_c4_umol}       color={ACCENT}  />
            <KPI label="Avg EMA (mmol/mol Cr)"   value={kpis.avg_ema_mmol_cr}   color={ACCENT6} />
            <KPI label="Avg C0 (µmol/L)"         value={kpis.avg_c0_umol}       color={ACCENT7} />
          </div>

          {/* Clinical summary */}
          <div className="row mb-3">
            <div className="col-md-8">
              <div className="card shadow-sm">
                <div className="card-body py-2">
                  <div className="fw-bold small mb-1" style={{ color: ACCENT }}>Clinical Summary</div>
                  <p className="small text-muted mb-0">{ov?.clinical_summary}</p>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-body py-2">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Phenotype Distribution</div>
                  {Object.entries(phDist).map(([ph, n]) => (
                    <PctBar key={ph} label={ph} pct={Math.round(100 * n / (kpis.total_patients || 1))}
                      color={ph.includes('Asymptomatic') ? ACCENT7 : ph.includes('Biochemical') ? ACCENT2 : ACCENT} />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* FAO Triad comparison */}
          <div className="row mb-3">
            <div className="col-12">
              <div className="card shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
                <div className="card-body py-2">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT4 }}>FAO Acyl-CoA Dehydrogenase Triad</div>
                  <div className="table-responsive">
                    <table className="table table-sm table-bordered small mb-0">
                      <thead style={{ background: '#e3f2fd' }}>
                        <tr>
                          <th>Enzyme</th><th>Gene</th><th>Chain Length</th><th>NBS Marker</th><th>Clinical Hallmark</th><th>Severity</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td><strong>VLCAD</strong></td><td>ACADVL</td><td>C14–C20 (very-long)</td>
                          <td>C14:1 &gt;0.3 µmol/L</td>
                          <td>Cardiomyopathy + Rhabdomyolysis</td>
                          <td><span className="badge bg-danger">Severe</span></td>
                        </tr>
                        <tr>
                          <td><strong>MCAD</strong></td><td>ACADM</td><td>C6–C12 (medium)</td>
                          <td>C8 &gt;0.3 µmol/L</td>
                          <td>Hypoketotic hypoglycaemia (Reye-like)</td>
                          <td><span className="badge bg-warning text-dark">Serious</span></td>
                        </tr>
                        <tr style={{ background: '#fff3e0' }}>
                          <td><strong>SCAD &#x2190;</strong></td><td>ACADS</td><td>C4–C6 (short)</td>
                          <td>C4 &gt;0.5 µmol/L (nonspecific)</td>
                          <td>Usually BENIGN; hypotonia if biallelic null</td>
                          <td><span className="badge bg-success">Usually Benign</span></td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Key negatives */}
          <div className="row mb-3">
            <div className="col-md-6">
              <InfoBox title="&#x2705; KEY NEGATIVES (Critical Differentials)" color={ACCENT3}>
                <ul className="mb-0 ps-3">
                  <li><strong>C8 NORMAL</strong> — KEY NEGATIVE vs MCAD (MCAD elevates C8; SCAD does NOT)</li>
                  <li><strong>C14:1 NORMAL</strong> — KEY NEGATIVE vs VLCAD (VLCAD elevates C14:1; SCAD does NOT)</li>
                  <li><strong>C3 NORMAL</strong> — KEY NEGATIVE vs PA (propionyl-CoA not involved in SCAD)</li>
                  <li><strong>No HG/SG/PPG</strong> — KEY NEGATIVE vs MCAD (no glycine conjugates of medium-chain acids)</li>
                  <li><strong>No Cardiomyopathy</strong> — KEY NEGATIVE vs VLCAD (short-chain FAO does not affect heart)</li>
                </ul>
              </InfoBox>
            </div>
            <div className="col-md-6">
              <InfoBox title="&#x26a0;&#xfe0f; Common Susceptibility Variants (NOT Pathogenic Alone)" color={ACCENT5}>
                <table className="table table-sm small mb-0">
                  <thead><tr><th>Variant</th><th>Population Freq</th><th>Clinical Significance</th></tr></thead>
                  <tbody>
                    <tr><td>625G&gt;A (p.Gly209Ser)</td><td><strong>7%</strong></td><td>Susceptibility only — NOT pathogenic alone</td></tr>
                    <tr><td>511C&gt;T (p.Arg171Trp)</td><td><strong>14%</strong></td><td>Susceptibility only — NOT pathogenic alone</td></tr>
                  </tbody>
                </table>
                <div className="mt-1 text-muted">Biallelic NULL alleles (catalytic/splice/frameshift) required for true clinical SCAD</div>
              </InfoBox>
            </div>
          </div>

          {/* Gene info */}
          <div className="row mb-3">
            <div className="col-md-6">
              <InfoBox title="Gene &amp; Protein" color={ACCENT7}>
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr><td>Gene</td><td>{ov?.gene}</td></tr>
                    <tr><td>Locus</td><td>{ov?.locus}</td></tr>
                    <tr><td>Protein</td><td>{ov?.protein}</td></tr>
                    <tr><td>Inheritance</td><td>{ov?.inheritance}</td></tr>
                    <tr><td>OMIM Gene</td><td>*{ov?.omim_gene}</td></tr>
                    <tr><td>OMIM Disease</td><td>#{ov?.omim_disease}</td></tr>
                    <tr><td>Prevalence</td><td>{ov?.prevalence}</td></tr>
                  </tbody>
                </table>
              </InfoBox>
            </div>
            <div className="col-md-6">
              <InfoBox title="NBS Marker &amp; Controversy" color={ACCENT2}>
                <div className="mb-1"><strong>Primary NBS marker:</strong> {ov?.primary_nbs_marker}</div>
                <div className="mb-1"><strong>NBS controversy:</strong> {ov?.nbs_controversy}</div>
                <div className="mb-1"><strong>Characteristic urine OA:</strong> {ov?.characteristic_urine?.join(', ')}</div>
                <div className="mb-1"><strong>Key negatives:</strong> {ov?.key_negatives?.join('; ')}</div>
                <div><strong>First-line:</strong> {ov?.first_line_treatment}</div>
              </InfoBox>
            </div>
          </div>
        </div>
      )}

      {/* ─── TAB 1: Patients & Biomarkers ─── */}
      {tab === 1 && (
        <div>
          <div className="row mb-3">
            {Object.entries(bd?.biomarkers || {}).map(([key, bm]) => (
              <div key={key} className="col-md-6 mb-3">
                <div className="card shadow-sm h-100" style={{
                  borderLeft: `4px solid ${bm.color === 'danger' ? '#b71c1c' : bm.color === 'warning' ? '#e65100' : '#1b5e20'}`
                }}>
                  <div className="card-body py-2">
                    <div className="d-flex justify-content-between align-items-start mb-1">
                      <div className="fw-bold small">{bm.label}</div>
                      <span className="badge ms-2" style={{
                        background: bm.color === 'danger' ? '#b71c1c' : bm.color === 'warning' ? '#e65100' : '#1b5e20',
                        fontSize: 10,
                      }}>{bm.direction}</span>
                    </div>
                    <div className="small text-muted mb-1">
                      <span className="me-2">Normal: <code>{bm.normal}</code></span>
                      <span>Status: <strong>{bm.status}</strong></span>
                    </div>
                    <div className="small text-muted">{bm.rationale}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Phenotype-biomarker patterns */}
          <div className="card shadow-sm mb-3">
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Phenotype–Biomarker Patterns</div>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: '#fff3e0' }}>
                    <tr>
                      <th>Phenotype</th><th>Prevalence</th><th>C4 (µmol/L)</th>
                      <th>EMA (mmol/mol Cr)</th><th>Glucose</th><th>β-OHB</th><th>Variant</th><th>Prognosis</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd?.phenotype_patterns || []).map((row, i) => (
                      <tr key={i}>
                        <td><strong>{row.phenotype}</strong></td>
                        <td>{row.prevalence}</td>
                        <td>{row.c4}</td>
                        <td>{row.ema}</td>
                        <td>{row.glucose}</td>
                        <td>{row.bohb}</td>
                        <td style={{ fontSize: 11 }}>{row.variant}</td>
                        <td>{row.prognosis}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Patient sample */}
          <div className="card shadow-sm mb-3">
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT }}>
                Patient Sample (n=10 of {ov?.n_patients}, seed {ov?.seed})
              </div>
              <div className="table-responsive">
                <table className="table table-sm table-hover small mb-0">
                  <thead style={{ background: '#fff3e0' }}>
                    <tr>
                      <th>ID</th><th>Phenotype</th><th>C4</th><th>C4/C2</th><th>C8</th>
                      <th>EMA</th><th>MSA</th><th>Glucose</th><th>β-OHB</th>
                      <th>Hypotonia</th><th>Seizures</th><th>Riboflavin?</th><th>Variant</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd?.patient_sample || []).map((p, i) => (
                      <tr key={i}>
                        <td><strong>{p.id}</strong></td>
                        <td style={{ maxWidth: 130, fontSize: 11 }}>{p.phenotype}</td>
                        <td className={p.c4 > 1.0 ? 'text-danger fw-bold' : 'text-warning'}>{p.c4}</td>
                        <td>{p.c4_c2_ratio}</td>
                        <td className="text-success">{p.c8}</td>
                        <td className={p.ema > 40 ? 'text-danger' : 'text-warning'}>{p.ema}</td>
                        <td>{p.msa}</td>
                        <td className={p.glucose < 3.0 ? 'text-danger fw-bold' : ''}>{p.glucose}</td>
                        <td>{p.bohb}</td>
                        <td>{p.hypotonia ? '✓' : '—'}</td>
                        <td className={p.seizures ? 'text-danger' : ''}>{p.seizures ? '✓' : '—'}</td>
                        <td>{p.riboflavin_tried ? (p.riboflavin_resp ? '✓ resp' : 'no resp') : '—'}</td>
                        <td style={{ fontSize: 10 }}>{p.variant}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Variant table */}
          <div className="card shadow-sm">
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT5 }}>ACADS Variants</div>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: '#f3e5f5' }}>
                    <tr><th>Variant</th><th>Cohort Freq</th><th>Domain</th><th>Phenotype</th><th>Note</th></tr>
                  </thead>
                  <tbody>
                    {(bd?.variant_table || []).map((v, i) => (
                      <tr key={i} style={{ background: v.variant.includes('625G') || v.variant.includes('511C') ? '#fff8e1' : '' }}>
                        <td><code>{v.variant}</code></td>
                        <td>{v.freq}%</td>
                        <td>{v.domain}</td>
                        <td>{v.phenotype}</td>
                        <td style={{ fontSize: 11 }}>{v.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ─── TAB 2: Treatments & Controversy ─── */}
      {tab === 2 && (
        <div>
          {/* Key differentials */}
          <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT4 }}>Key Differentials</div>
              <div className="table-responsive">
                <table className="table table-sm small mb-0">
                  <thead><tr><th>Comparison</th><th>Key Distinguishing Feature</th></tr></thead>
                  <tbody>
                    {Object.entries(bd?.key_differentials || {}).map(([k, v]) => (
                      <tr key={k}><td><strong>{k.replace(/_/g, ' ')}</strong></td><td>{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Exam pearls */}
          <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT}` }}>
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Exam Pearls</div>
              <ul className="mb-0 small text-muted ps-3">
                {(bd?.exam_pearls || []).map((p, i) => <li key={i} className="mb-1">{p}</li>)}
              </ul>
            </div>
          </div>

          {/* Treatment table */}
          <div className="card shadow-sm mb-3">
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT8 }}>Treatment &amp; Management</div>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: '#e0f2f1' }}>
                    <tr><th>Intervention</th><th>Evidence Level</th><th>Rationale</th><th>CI?</th></tr>
                  </thead>
                  <tbody>
                    {(bd?.treatment_table || []).map((t, i) => (
                      <tr key={i} style={{
                        background: t.contraindication ? '#ffebee' : ''
                      }}>
                        <td><strong>{t.intervention}</strong></td>
                        <td>{t.level}</td>
                        <td style={{ fontSize: 11 }}>{t.rationale}</td>
                        <td>
                          {t.contraindication
                            ? <span className="badge bg-danger" style={{ fontSize: 10 }}>{t.contraindication}</span>
                            : <span className="text-success">—</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* NBS Controversy Detail */}
          <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT2 }}>NBS Panel Controversy — SCAD vs Other FAO Disorders</div>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: '#fff3e0' }}>
                    <tr><th>Feature</th><th>SCAD</th><th>MCAD</th><th>VLCAD</th></tr>
                  </thead>
                  <tbody>
                    <tr><td>NBS Marker</td><td>C4 (nonspecific)</td><td>C8 (specific)</td><td>C14:1 (specific)</td></tr>
                    <tr><td>NBS Controversy</td><td className="text-danger"><strong>HIGH — most controversial</strong></td><td className="text-success">Low — well justified</td><td className="text-success">Low — well justified</td></tr>
                    <tr><td>% Asymptomatic NBS positives</td><td className="text-warning"><strong>~70%</strong></td><td>~55%</td><td>~35%</td></tr>
                    <tr><td>Common variant problem</td><td className="text-danger"><strong>Yes — 625G&gt;A, 511C&gt;T</strong></td><td>Rare</td><td>Rare</td></tr>
                    <tr><td>UK NBS Panel</td><td className="text-danger"><strong>Removed 2012</strong></td><td>Included</td><td>Included</td></tr>
                    <tr><td>Severity of true disease</td><td>Mild (hypotonia)</td><td>Severe (Reye-like, fatal)</td><td>Severe (cardiomyopathy)</td></tr>
                    <tr><td>Absolute CI in management</td><td className="text-success">None</td><td className="text-danger">Fasting + KD (ABSOLUTE)</td><td className="text-danger">Fasting + KD (ABSOLUTE)</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ─── TAB 3: Definitions ─── */}
      {tab === 3 && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <div className="card shadow-sm mb-3">
                <div className="card-body py-2">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Disease Reference</div>
                  <table className="table table-sm small mb-0">
                    <tbody>
                      {Object.entries({
                        'Disease':         def?.disease_name,
                        'Gene':            def?.gene,
                        'Locus':           def?.locus,
                        'OMIM Gene':       def?.omim_gene,
                        'OMIM Disease':    def?.omim_disease,
                        'Inheritance':     def?.inheritance,
                        'Protein':         def?.protein,
                        'Pathway':         def?.pathway,
                        'Prevalence':      def?.prevalence,
                        'NBS Marker':      def?.nbs_marker,
                      }).map(([k, v]) => (
                        <tr key={k}><td className="text-muted">{k}</td><td><strong>{v}</strong></td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT5}` }}>
                <div className="card-body py-2">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT5 }}>Common Susceptibility Variants</div>
                  {Object.entries(def?.common_variants || {}).map(([v, desc]) => (
                    <div key={v} className="mb-2">
                      <code className="fw-bold">{v}</code>
                      <div className="small text-muted mt-1">{desc}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
                <div className="card-body py-2">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT2 }}>NBS Controversy</div>
                  <div className="small text-muted">{def?.nbs_controversy}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Key exam facts */}
          <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT}` }}>
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Key Exam Facts</div>
              <ul className="mb-0 small text-muted ps-3">
                {(def?.key_exam_facts || []).map((f, i) => <li key={i} className="mb-1">{f}</li>)}
              </ul>
            </div>
          </div>

          {/* Confirmatory biomarkers */}
          <div className="card shadow-sm mb-3">
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT6 }}>Confirmatory Biomarkers</div>
              {Object.entries(def?.confirmatory_biomarkers || {}).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <span className="fw-bold text-capitalize small">{k.replace(/_/g, ' ')}: </span>
                  <span className="small text-muted">{v}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Enzymatic function */}
          <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT7}` }}>
            <div className="card-body py-2">
              <div className="fw-bold small mb-1" style={{ color: ACCENT7 }}>Enzymatic Function</div>
              <div className="small text-muted">{def?.enzymatic_function}</div>
            </div>
          </div>

          {/* Glossary */}
          <div className="card shadow-sm mb-3">
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT7 }}>Glossary</div>
              <div className="row">
                {Object.entries(def?.glossary || {}).map(([term, meaning]) => (
                  <div key={term} className="col-md-6 mb-1 small">
                    <code>{term}</code> <span className="text-muted">— {meaning}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* References */}
          <div className="card shadow-sm">
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT7 }}>References</div>
              <ul className="mb-0 small text-muted ps-3">
                {(def?.references || []).map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Back link */}
      <div className="mt-3">
        <Link href="/" className="text-decoration-none small" style={{ color: ACCENT }}>
          &larr; Back to Dashboard Home
        </Link>
      </div>
    </div>
  );
}

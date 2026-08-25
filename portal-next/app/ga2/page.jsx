'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Treatments & RR-MADD', 'Definitions'];

// GA2/MADD colour scheme — deep amber-red (pan-acylcarnitinemia; ETF conduit; riboflavin response)
const ACCENT  = '#b71c1c';   // deep red — GA2 severity / crisis
const ACCENT2 = '#e65100';   // deep orange — RR-MADD / riboflavin response hallmark
const ACCENT3 = '#1b5e20';   // deep green — KEY POSITIVES (C5, C8 elevated vs SCAD)
const ACCENT4 = '#01579b';   // deep blue — ETF conduit / electron transfer biology
const ACCENT5 = '#4a148c';   // dark purple — Type I/II neonatal severe
const ACCENT6 = '#880e4f';   // dark rose — EMA / dicarboxylic acids
const ACCENT7 = '#37474f';   // dark slate — NBS / epidemiology
const ACCENT8 = '#006064';   // dark teal — riboflavin / CoQ10 treatment

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

export default function GA2Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/ga2/overview`).then(r => r.json()),
      fetch(`${API}/api/ga2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ga2/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center text-muted">Loading GA2/MADD Dashboard&hellip;</div>;
  if (err)     return <div className="p-4 text-center text-danger">Error: {err}</div>;

  const kpis   = ov?.kpis || {};
  const phDist = ov?.phenotype_distribution || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          &#x1f9ec; GA2 / MADD Dashboard
        </h4>
        <div className="text-muted small">
          Glutaric Acidemia Type II / Multiple Acyl-CoA Dehydrogenase Deficiency &mdash;
          ETFA (15q23&ndash;q25) / ETFB (19q13.41) / ETFDH (4q32.1) &middot; AR &middot; OMIM #231680
        </div>
        <div className="text-muted small mt-1">
          <span className="badge me-1" style={{ background: ACCENT2 }}>RR-MADD: Riboflavin-Responsive MADD</span>
          <span className="badge me-1" style={{ background: ACCENT4 }}>ETF Electron Conduit</span>
          <span className="badge" style={{ background: ACCENT }}>Pan-Acylcarnitinemia NBS Pattern</span>
        </div>
      </div>

      {/* RR-MADD alert */}
      <div className="alert py-2 mb-3" style={{ background: '#fff3e0', borderLeft: `4px solid ${ACCENT2}`, fontSize: 13 }}>
        <strong>&#x1f4a1; RR-MADD (Type III):</strong> ETFDH mutations → <strong>Riboflavin 100–300 mg/day</strong> produces
        a <strong>DRAMATIC response</strong> — acylcarnitines and EMA normalise within days–weeks; myopathy resolves within months.
        This is the <strong>hallmark</strong> of riboflavin-responsive MADD and mandates riboflavin trial before muscle biopsy.
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
            <KPI label="Total Patients"         value={kpis.total_patients}     color={ACCENT}  />
            <KPI label="Type III (RR-MADD)"     value={kpis.type3_rr_madd}      color={ACCENT2} />
            <KPI label="Type II Neonatal"       value={kpis.type2_n}            color={ACCENT5} />
            <KPI label="Type I (Anomalies)"     value={kpis.type1_n}            color={ACCENT5} />
            <KPI label="Riboflavin Response"    value={kpis.riboflavin_resp_n}  color={ACCENT8} />
            <KPI label="With Seizures"          value={kpis.seizures_n}         color={ACCENT}  />
            <KPI label="Cardiomyopathy"         value={kpis.cardiomyopathy_n}   color={ACCENT}  />
            <KPI label="Congen. Anomalies"      value={kpis.congen_anomalies_n} color={ACCENT5} />
            <KPI label="Metabolic Crisis"       value={kpis.crisis_n}           color={ACCENT}  />
            <KPI label="Avg C4 (µmol/L)"        value={kpis.avg_c4_umol}        color={ACCENT}  />
            <KPI label="Avg C5 (µmol/L)"        value={kpis.avg_c5_umol}        color={ACCENT3} />
            <KPI label="Avg EMA (mmol/mol Cr)"  value={kpis.avg_ema_mmol_cr}    color={ACCENT6} />
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
                    <PctBar key={ph} label={ph.split(' — ')[0]} pct={Math.round(100 * n / (kpis.total_patients || 1))}
                      color={ph.includes('III') ? ACCENT2 : ph.includes('I ') ? ACCENT5 : ACCENT} />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* ETF conduit biology */}
          <div className="row mb-3">
            <div className="col-12">
              <div className="card shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
                <div className="card-body py-2">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT4 }}>ETF Electron Conduit — Why ALL Dehydrogenases Are Blocked</div>
                  <div className="table-responsive">
                    <table className="table table-sm table-bordered small mb-0">
                      <thead style={{ background: '#e3f2fd' }}>
                        <tr>
                          <th>Acyl-CoA DH Enzyme</th><th>Chain Length</th><th>NBS Acylcarnitine Elevated</th><th>Urine OA</th><th>Blocked in GA2?</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr><td><strong>SCAD</strong></td><td>C4–C6 (short)</td><td>C4 ↑</td><td>EMA ↑</td><td><span className="badge bg-danger">Yes</span></td></tr>
                        <tr><td><strong>MCAD</strong></td><td>C6–C12 (medium)</td><td>C8 ↑</td><td>Adipic/suberic ↑</td><td><span className="badge bg-danger">Yes</span></td></tr>
                        <tr><td><strong>VLCAD</strong></td><td>C14–C20 (very-long)</td><td>C14:1 ↑ / C16 ↑</td><td>—</td><td><span className="badge bg-danger">Yes</span></td></tr>
                        <tr><td><strong>IVD</strong></td><td>Isovaleryl-CoA</td><td>C5 ↑</td><td>Isovalerylglycine ↑</td><td><span className="badge bg-danger">Yes</span></td></tr>
                        <tr><td><strong>GCD</strong></td><td>Glutaryl-CoA</td><td>C5-DC ↑</td><td>Glutaric acid ↑</td><td><span className="badge bg-danger">Yes</span></td></tr>
                        <tr style={{ background: '#e8f5e9' }}>
                          <td colSpan={4}><strong>→ ETF (ETFA+ETFB) accepts FADH₂ from ALL above → ETFDH transfers to CoQ10 → Respiratory chain</strong></td>
                          <td><span className="badge bg-primary">Conduit</span></td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Key positives vs SCAD + Clinical types */}
          <div className="row mb-3">
            <div className="col-md-6">
              <InfoBox title="&#x2705; KEY POSITIVES (GA2 vs SCAD — Critical Differentials)" color={ACCENT3}>
                <ul className="mb-0 ps-3">
                  <li><strong>C5 ELEVATED</strong> — KEY POSITIVE vs SCAD (SCAD = C4 only; GA2 elevates C5 via IVD block)</li>
                  <li><strong>C8 ELEVATED</strong> — KEY POSITIVE vs SCAD (SCAD = C8 NORMAL; GA2 = C8 elevated via MCAD block)</li>
                  <li><strong>C14:1 ELEVATED</strong> — KEY POSITIVE vs SCAD (SCAD = C14:1 NORMAL; GA2 elevates all chain lengths)</li>
                  <li><strong>EMA ≫ SCAD</strong> — GA2 EMA &gt;300 mmol/mol Cr; SCAD EMA &lt;50 mmol/mol Cr</li>
                  <li><strong>PAN-ACYLCARNITINEMIA</strong> — all chain-length acylcarnitines elevated simultaneously</li>
                </ul>
              </InfoBox>
            </div>
            <div className="col-md-6">
              <InfoBox title="&#x26a0;&#xfe0f; Clinical Types — GA2/MADD" color={ACCENT5}>
                <table className="table table-sm small mb-0">
                  <thead><tr><th>Type</th><th>Gene</th><th>Onset</th><th>Riboflavin?</th></tr></thead>
                  <tbody>
                    <tr><td><strong>I</strong></td><td>ETFA/ETFB null</td><td>Neonatal + anomalies</td><td className="text-danger">No</td></tr>
                    <tr><td><strong>II</strong></td><td>ETFA/ETFB</td><td>Neonatal, no anomalies</td><td className="text-warning">Partial</td></tr>
                    <tr style={{ background: '#e8f5e9' }}>
                      <td><strong>III</strong></td><td>ETFDH</td><td>Infancy–adult</td><td className="text-success fw-bold">DRAMATIC</td>
                    </tr>
                  </tbody>
                </table>
              </InfoBox>
            </div>
          </div>

          {/* Gene info */}
          <div className="row mb-3">
            <div className="col-md-6">
              <InfoBox title="Gene &amp; Protein" color={ACCENT7}>
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr><td>Genes</td><td>{ov?.gene}</td></tr>
                    <tr><td>Loci</td><td>{ov?.locus}</td></tr>
                    <tr><td>Proteins</td><td style={{ fontSize: 11 }}>{ov?.protein}</td></tr>
                    <tr><td>Inheritance</td><td>{ov?.inheritance}</td></tr>
                    <tr><td>OMIM Genes</td><td>{ov?.omim_gene}</td></tr>
                    <tr><td>OMIM Disease</td><td>{ov?.omim_disease}</td></tr>
                    <tr><td>Prevalence</td><td>{ov?.prevalence}</td></tr>
                  </tbody>
                </table>
              </InfoBox>
            </div>
            <div className="col-md-6">
              <InfoBox title="NBS Pattern &amp; RR-MADD Key" color={ACCENT2}>
                <div className="mb-1"><strong>Primary NBS:</strong> <span style={{ color: ACCENT }}>{ov?.primary_nbs_marker}</span></div>
                <div className="mb-1"><strong>RR-MADD hallmark:</strong> {ov?.rr_madd_key}</div>
                <div className="mb-1"><strong>Urine OA hallmarks:</strong></div>
                <ul className="mb-1 ps-3" style={{ fontSize: 12 }}>
                  {(ov?.urine_oa_hallmarks || []).map((u, i) => <li key={i}>{u}</li>)}
                </ul>
                <div><strong>First-line:</strong> {ov?.first_line_treatment}</div>
              </InfoBox>
            </div>
          </div>

          {/* Absolute CIs */}
          <div className="alert alert-danger py-2 mb-3" style={{ fontSize: 13 }}>
            <strong>&#x26d4; ABSOLUTE CONTRAINDICATIONS (ALL Types):</strong>{' '}
            {(ov?.absolute_ci || []).join(' · ')}
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
              <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Phenotype–Biomarker Patterns by Clinical Type</div>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: '#ffebee' }}>
                    <tr>
                      <th>Type</th><th>Prevalence</th><th>Onset</th><th>C4</th><th>C5</th><th>C8</th><th>C16</th>
                      <th>EMA (mmol/mol)</th><th>Glucose</th><th>Riboflavin Resp</th><th>Gene</th><th>Prognosis</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd?.phenotype_patterns || []).map((row, i) => (
                      <tr key={i} style={{ background: row.type.includes('RR-MADD') ? '#e8f5e9' : row.type.includes('I neonatal') ? '#fce4ec' : '' }}>
                        <td><strong>{row.type}</strong></td>
                        <td>{row.prevalence}</td>
                        <td>{row.onset}</td>
                        <td>{row.c4}</td>
                        <td className="fw-bold text-success">{row.c5}</td>
                        <td className="fw-bold text-danger">{row.c8}</td>
                        <td>{row.c16}</td>
                        <td className={row.ema.startsWith('8') ? 'text-success' : 'text-danger fw-bold'}>{row.ema}</td>
                        <td className="text-danger fw-bold">{row.glucose}</td>
                        <td className={row.riboflavin_response.includes('DRAMATIC') ? 'text-success fw-bold' : row.riboflavin_response.includes('None') ? 'text-danger' : 'text-warning'}>
                          {row.riboflavin_response}
                        </td>
                        <td><code>{row.gene}</code></td>
                        <td style={{ fontSize: 11 }}>{row.prognosis}</td>
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
                  <thead style={{ background: '#ffebee' }}>
                    <tr>
                      <th>ID</th><th>Type</th><th>Gene</th><th>C4</th><th>C5</th><th>C8</th><th>C16</th>
                      <th>EMA</th><th>Glucose</th><th>Cardio</th><th>Seizures</th><th>Crisis</th><th>Ribo?</th><th>Variant</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd?.patient_sample || []).map((p, i) => (
                      <tr key={i} style={{ background: p.rr_madd ? '#f1f8e9' : p.congen_anomalies ? '#fce4ec' : '' }}>
                        <td><strong>{p.id}</strong></td>
                        <td style={{ maxWidth: 100, fontSize: 10 }}>{p.phenotype.split(' ')[0]} {p.phenotype.split(' ')[1]}</td>
                        <td><code>{p.gene}</code></td>
                        <td className={p.c4_umol > 3.0 ? 'text-danger fw-bold' : 'text-warning'}>{p.c4_umol}</td>
                        <td className="text-danger fw-bold">{p.c5_umol}</td>
                        <td className="text-danger fw-bold">{p.c8_umol}</td>
                        <td className={p.c16_umol > 10 ? 'text-danger' : 'text-warning'}>{p.c16_umol}</td>
                        <td className={p.ema_mmol_cr > 300 ? 'text-danger fw-bold' : 'text-warning'}>{p.ema_mmol_cr}</td>
                        <td className={p.glucose_mmol < 2.0 ? 'text-danger fw-bold' : ''}>{p.glucose_mmol}</td>
                        <td>{p.cardiomyopathy ? '✓' : '—'}</td>
                        <td className={p.seizures ? 'text-danger' : ''}>{p.seizures ? '✓' : '—'}</td>
                        <td>{p.metabolic_crisis ? '✓' : '—'}</td>
                        <td className={p.riboflavin_resp ? 'text-success fw-bold' : ''}>{p.riboflavin_resp ? '✓ resp' : p.rr_madd ? 'trial' : '—'}</td>
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
              <div className="fw-bold small mb-2" style={{ color: ACCENT5 }}>ETFA / ETFB / ETFDH Variants</div>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: '#f3e5f5' }}>
                    <tr><th>Variant</th><th>Gene</th><th>Cohort Freq</th><th>Domain</th><th>Type</th><th>Phenotype</th><th>Note</th></tr>
                  </thead>
                  <tbody>
                    {(bd?.variant_table || []).map((v, i) => (
                      <tr key={i} style={{ background: v.gene === 'ETFDH' ? '#e8f5e9' : '#fce4ec' }}>
                        <td><code>{v.variant}</code></td>
                        <td><span className="badge" style={{ background: v.gene === 'ETFDH' ? ACCENT2 : ACCENT5, fontSize: 10 }}>{v.gene}</span></td>
                        <td>{v.freq}%</td>
                        <td>{v.domain}</td>
                        <td>{v.type}</td>
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

      {/* ─── TAB 2: Treatments & RR-MADD ─── */}
      {tab === 2 && (
        <div>
          {/* RR-MADD riboflavin highlight */}
          <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT2 }}>&#x1f3c6; RR-MADD Riboflavin Response — Hallmark of Type III (ETFDH)</div>
              <div className="small text-muted">
                <strong>Mechanism:</strong> ETFDH missense mutations destabilise FAD binding. Riboflavin (B2) → FAD supplementation restores
                FAD binding pocket geometry in residual protein → near-normal electron transfer from ETF to CoQ10.
                Response is DRAMATIC: C4/C5/C8 acylcarnitines and EMA normalise within <strong>days–weeks</strong>;
                proximal myopathy reverses within <strong>months</strong>. The p.Arg191Trp East Asian founder variant
                (~50% of Asian RR-MADD) shows consistently excellent response.
                <strong className="text-danger"> Riboflavin trial MANDATORY before muscle biopsy</strong> in any adult
                with proximal myopathy + elevated acylcarnitines on metabolic panel.
              </div>
            </div>
          </div>

          {/* Key differentials */}
          <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT4 }}>Key Differentials</div>
              <div className="table-responsive">
                <table className="table table-sm small mb-0">
                  <thead><tr><th>Comparison</th><th>Key Distinguishing Feature</th></tr></thead>
                  <tbody>
                    {Object.entries(bd?.key_differentials || {}).map(([k, v]) => (
                      <tr key={k}><td><strong>{k.replace(/_/g, ' ')}</strong></td><td style={{ fontSize: 12 }}>{v}</td></tr>
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
                      <tr key={i} style={{ background: t.contraindication ? '#ffebee' : t.intervention.includes('Riboflavin') ? '#e8f5e9' : '' }}>
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

          {/* FAO-disorder CI comparison */}
          <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT}` }}>
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Contraindication Comparison — GA2 vs Other FAO Disorders</div>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: '#ffebee' }}>
                    <tr><th>Feature</th><th>GA2 (ETFA/B/DH)</th><th>SCAD (ACADS)</th><th>MCAD (ACADM)</th><th>VLCAD (ACADVL)</th><th>LCHAD (HADHA)</th></tr>
                  </thead>
                  <tbody>
                    <tr><td>VPA</td>
                      <td className="text-danger fw-bold">ABSOLUTE CI</td>
                      <td className="text-warning">High risk (sympt.)</td>
                      <td className="text-warning">High risk</td>
                      <td className="text-warning">High risk</td>
                      <td className="text-warning">High risk</td>
                    </tr>
                    <tr><td>Ketogenic Diet</td>
                      <td className="text-danger fw-bold">ABSOLUTE CI</td>
                      <td className="text-warning">Relative CI</td>
                      <td className="text-danger fw-bold">ABSOLUTE CI</td>
                      <td className="text-danger fw-bold">ABSOLUTE CI</td>
                      <td className="text-danger fw-bold">ABSOLUTE CI</td>
                    </tr>
                    <tr><td>MCT Oil</td>
                      <td className="text-danger">INEFFECTIVE (still needs ETF)</td>
                      <td>Not indicated</td>
                      <td>Not indicated</td>
                      <td className="text-success fw-bold">THERAPEUTIC (bypasses VLCAD)</td>
                      <td className="text-success fw-bold">THERAPEUTIC (bypasses MTP)</td>
                    </tr>
                    <tr><td>Riboflavin</td>
                      <td className="text-success fw-bold">Level A (Type III), Level B (Type II ETFA)</td>
                      <td className="text-warning">Level B (modest)</td>
                      <td className="text-danger">NOT indicated</td>
                      <td className="text-danger">NOT indicated</td>
                      <td className="text-danger">NOT indicated</td>
                    </tr>
                    <tr><td>Fasting</td>
                      <td className="text-danger fw-bold">ABSOLUTE CI (all types)</td>
                      <td className="text-warning">Level B avoid</td>
                      <td className="text-danger fw-bold">ABSOLUTE CI</td>
                      <td className="text-danger fw-bold">ABSOLUTE CI</td>
                      <td className="text-danger fw-bold">ABSOLUTE CI</td>
                    </tr>
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
                        'Genes':           def?.gene,
                        'Loci':            def?.locus,
                        'OMIM Genes':      def?.omim_gene,
                        'OMIM Disease':    def?.omim_disease,
                        'Inheritance':     def?.inheritance,
                        'Pathway':         def?.pathway,
                        'Prevalence':      def?.prevalence,
                        'NBS Marker':      def?.nbs_marker,
                      }).map(([k, v]) => (
                        <tr key={k}><td className="text-muted">{k}</td><td><strong style={{ fontSize: 12 }}>{v}</strong></td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT5}` }}>
                <div className="card-body py-2">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT5 }}>Common Variants</div>
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
                  <div className="fw-bold small mb-2" style={{ color: ACCENT2 }}>RR-MADD Definition</div>
                  <div className="small text-muted">{def?.rr_madd_definition}</div>
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

          {/* Enzymatic function */}
          <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
            <div className="card-body py-2">
              <div className="fw-bold small mb-1" style={{ color: ACCENT4 }}>Enzymatic Function (ETF/ETFDH System)</div>
              <div className="small text-muted">{def?.enzymatic_function}</div>
            </div>
          </div>

          {/* Metabolic block */}
          <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT}` }}>
            <div className="card-body py-2">
              <div className="fw-bold small mb-1" style={{ color: ACCENT }}>Metabolic Block</div>
              <div className="small text-muted">{def?.metabolic_block}</div>
            </div>
          </div>

          {/* Clinical types */}
          <div className="card shadow-sm mb-3">
            <div className="card-body py-2">
              <div className="fw-bold small mb-2" style={{ color: ACCENT5 }}>Clinical Types</div>
              {Object.entries(def?.clinical_types || {}).map(([type, desc]) => (
                <div key={type} className="mb-2">
                  <span className="fw-bold small">{type.replace(/_/g, ' ')}: </span>
                  <span className="small text-muted">{desc}</span>
                </div>
              ))}
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

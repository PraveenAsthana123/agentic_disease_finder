'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// VLCAD / ACADVL colour scheme — deep purple-maroon (cardiac / rhabdo severity)
const ACCENT  = '#4a0072';   // deep purple — VLCAD / primary
const ACCENT2 = '#880e4f';   // dark rose — C14:1 primary NBS marker
const ACCENT3 = '#b71c1c';   // deep red — CARDIOMYOPATHY / FASTING ABSOLUTE CI
const ACCENT4 = '#1b5e20';   // deep green — KEY NEGATIVES (C8 normal)
const ACCENT5 = '#bf360c';   // deep orange — RHABDOMYOLYSIS / CK elevation
const ACCENT6 = '#0d47a1';   // deep blue — MCT therapeutic (beneficial)
const ACCENT7 = '#37474f';   // dark slate — epidemiology
const ACCENT8 = '#4e342e';   // dark brown — founder variant p.Val283Ala

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

export default function VLCADPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/vlcad/overview`).then(r => r.json()),
      fetch(`${API}/api/vlcad/breakdown`).then(r => r.json()),
      fetch(`${API}/api/vlcad/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center text-muted">Loading VLCAD Dashboard…</div>;
  if (err)     return <div className="p-4 text-center text-danger">Error: {err}</div>;

  const kpis   = ov?.kpis || {};
  const phDist = ov?.phenotype_distribution || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          &#x1f9ec; VLCAD Deficiency Dashboard
        </h4>
        <div className="text-muted small">
          Very Long Chain Acyl-CoA Dehydrogenase Deficiency &mdash; ACADVL &middot; 17p13.1 &middot; AR &middot; OMIM #201475
        </div>
        <div className="text-muted small">
          C14:1 PRIMARY NBS MARKER &middot; Cardiomyopathy hallmark &middot; MCT oil therapeutic &middot; KD absolute CI
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          <Alert
            text="⚠ FASTING: ABSOLUTE CONTRAINDICATION — triggers long-chain FAO demand on blocked VLCAD → hypoketotic crisis + cardiac arrhythmia. IV Glucose 10% + MCT oil during any illness."
            variant="danger"
          />
          <Alert
            text="⚠ KETOGENIC DIET: ABSOLUTE CI — long-chain fat is the BLOCKED substrate in VLCAD; KD floods the deficient enzyme → catastrophic cardiomyopathy crisis. MCT oil is THERAPEUTIC (bypasses VLCAD via intact MCAD). VPA: HIGH RISK."
            variant="warning"
          />
          <Alert
            text="✅ MCT OIL IS THERAPEUTIC — C8/C10 medium-chain triglycerides bypass VLCAD completely; processed by intact MCAD. Cardiomyopathy is REVERSIBLE with MCT diet."
            variant="success"
          />

          {/* KPI strip */}
          <div className="row mb-3">
            <KPI label="Total Patients"      value={kpis.total_patients}      color={ACCENT}  />
            <KPI label="Severe Infantile"    value={kpis.severe_infantile_n}  color={ACCENT3} />
            <KPI label="Cardiomyopathy"      value={kpis.cardiomyopathy_n}    color={ACCENT3} />
            <KPI label="Rhabdomyolysis"      value={kpis.rhabdomyolysis_n}    color={ACCENT5} />
            <KPI label="Avg C14:1 (µmol/L)" value={kpis.avg_c14_1_umol}      color={ACCENT2} />
            <KPI label="Avg CK (U/L)"       value={kpis.avg_ck_u_l}          color={ACCENT5} />
          </div>

          {/* Second KPI row */}
          <div className="row mb-3">
            <KPI label="Episodic/Hepatic"    value={kpis.episodic_n}          color={ACCENT7} />
            <KPI label="Mild/Myopathic"      value={kpis.mild_myopathic_n}    color={ACCENT6} />
            <KPI label="Seizures"            value={kpis.seizures_n}          color={ACCENT}  />
            <KPI label="Troponin Elevated"   value={kpis.troponin_elevated_n} color={ACCENT3} />
            <KPI label="Avg C0 (µmol/L)"    value={kpis.avg_c0_umol}         color={ACCENT7} />
          </div>

          {/* Phenotype distribution */}
          <div className="row mb-3">
            <div className="col-md-5">
              <div className="card shadow-sm p-3">
                <div className="fw-bold mb-2" style={{ color: ACCENT }}>Phenotype Distribution</div>
                {Object.entries(phDist).map(([ph, n]) => (
                  <PctBar
                    key={ph}
                    label={ph}
                    pct={Math.round(100 * n / (kpis.total_patients || 40))}
                    color={
                      ph.includes('Severe')  ? ACCENT3 :
                      ph.includes('Episodic') ? ACCENT7 :
                      ACCENT6
                    }
                  />
                ))}
              </div>
            </div>
            <div className="col-md-7">
              <div className="card shadow-sm p-3 h-100">
                <div className="fw-bold mb-2" style={{ color: ACCENT }}>Gene &amp; Enzyme</div>
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>{ov?.gene}</td></tr>
                    <tr><td className="fw-bold">Locus</td><td>{ov?.locus}</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*{ov?.omim_gene}</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#{ov?.omim_disease}</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>{ov?.protein}</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>{ov?.prevalence}</td></tr>
                    <tr><td className="fw-bold">Primary NBS</td><td style={{ color: ACCENT2 }}>{ov?.primary_nbs_marker}</td></tr>
                    <tr><td className="fw-bold">Discriminating Ratio</td><td style={{ color: ACCENT2 }}>{ov?.discriminating_ratio}</td></tr>
                    <tr><td className="fw-bold">Hallmark Feature</td><td style={{ color: ACCENT3 }}>{ov?.hallmark_feature}</td></tr>
                    <tr><td className="fw-bold">MCT Oil</td><td style={{ color: ACCENT6 }}>{ov?.therapeutic_fat}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Key negatives vs MCAD */}
          <div className="card shadow-sm p-3 mb-3">
            <div className="fw-bold mb-2" style={{ color: ACCENT4 }}>KEY NEGATIVES (VLCAD differentiation from MCAD)</div>
            <div className="row">
              {Object.entries(ov?.key_negatives || {}).map(([k, v]) => (
                <div key={k} className="col-md-6 mb-2">
                  <InfoBox title={k.replace(/_/g,' ').toUpperCase()} color={ACCENT4}>{v}</InfoBox>
                </div>
              ))}
            </div>
          </div>

          {/* Clinical summary */}
          <InfoBox title="Clinical Summary" color={ACCENT}>
            {ov?.clinical_summary}
          </InfoBox>

          {/* Top variant */}
          <InfoBox title={`Most Common Allele: ${ov?.top_variant} (~${ov?.top_variant_pct}% of cohort)`} color={ACCENT8}>
            p.Val283Ala (c.848T>C) — most common allele in mild/adult myopathic phenotype.
            Retains ~25% residual VLCAD activity; substrate-binding domain; temperature-sensitive.
            Adult-onset exercise-induced rhabdomyolysis; often detected only by NBS or after rhabdomyolysis episode.
          </InfoBox>
        </div>
      )}

      {/* ── TAB 1: PATIENTS & BIOMARKERS ── */}
      {tab === 1 && (
        <div>
          <Alert
            text="BIOMARKER PROFILE: C14:1 ↑↑↑ (PRIMARY NBS) + C14 ↑↑ + C14:2 ↑ + C16 ↑ + C18:1 ↑ + C0 ↓ + CK ↑↑↑ (rhabdomyolysis) + Troponin ↑ (cardiac) + HYPOketotic hypoglycaemia. C8 NORMAL (KEY NEGATIVE vs MCAD)."
            variant="info"
          />

          {/* Biomarker reference table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>Biomarker Reference Panel</div>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Biomarker</th><th>Normal</th><th>VLCAD Status</th><th>Direction</th><th>Clinical Note</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(bd?.biomarkers || {}).map(([k, bm]) => (
                    <tr key={k}>
                      <td className="fw-bold">{bm.label}</td>
                      <td className="text-muted">{bm.normal}</td>
                      <td>
                        <span className={`badge bg-${bm.color}`}>{bm.status?.substring(0, 50)}</span>
                      </td>
                      <td className="fw-bold" style={{
                        color: bm.direction?.startsWith('↑') ? '#b71c1c' :
                               bm.direction?.startsWith('↓') ? '#e65100' :
                               '#1b5e20'
                      }}>{bm.direction}</td>
                      <td style={{ maxWidth: 350 }}>{bm.rationale?.substring(0, 180)}…</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Patient sample table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>
              Patient Sample (n={bd?.patients?.length || 0} shown of {bd?.n_patients || 40})
            </div>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead className="table-light">
                  <tr>
                    <th>ID</th><th>Phenotype</th><th>Onset (mo)</th>
                    <th>C14:1</th><th>C14:1/C2</th><th>C16</th><th>C8 (↓ KEY NEG)</th>
                    <th>CK (U/L)</th><th>Cardiac</th><th>Glucose</th><th>Variant</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.patients || []).map((p, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{p.id}</td>
                      <td>
                        <span className={`badge ${
                          p.phenotype?.includes('Severe') ? 'bg-danger' :
                          p.phenotype?.includes('Episodic') ? 'bg-warning text-dark' :
                          'bg-success'
                        }`}>{p.phenotype?.split('(')[0].trim()}</span>
                      </td>
                      <td>{p.onset_mo}</td>
                      <td style={{ color: '#880e4f', fontWeight: 'bold' }}>{p.c14_1}</td>
                      <td style={{ color: p.c14_1_c2 > 0.07 ? '#b71c1c' : '#1b5e20' }}>{p.c14_1_c2?.toFixed(3)}</td>
                      <td>{p.c16}</td>
                      <td style={{ color: '#1b5e20' }}>{p.c8} ✓</td>
                      <td style={{ color: p.ck > 5000 ? '#b71c1c' : 'inherit', fontWeight: p.ck > 5000 ? 'bold' : 'normal' }}>
                        {p.ck?.toLocaleString()}
                      </td>
                      <td style={{ color: p.cardiomyopathy ? '#b71c1c' : '#1b5e20' }}>
                        {p.cardiomyopathy ? '❤️ Yes' : '—'}
                      </td>
                      <td style={{ color: p.glucose < 3.0 ? '#b71c1c' : 'inherit' }}>{p.glucose}</td>
                      <td className="text-muted" style={{ maxWidth: 160, fontSize: 11 }}>
                        {p.variant?.substring(0, 30)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Variants table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>ACADVL Variants</div>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead className="table-light">
                  <tr><th>Variant</th><th>Frequency</th><th>Domain</th><th>Phenotype</th><th>Note</th></tr>
                </thead>
                <tbody>
                  {(bd?.variants || []).map((v, i) => (
                    <tr key={i}>
                      <td className="fw-bold" style={{ color: ACCENT8 }}>{v.variant}</td>
                      <td>{v.freq}%</td>
                      <td>{v.domain}</td>
                      <td>{v.phenotype}</td>
                      <td className="text-muted">{v.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: SEIZURES & TREATMENTS ── */}
      {tab === 2 && (
        <div>
          <Alert
            text="MCT OIL IS THE CORNERSTONE — C8/C10 bypasses VLCAD entirely via intact MCAD. Cardiomyopathy is REVERSIBLE with MCT diet. KD is ABSOLUTE CI. Rhabdomyolysis: IV fluids + glucose + rest."
            variant="success"
          />

          {/* Phenotype profiles */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>Phenotype Profiles</div>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead className="table-light">
                  <tr><th>Phenotype</th><th>Prevalence</th><th>Onset</th><th>Features</th><th>Key Point</th><th>Prognosis</th></tr>
                </thead>
                <tbody>
                  {Object.entries(bd?.phenotype_profiles || {}).map(([ph, info], i) => (
                    <tr key={i}>
                      <td className="fw-bold" style={{ color:
                        ph.includes('Severe') ? ACCENT3 :
                        ph.includes('Episodic') ? ACCENT7 :
                        ACCENT6
                      }}>{ph}</td>
                      <td>{info.prevalence}</td>
                      <td>{info.onset}</td>
                      <td style={{ maxWidth: 200 }}>{info.features}</td>
                      <td style={{ maxWidth: 200, color: ACCENT6 }}>{info.key_point}</td>
                      <td>{info.prognosis}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Treatment reference */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>Treatment Reference</div>
            <div className="row g-2 p-2">
              {Object.entries(bd?.treatments || {}).map(([k, tx]) => (
                <div key={k} className="col-md-6">
                  <div className="card shadow-sm p-2 h-100" style={{
                    borderLeft: `4px solid ${
                      k === 'kd_absolute_ci' || k === 'vpa_avoid' ? ACCENT3 :
                      k === 'mct_oil' ? ACCENT6 :
                      ACCENT
                    }`
                  }}>
                    <div className="fw-bold small mb-1" style={{
                      color: k === 'kd_absolute_ci' || k === 'vpa_avoid' ? ACCENT3 :
                             k === 'mct_oil' ? ACCENT6 :
                             ACCENT
                    }}>{tx.label}</div>
                    <div className="text-muted small" style={{ fontSize: 11 }}>
                      <span className="badge bg-secondary me-1">{tx.level?.substring(0, 30)}</span>
                      {tx.rationale?.substring(0, 220)}…
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* VLCAD vs MCAD comparison */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>
              VLCAD vs MCAD — Critical Exam Differentiators
            </div>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead className="table-light">
                  <tr><th>Feature</th><th style={{ color: ACCENT }}>VLCAD (ACADVL)</th><th style={{ color: '#01579b' }}>MCAD (ACADM)</th></tr>
                </thead>
                <tbody>
                  <tr><td className="fw-bold">Primary NBS Marker</td>
                    <td style={{ color: ACCENT2 }}>C14:1 (Tetradecenoylcarnitine) ↑↑↑</td>
                    <td style={{ color: '#01579b' }}>C8 (Octanoylcarnitine) ↑↑↑</td></tr>
                  <tr><td className="fw-bold">C8 Status</td>
                    <td style={{ color: '#1b5e20' }}>NORMAL ✓ (KEY NEGATIVE)</td>
                    <td style={{ color: '#b71c1c' }}>ELEVATED ↑↑↑ (PRIMARY MARKER)</td></tr>
                  <tr><td className="fw-bold">Cardiomyopathy</td>
                    <td style={{ color: ACCENT3 }}>HALLMARK — HCM/DCM; ± arrhythmia</td>
                    <td style={{ color: '#1b5e20' }}>ABSENT (major differentiator)</td></tr>
                  <tr><td className="fw-bold">Rhabdomyolysis</td>
                    <td style={{ color: ACCENT5 }}>YES — CK 10,000–100,000 U/L; exercise-induced</td>
                    <td style={{ color: '#1b5e20' }}>NO (CK normal or mildly elevated)</td></tr>
                  <tr><td className="fw-bold">Pathognomonic Urine</td>
                    <td style={{ color: '#1b5e20' }}>None (3-OH-dicarboxylic acids non-specific)</td>
                    <td style={{ color: '#880e4f' }}>HG + SG + PPG (pathognomonic glycine conjugates)</td></tr>
                  <tr><td className="fw-bold">MCT Oil</td>
                    <td style={{ color: ACCENT6 }}>THERAPEUTIC — bypasses VLCAD via intact MCAD</td>
                    <td style={{ color: '#1b5e20' }}>Neutral (not specifically needed)</td></tr>
                  <tr><td className="fw-bold">Ketogenic Diet</td>
                    <td style={{ color: ACCENT3 }}>ABSOLUTE CI — floods blocked VLCAD (long-chain fat)</td>
                    <td style={{ color: ACCENT3 }}>ABSOLUTE CI — fasting intervals trigger crisis</td></tr>
                  <tr><td className="fw-bold">Chain Length</td>
                    <td>C14–C20 (very long chain)</td>
                    <td>C6–C12 (medium chain)</td></tr>
                  <tr><td className="fw-bold">Structure</td>
                    <td>Homodimer; mitochondrial INNER MEMBRANE; 655 aa</td>
                    <td>Homotetramer; mitochondrial MATRIX; 421 aa</td></tr>
                  <tr><td className="fw-bold">Prevalence</td>
                    <td>~1:40,000–80,000</td>
                    <td>~1:10,000–15,000 (most common FAO disorder)</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: DEFINITIONS ── */}
      {tab === 3 && (
        <div>
          <div className="row mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm p-3 h-100">
                <div className="fw-bold mb-2" style={{ color: ACCENT }}>Gene &amp; Protein</div>
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>{def?.gene}</td></tr>
                    <tr><td className="fw-bold">Locus</td><td>{def?.locus}</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>{def?.omim_gene}</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>{def?.omim_disease}</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>{def?.inheritance}</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>{def?.prevalence}</td></tr>
                    <tr><td className="fw-bold">Protein Class</td><td>{def?.protein_class}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm p-3 h-100">
                <div className="fw-bold mb-2" style={{ color: ACCENT2 }}>NBS Markers</div>
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr>
                      <td className="fw-bold">Primary</td>
                      <td style={{ color: ACCENT2 }}>{def?.nbs_markers?.primary}</td>
                    </tr>
                    <tr>
                      <td className="fw-bold">Secondary</td>
                      <td>{(def?.nbs_markers?.secondary || []).join(', ')}</td>
                    </tr>
                    <tr>
                      <td className="fw-bold">Best Ratio</td>
                      <td style={{ color: ACCENT2 }}>{def?.nbs_markers?.best_ratio}</td>
                    </tr>
                    <tr>
                      <td className="fw-bold" style={{ color: ACCENT4 }}>KEY NEGATIVE</td>
                      <td style={{ color: ACCENT4 }}>{def?.nbs_markers?.key_negative}</td>
                    </tr>
                  </tbody>
                </table>
                <div className="fw-bold small mt-2 mb-1" style={{ color: ACCENT3 }}>Clinical Triad</div>
                {(def?.clinical_triad || []).map((t, i) => (
                  <div key={i} className="small text-muted">• {t}</div>
                ))}
              </div>
            </div>
          </div>

          <InfoBox title="Enzyme Function" color={ACCENT}>
            {def?.enzyme_function}
          </InfoBox>

          <InfoBox title="Pathomechanism" color={ACCENT3}>
            {def?.pathomechanism}
          </InfoBox>

          {/* Treatment overview */}
          <div className="card shadow-sm p-3 mb-3">
            <div className="fw-bold mb-2" style={{ color: ACCENT }}>Treatment Overview</div>
            <table className="table table-sm small mb-0">
              <tbody>
                <tr>
                  <td className="fw-bold" style={{ color: ACCENT6 }}>First Line</td>
                  <td>{def?.treatment_overview?.first_line}</td>
                </tr>
                <tr>
                  <td className="fw-bold" style={{ color: ACCENT3 }}>Emergency</td>
                  <td>{def?.treatment_overview?.emergency}</td>
                </tr>
                <tr>
                  <td className="fw-bold" style={{ color: ACCENT3 }}>Absolute CI</td>
                  <td style={{ color: ACCENT3 }}>{(def?.treatment_overview?.absolute_ci || []).join(' + ')}</td>
                </tr>
                <tr>
                  <td className="fw-bold" style={{ color: '#e65100' }}>High Risk Drugs</td>
                  <td style={{ color: '#e65100' }}>{(def?.treatment_overview?.high_risk || []).join(', ')}</td>
                </tr>
                <tr>
                  <td className="fw-bold" style={{ color: ACCENT6 }}>MCT Rationale</td>
                  <td style={{ color: ACCENT6 }}>{def?.treatment_overview?.mct_rationale}</td>
                </tr>
                <tr>
                  <td className="fw-bold">Monitoring</td>
                  <td>{def?.treatment_overview?.monitoring}</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Key differentials */}
          <div className="card shadow-sm p-3 mb-3">
            <div className="fw-bold mb-2" style={{ color: ACCENT }}>Key Differentials</div>
            {Object.entries(def?.key_differentials || {}).map(([k, v]) => (
              <InfoBox key={k} title={k.replace(/_/g,' ')} color={ACCENT7}>{v}</InfoBox>
            ))}
          </div>

          {/* Exam pearls */}
          <div className="card shadow-sm p-3 mb-3">
            <div className="fw-bold mb-2" style={{ color: ACCENT }}>Key Exam Pearls</div>
            {(def?.key_exam_pearls || []).map((pearl, i) => (
              <div key={i} className="mb-1 small" style={{ color: i < 3 ? ACCENT3 : 'inherit' }}>
                <strong>{i + 1}.</strong> {pearl}
              </div>
            ))}
          </div>

          {/* Related disorders */}
          <div className="card shadow-sm p-3 mb-3">
            <div className="fw-bold mb-2" style={{ color: ACCENT }}>Related FAO Disorders</div>
            {(def?.related_disorders || []).map((d, i) => (
              <div key={i} className="small text-muted mb-1">• {d}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

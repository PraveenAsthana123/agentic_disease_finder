'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1b5e20';   // deep green — GNMT / SAM safety valve / liver
const ACCENT2 = '#b71c1c';   // deep red — SAM massively elevated / severe
const ACCENT3 = '#4a148c';   // deep purple — SAM/SAH ratio pathognomonic
const ACCENT4 = '#1565c0';   // blue — Level A treatment
const ACCENT5 = '#37474f';   // slate — KEY NEGATIVES / normal values
const ACCENT6 = '#006064';   // teal — methionine restriction / normal tHcy
const ACCENT7 = '#e65100';   // deep orange — liver disease / hepatotoxicity
const ACCENT8 = '#880e4f';   // deep pink — absolute contraindications

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

export default function GNMTPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [br, setBr]       = useState(null);
  const [df, setDf]       = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/gnmt/overview`).then(r => r.json()),
      fetch(`${API}/api/gnmt/breakdown`).then(r => r.json()),
      fetch(`${API}/api/gnmt/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBr(b); setDf(d); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <div className="alert alert-danger m-4">{error}</div>;
  if (!ov)   return <div className="text-center mt-5"><div className="spinner-border" /></div>;

  const kpi    = ov.kpis || {};
  const kpiPct = br?.kpi_pcts || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT3} 100%)` }}>
        <div className="d-flex justify-content-between align-items-start flex-wrap gap-2">
          <div>
            <h4 className="mb-1 fw-bold">🧬 {ov.title}</h4>
            <div style={{ fontSize: 13, opacity: 0.9 }}>{ov.subtitle}</div>
          </div>
          <div className="text-end" style={{ fontSize: 12, opacity: 0.85 }}>
            <div>{ov.chromosome} · {ov.inheritance}</div>
            <div>{ov.omim_gene} · {ov.omim_disease}</div>
            <div>N = {ov.cohort_n} patients · Seed 107</div>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert
        variant="danger"
        text="⛔ SAMe / SAM Supplements — ABSOLUTE CONTRAINDICATION: directly worsens SAM toxicity (no GNMT safety valve). OPPOSITE of MAT1A where SAM is Level A treatment."
      />
      <Alert
        variant="danger"
        text="⛔ Betaine (TMG) — ABSOLUTE CONTRAINDICATION: BHMT drives more methionine → more SAM in an already-overloaded system. No benefit (tHcy is NORMAL)."
      />
      <Alert
        variant="warning"
        text="⚠ SAM/SAH RATIO MARKEDLY ELEVATED (>10–25) — PATHOGNOMONIC for GNMT. OPPOSITE of AHCY (ratio severely LOW <0.5). Sarcosine ABSENT (GNMT product not made)."
      />

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link${tab === i ? ' active' : ''}`}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div className="row g-2 mb-3">
            <KPI label="Avg Methionine (µmol/L)" value={kpi.avg_methionine_umol_l} color={ACCENT2} />
            <KPI label="Avg SAM (µmol/L)" value={kpi.avg_sam_umol_l} color={ACCENT3} />
            <KPI label="Avg SAH (µmol/L)" value={kpi.avg_sah_umol_l} color={ACCENT5} />
            <KPI label="Avg SAM/SAH Ratio" value={kpi.avg_sam_sah_ratio} color={ACCENT8} />
            <KPI label="Avg tHcy (µmol/L)" value={kpi.avg_homocysteine_umol_l} color={ACCENT6} />
            <KPI label="Avg AST (U/L)" value={kpi.avg_ast_u_l} color={ACCENT7} />
            <KPI label="Liver Disease %" value={`${kpi.pct_liver_disease}%`} color={ACCENT7} />
            <KPI label="Seizures %" value={`${kpi.pct_seizures}%`} color={ACCENT2} />
            <KPI label="NBS Detected %" value={`${kpi.pct_nbs_detected}%`} color={ACCENT4} />
            <KPI label="Sarcosine Absent" value="100%" color={ACCENT3} />
            <KPI label="Myopathy" value="0% ✓" color={ACCENT5} />
            <KPI label="tHcy NORMAL" value="100% ✓" color={ACCENT6} />
          </div>

          {/* GNMT vs HHcy Comparison Table */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>
              GNMT vs All Inherited HHcy / Hypermethioninemia Disorders — Biomarker Fingerprint
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
                  <thead className="table-dark">
                    <tr>
                      <th>Feature</th>
                      <th style={{ background: '#1b5e20', color: '#fff' }}>GNMT ✦</th>
                      <th>MAT1A</th>
                      <th>AHCY</th>
                      <th>CBS</th>
                      <th>MTHFR</th>
                      <th>cblE/cblG</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="fw-semibold">Methionine</td>
                      <td style={{ background: '#fff9c4' }}>↑ HIGH (40–500)</td>
                      <td>↑↑↑ EXTREME (200–2000+)</td>
                      <td>↑↑ HIGH (200–600)</td>
                      <td>↑↑ HIGH (60–500)</td>
                      <td>↓ LOW</td>
                      <td>↓ LOW</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">SAM</td>
                      <td style={{ background: '#e8f5e9' }}>↑↑↑ VERY HIGH (300–2000+) PATHOGNOMONIC</td>
                      <td>↓↓ VERY LOW (&lt;50) PATHOGNOMONIC</td>
                      <td>↑ ELEVATED</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">SAH</td>
                      <td style={{ background: '#e8f5e9' }}>↓ LOW / NORMAL (5–30)</td>
                      <td>↓ LOW / NORMAL</td>
                      <td>↑↑↑ MASSIVELY HIGH PATHOGNOMONIC</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">SAM/SAH Ratio</td>
                      <td style={{ background: '#c8e6c9', fontWeight: 'bold' }}>↑↑↑ VERY HIGH (&gt;10–25) PATHOGNOMONIC</td>
                      <td>Normal or ↑</td>
                      <td style={{ color: ACCENT8, fontWeight: 'bold' }}>↓↓↓ VERY LOW (&lt;0.5) PATHOGNOMONIC</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">tHcy</td>
                      <td style={{ background: '#e8f5e9' }}>NORMAL (&lt;20) ✓</td>
                      <td>NORMAL (&lt;30) ✓</td>
                      <td>↑ MODERATE (40–150)</td>
                      <td>↑↑↑ HIGHEST (100–500)</td>
                      <td>↑↑ HIGH (50–300)</td>
                      <td>↑↑ HIGH (40–200)</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">Sarcosine</td>
                      <td style={{ background: '#c8e6c9', fontWeight: 'bold' }}>ABSENT ✓ (cannot make it)</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">MMA</td>
                      <td>NORMAL ✓</td>
                      <td>NORMAL ✓</td>
                      <td>NORMAL ✓</td>
                      <td>NORMAL ✓</td>
                      <td>NORMAL ✓</td>
                      <td>NORMAL ✓</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">Liver Disease</td>
                      <td style={{ background: '#fff9c4' }}>PROMINENT 60–80% (SAM toxicity)</td>
                      <td>40–50% (SAM deficiency)</td>
                      <td>70–75% (PEMT impaired)</td>
                      <td>ABSENT</td>
                      <td>ABSENT</td>
                      <td>ABSENT</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">Myopathy</td>
                      <td>ABSENT ✓</td>
                      <td>ABSENT ✓</td>
                      <td style={{ color: ACCENT2, fontWeight: 'bold' }}>85–90% HALLMARK</td>
                      <td>ABSENT ✓</td>
                      <td>ABSENT ✓</td>
                      <td>ABSENT ✓</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">Ectopia Lentis</td>
                      <td>ABSENT ✓</td>
                      <td>ABSENT ✓</td>
                      <td>ABSENT ✓</td>
                      <td style={{ color: ACCENT2, fontWeight: 'bold' }}>90% PATHOGNOMONIC</td>
                      <td>ABSENT ✓</td>
                      <td>ABSENT ✓</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">Breath Odor (DMS)</td>
                      <td>ABSENT ✓</td>
                      <td style={{ color: ACCENT2 }}>PRESENT PATHOGNOMONIC</td>
                      <td>ABSENT ✓</td>
                      <td>ABSENT ✓</td>
                      <td>ABSENT ✓</td>
                      <td>ABSENT ✓</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">SAM Supplement Rx</td>
                      <td style={{ color: ACCENT8, fontWeight: 'bold' }}>ABSOLUTE CI ⛔</td>
                      <td style={{ color: ACCENT4, fontWeight: 'bold' }}>Level A ✓</td>
                      <td style={{ color: ACCENT8 }}>ABSOLUTE CI ⛔</td>
                      <td>Level B</td>
                      <td>Not used</td>
                      <td>Not used</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">Betaine Rx</td>
                      <td style={{ color: ACCENT8, fontWeight: 'bold' }}>ABSOLUTE CI ⛔</td>
                      <td style={{ color: ACCENT8 }}>ABSOLUTE CI ⛔</td>
                      <td>HIGH RISK ⚠</td>
                      <td style={{ color: ACCENT4 }}>Level A ✓</td>
                      <td style={{ color: ACCENT4 }}>Level A ✓</td>
                      <td style={{ color: ACCENT4 }}>Level A ✓</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">NBS Detection</td>
                      <td>~50–60% (met ↑ but borderline)</td>
                      <td>~90% (extreme met)</td>
                      <td>~70% (met ↑ + SAH)</td>
                      <td>~60% (met ↑)</td>
                      <td>INVISIBLE</td>
                      <td>INVISIBLE</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* SAM Cycle Pathway */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>
              SAM Cycle — GNMT Safety Valve Position &amp; Consequence of LOF
            </div>
            <div className="card-body">
              <div className="row g-3">
                <div className="col-md-6">
                  <h6 className="fw-bold text-success">Normal SAM Cycle (GNMT Intact)</h6>
                  <ol style={{ fontSize: 13 }}>
                    <li>L-Methionine + ATP → <strong>SAM</strong> (via MAT1A/MAT3, liver)</li>
                    <li>SAM → SAH + Sarcosine (via <strong>GNMT</strong> — consumes 50–75% of hepatic SAM)</li>
                    <li>SAM → SAH (via ALL other methyltransferases: DNMT, COMT, GAMT, PEMT…)</li>
                    <li>SAH → Adenosine + Hcy (via AHCY — only route to clear SAH)</li>
                    <li>Hcy → Methionine (via MTR + MeCbl + 5-methylTHF) — remethylation</li>
                    <li>Hcy → Cystathionine (via CBS + PLP) — transsulfuration</li>
                    <li>Sarcosine → Glycine (via SARDH) — recycled back</li>
                  </ol>
                </div>
                <div className="col-md-6">
                  <h6 className="fw-bold" style={{ color: ACCENT2 }}>GNMT LOF — SAM Accumulation</h6>
                  <ol style={{ fontSize: 13 }}>
                    <li>L-Methionine + ATP → SAM (MAT1A still active — SYNTHESIS INTACT)</li>
                    <li style={{ color: ACCENT2 }}>SAM ✗→ GNMT BLOCKED → Sarcosine NOT MADE</li>
                    <li style={{ color: ACCENT2 }}>SAM accumulates — safety valve removed</li>
                    <li>SAM → SAH via other methyltransferases (minor; not enough)</li>
                    <li style={{ color: ACCENT2 }}>SAH stays LOW (GNMT was major SAH producer)</li>
                    <li style={{ color: ACCENT2 }}>SAM/SAH ratio MARKEDLY ELEVATED (&gt;10–25)</li>
                    <li>Hcy: NORMAL (CBS, MTR, MTRR all intact)</li>
                  </ol>
                  <div className="alert alert-success py-1 mt-2" style={{ fontSize: 12 }}>
                    <strong>Why tHcy is NORMAL:</strong> The homocysteine remethylation and transsulfuration pathways are completely intact in GNMT deficiency. SAM accumulation does not affect CBS/MTR function.
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Function/mechanism */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>
                  GNMT — The SAM Safety Valve (Unique Biology)
                </div>
                <div className="card-body" style={{ fontSize: 13 }}>
                  <p>{ov.function}</p>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT7, color: '#fff' }}>
                  Mechanism of SAM Toxicity &amp; Liver Disease
                </div>
                <div className="card-body" style={{ fontSize: 13 }}>
                  <p>{ov.mechanism}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Key positive/negative */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>
              Key Positive &amp; Negative Features — GNMT Diagnostic Signature
            </div>
            <div className="card-body" style={{ fontSize: 13 }}>
              <p>{ov.key_positive_features}</p>
            </div>
          </div>

          {/* NBS */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT4, color: '#fff' }}>
                  NBS Primary — Methionine Elevation
                </div>
                <div className="card-body" style={{ fontSize: 13 }}>
                  <p>{ov.nbs_primary}</p>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT4, color: '#fff' }}>
                  NBS Secondary — SAM/SAH Ratio + Sarcosine
                </div>
                <div className="card-body" style={{ fontSize: 13 }}>
                  <p>{ov.nbs_secondary}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Phenotype distribution */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>
              Phenotype Distribution (N={ov.cohort_n})
            </div>
            <div className="card-body">
              {Object.entries(ov.phenotype_distribution || {}).map(([ph, cnt]) => (
                <PctBar
                  key={ph}
                  label={`${ph} (n=${cnt})`}
                  pct={Math.round(cnt / ov.cohort_n * 100)}
                  color={ph.includes('Severe') ? ACCENT2 : ph.includes('Classic') ? ACCENT3 : ACCENT6}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 1: Patients & Biomarkers ── */}
      {tab === 1 && br && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>
                  Clinical Feature Prevalence
                </div>
                <div className="card-body">
                  <PctBar label="Liver Disease" pct={kpiPct.pct_liver_disease} color={ACCENT7} />
                  <PctBar label="Hepatomegaly" pct={kpiPct.pct_hepatomegaly} color={ACCENT7} />
                  <PctBar label="IDD" pct={kpiPct.pct_idd} color={ACCENT3} />
                  <PctBar label="Seizures" pct={kpiPct.pct_seizures} color={ACCENT2} />
                  <PctBar label="White Matter Disease" pct={kpiPct.pct_white_matter} color={ACCENT3} />
                  <PctBar label="Psychiatric" pct={kpiPct.pct_psychiatric} color={ACCENT5} />
                  <PctBar label="NBS Detected" pct={kpiPct.pct_nbs} color={ACCENT4} />
                  <PctBar label="Myopathy" pct={0} color={ACCENT5} />
                  <PctBar label="Cardiomyopathy" pct={0} color={ACCENT5} />
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>
                  Biomarker Ranges by Phenotype
                </div>
                <div className="card-body">
                  <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                    <thead className="table-dark">
                      <tr><th>Biomarker</th><th>Mild</th><th>Classic</th><th>Severe</th></tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Methionine (µmol/L)</td>
                        <td>40–120</td>
                        <td>80–250</td>
                        <td>180–500</td>
                      </tr>
                      <tr style={{ background: '#e8f5e9' }}>
                        <td className="fw-bold">SAM (µmol/L) ↑↑↑</td>
                        <td>120–400</td>
                        <td>300–900</td>
                        <td>800–2200</td>
                      </tr>
                      <tr>
                        <td>SAH (µmol/L) — LOW</td>
                        <td>5–20</td>
                        <td>6–25</td>
                        <td>8–30</td>
                      </tr>
                      <tr style={{ background: '#c8e6c9' }}>
                        <td className="fw-bold">SAM/SAH Ratio ↑↑↑</td>
                        <td>6–80</td>
                        <td>12–150</td>
                        <td>27–275</td>
                      </tr>
                      <tr>
                        <td>tHcy (µmol/L) — NORMAL</td>
                        <td>5–13</td>
                        <td>5–16</td>
                        <td>6–18</td>
                      </tr>
                      <tr>
                        <td>Sarcosine</td>
                        <td colSpan={3} className="text-center fw-bold" style={{ color: ACCENT3 }}>
                          ABSENT (undetectable) — ALL phenotypes
                        </td>
                      </tr>
                      <tr>
                        <td>MMA</td>
                        <td colSpan={3} className="text-center">NORMAL (all)</td>
                      </tr>
                      <tr>
                        <td>AST (U/L)</td>
                        <td>30–100</td>
                        <td>60–250</td>
                        <td>180–700</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Variant breakdown */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>
              GNMT Variant Distribution (Cohort N={br.cohort_n})
            </div>
            <div className="card-body">
              <div className="row g-2">
                {(br.variants || []).map(v => (
                  <div className="col-md-4 col-sm-6" key={v.variant}>
                    <div className="card h-100 shadow-sm">
                      <div className="card-body py-2">
                        <div className="fw-bold" style={{ fontSize: 13 }}>{v.variant}</div>
                        <PctBar label={`n=${v.count}`} pct={v.pct} color={ACCENT3} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Patient sample table */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>
              Patient Sample — First 12 of 40 (seed 107)
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover table-bordered mb-0" style={{ fontSize: 11 }}>
                  <thead className="table-dark">
                    <tr>
                      <th>ID</th><th>Phenotype</th><th>Variant</th>
                      <th>Met (µmol/L)</th><th>SAM (µmol/L)</th>
                      <th>SAM/SAH</th><th>SAH</th>
                      <th>tHcy</th><th>AST</th>
                      <th>Liver</th><th>Sz</th><th>NBS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(br.patient_sample || []).map(p => (
                      <tr key={p.id}>
                        <td>{p.id}</td>
                        <td style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.phenotype}</td>
                        <td>{p.variant}</td>
                        <td>{p.methionine}</td>
                        <td style={{ fontWeight: 'bold', color: ACCENT3 }}>{p.sam}</td>
                        <td style={{ fontWeight: 'bold', color: ACCENT8 }}>{p.sam_sah_ratio}</td>
                        <td>{p.sah}</td>
                        <td style={{ color: ACCENT6 }}>{p.homocysteine}</td>
                        <td>{p.ast}</td>
                        <td>{p.liver_disease ? '✓' : '—'}</td>
                        <td>{p.seizures ? '✓' : '—'}</td>
                        <td>{p.nbs ? '✓' : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: Seizures & Triggers ── */}
      {tab === 2 && br && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT2, color: '#fff' }}>
                  Seizure Type Distribution
                </div>
                <div className="card-body">
                  {(br.seizure_types || []).map(s => (
                    <PctBar key={s.type} label={s.type} pct={s.pct} color={ACCENT2} />
                  ))}
                  <div className="alert alert-info mt-2 py-2" style={{ fontSize: 12 }}>
                    <strong>Seizure prevalence:</strong> ~25–35% overall. Lower than AHCY/CBS because tHcy is NORMAL.
                    Seizures driven by SAM toxicity and white matter effects, not homocysteine excitotoxicity.
                    Liver disease severity correlates with seizure risk (hepatic encephalopathy component).
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT7, color: '#fff' }}>
                  Phenotype × Clinical Feature Matrix
                </div>
                <div className="card-body p-0">
                  <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
                    <thead className="table-dark">
                      <tr><th>Feature</th><th>Mild (40%)</th><th>Classic (45%)</th><th>Severe (15%)</th></tr>
                    </thead>
                    <tbody>
                      <tr><td>Liver Disease</td><td>55%</td><td>75%</td><td>90%</td></tr>
                      <tr><td>Hepatomegaly</td><td>45%</td><td>70%</td><td>85%</td></tr>
                      <tr><td>IDD</td><td>10%</td><td>50%</td><td>75%</td></tr>
                      <tr><td>Seizures</td><td>5%</td><td>25%</td><td>45%</td></tr>
                      <tr><td>White Matter</td><td>5%</td><td>22%</td><td>40%</td></tr>
                      <tr><td>SAM (avg, µmol/L)</td><td>~250</td><td>~600</td><td>~1500</td></tr>
                      <tr><td>tHcy</td><td colSpan={3} className="text-center" style={{ color: ACCENT6 }}>NORMAL in ALL phenotypes ✓</td></tr>
                      <tr><td>Myopathy</td><td colSpan={3} className="text-center" style={{ color: ACCENT5 }}>ABSENT in ALL ✓</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Metabolic triggers */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT8, color: '#fff' }}>
              Metabolic Triggers &amp; Contraindications — GNMT Safety Profile
            </div>
            <div className="card-body">
              {(br.metabolic_triggers || []).map((t, i) => (
                <div key={i} className="card mb-2 border-warning">
                  <div className="card-body py-2">
                    <div className="d-flex justify-content-between align-items-start mb-1">
                      <span className="fw-bold" style={{ fontSize: 13 }}>{t.trigger}</span>
                      <span className="badge ms-2" style={{
                        background: t.pct >= 85 ? ACCENT8 : t.pct >= 60 ? ACCENT7 : ACCENT3,
                        color: '#fff', fontSize: 11, whiteSpace: 'nowrap'
                      }}>{t.pct}% risk</span>
                    </div>
                    <div className="text-muted" style={{ fontSize: 12 }}>{t.mechanism}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: Treatments ── */}
      {tab === 3 && br && (
        <div>
          <div className="row g-3 mb-3">
            {(br.treatments || []).map((t, i) => (
              <div className="col-md-6" key={i}>
                <div className="card h-100 shadow-sm">
                  <div className="card-header d-flex justify-content-between align-items-center"
                    style={{ background: t.level.includes('Level A') ? ACCENT4 : t.level.includes('Level B') ? ACCENT : ACCENT5, color: '#fff', fontSize: 13 }}>
                    <span className="fw-bold">{t.name}</span>
                    <span className="badge bg-light text-dark ms-2" style={{ fontSize: 11 }}>{t.level}</span>
                  </div>
                  <div className="card-body" style={{ fontSize: 13 }}>
                    {t.note}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Drug risks */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT8, color: '#fff' }}>
              Drug Risk Summary — GNMT Deficiency
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Drug / Supplement</th><th>Risk Level</th><th>Mechanism</th></tr>
                </thead>
                <tbody>
                  {(br.drug_risks || []).map((d, i) => (
                    <tr key={i} style={{ background: d.risk.includes('ABSOLUTE') ? '#ffebee' : d.risk.includes('HIGH') ? '#fff3e0' : '#fffde7' }}>
                      <td className="fw-bold">{d.drug}</td>
                      <td style={{ color: d.risk.includes('ABSOLUTE') ? ACCENT8 : d.risk.includes('HIGH') ? ACCENT7 : ACCENT3, fontWeight: 'bold' }}>{d.risk}</td>
                      <td>{d.mechanism}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Treatment summary callout */}
          {df?.treatment_summary && (
            <div className="card mb-3">
              <div className="card-header fw-bold" style={{ background: ACCENT4, color: '#fff' }}>
                Treatment Summary
              </div>
              <div className="card-body">
                <div className="mb-2"><strong>First-line:</strong> {df.treatment_summary.first_line}</div>
                <div className="mb-2">
                  <strong>Absolute CI:</strong>{' '}
                  {(df.treatment_summary.absolute_ci || []).map((ci, i) => (
                    <span key={i} className="badge me-1 mb-1" style={{ background: ACCENT8, color: '#fff', fontSize: 11 }}>⛔ {ci}</span>
                  ))}
                </div>
                <div className="mb-2">
                  <strong>High Risk:</strong>{' '}
                  {(df.treatment_summary.high_risk || []).map((h, i) => (
                    <span key={i} className="badge me-1 mb-1" style={{ background: ACCENT7, color: '#fff', fontSize: 11 }}>⚠ {h}</span>
                  ))}
                </div>
                <div><strong>Liver Transplant:</strong> {df.treatment_summary.liver_transplant}</div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── TAB 4: Definitions ── */}
      {tab === 4 && df && (
        <div>
          {/* Gene card */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>
              Gene Card — GNMT
            </div>
            <div className="card-body">
              <table className="table table-sm table-bordered mb-0" style={{ fontSize: 13 }}>
                <tbody>
                  {Object.entries(df.gene_card || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold text-capitalize" style={{ width: '30%' }}>{k.replace(/_/g, ' ')}</td>
                      <td>{String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Key concepts */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>
              Key Concepts — GNMT Biology &amp; Pathophysiology
            </div>
            <div className="card-body">
              {(df.key_concepts || []).map((c, i) => (
                <div key={i} className="mb-3 p-3 rounded border" style={{ background: i % 2 === 0 ? '#f1f8e9' : '#e8eaf6' }}>
                  <div className="fw-bold mb-1" style={{ color: ACCENT3 }}>{c.term}</div>
                  <div style={{ fontSize: 13 }}>{c.definition}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Differential diagnosis */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT7, color: '#fff' }}>
              Differential Diagnosis — GNMT vs Related Disorders
            </div>
            <div className="card-body">
              {(df.differential_diagnosis || []).map((d, i) => (
                <div key={i} className="mb-2 p-3 rounded border-start border-4" style={{ borderColor: ACCENT7, background: '#fff8f0' }}>
                  <div className="fw-bold mb-1">{d.disorder}</div>
                  <div style={{ fontSize: 13 }}>{d.key_distinction}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Prevalence/rarity note */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT5, color: '#fff' }}>
              Epidemiology &amp; Rarity
            </div>
            <div className="card-body" style={{ fontSize: 13 }}>
              <p>{ov.prevalence}</p>
              <p>
                <strong>Worldwide:</strong> Fewer than 30 cases reported as of 2026. GNMT deficiency is among the rarest
                inherited metabolic disorders — even rarer than MTR (cblG, &lt;100 cases) and AHCY (&lt;50 cases).
                The paucity of cases means phenotypic spectrum and optimal treatment are still being defined.
              </p>
              <p>
                <strong>Folate-SAM link:</strong> GNMT is the primary enzyme linking folate status to SAM homeostasis.
                Individuals with low folate status have de-repressed GNMT — in heterozygotes this may lead to
                borderline SAM fluctuations without overt disease.
              </p>
              <p>
                <strong>Liver transplant outcomes:</strong> Cases with liver transplant show normalization of
                SAM/methionine post-operatively, confirming that GNMT's hepatic expression is the dominant
                source of the metabolic phenotype. Brain expression of GNMT is minimal.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Footer nav */}
      <div className="d-flex justify-content-between mt-4">
        <Link className="btn btn-outline-secondary btn-sm" href="/mat1a">← MAT1A</Link>
        <Link className="btn btn-outline-secondary btn-sm" href="/ahcy">AHCY →</Link>
      </div>
    </div>
  );
}

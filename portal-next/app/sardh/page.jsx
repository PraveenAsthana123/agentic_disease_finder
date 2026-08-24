'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

// SARDH color scheme — sarcosine/methylation / benign prognosis
const ACCENT  = '#4a0e8f';   // deep violet — sarcosine accumulation / SARDH pathway
const ACCENT2 = '#1b5e20';   // dark green — BENIGN prognosis / mostly asymptomatic
const ACCENT3 = '#e65100';   // deep orange — SARDH-GNMT metabolic contrast
const ACCENT4 = '#01579b';   // dark blue — folate/THF connection / NBS detected
const ACCENT5 = '#37474f';   // slate — key negatives / normal values
const ACCENT6 = '#880e4f';   // dark pink — absolute contraindications
const ACCENT7 = '#4e342e';   // dark brown — warnings / cautions
const ACCENT8 = '#006064';   // teal — asymptomatic / benign

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

export default function SARDHPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [br, setBr]       = useState(null);
  const [df, setDf]       = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/sardh/overview`).then(r => r.json()),
      fetch(`${API}/api/sardh/breakdown`).then(r => r.json()),
      fetch(`${API}/api/sardh/definitions`).then(r => r.json()),
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
            <div>N = {ov.cohort_n} patients · Seed 131</div>
          </div>
        </div>
      </div>

      {/* MOSTLY BENIGN banner */}
      <div className="alert py-2 mb-2 d-flex align-items-center gap-2" style={{ background: '#e8f5e9', border: '2px solid #2e7d32', fontSize: 13 }}>
        <span style={{ fontSize: 18 }}>✅</span>
        <span>
          <strong style={{ color: ACCENT2 }}>MOSTLY BENIGN — 50% of SARDH patients are ASYMPTOMATIC.</strong>
          {' '}Modern NBS reveals majority are incidental findings. Ascertainment bias inflated historical IDD estimates.
          Seizures, when present, respond to AED therapy. No liver disease. No myopathy.
        </span>
      </div>

      {/* Critical alerts */}
      <Alert
        variant="danger"
        text="⛔ SAMe / SAM Supplements — ABSOLUTE CONTRAINDICATION: SAM → GNMT → Sarcosine. SARDH cannot clear it → catastrophic sarcosine surge. OPPOSITE of MAT1A where SAM is Level A treatment."
      />
      <Alert
        variant="warning"
        text="⚠ SARDH vs GNMT — METABOLIC OPPOSITES: SARDH: Sarcosine MARKEDLY ELEVATED (50–800 µmol/L). GNMT: Sarcosine ABSENT. Single fastest discriminating biomarker between the two disorders."
      />
      <Alert
        variant="warning"
        text="⚠ Folate Paradox: High-dose folate (>400 µg/day) may worsen sarcosine load by driving GNMT activity → more sarcosine → accumulates when SARDH absent. Avoid excessive folate supplementation."
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
            <KPI label="Avg Sarcosine (µmol/L)" value={kpi.avg_sarcosine_umol_l} color={ACCENT} />
            <KPI label="Avg Glycine (µmol/L)" value={kpi.avg_glycine_umol_l} color={ACCENT7} />
            <KPI label="Avg Methionine — NORMAL" value={kpi.avg_methionine_umol_l} color={ACCENT5} />
            <KPI label="Avg tHcy — NORMAL" value={kpi.avg_homocysteine_umol_l} color={ACCENT5} />
            <KPI label="Avg SAM — NORMAL" value={kpi.avg_sam_umol_l} color={ACCENT5} />
            <KPI label="Avg SAM/SAH — NORMAL" value={kpi.avg_sam_sah_ratio} color={ACCENT5} />
            <KPI label="Seizures %" value={`${kpi.pct_seizures}%`} color={ACCENT3} />
            <KPI label="IDD %" value={`${kpi.pct_idd}%`} color={ACCENT} />
            <KPI label="NBS Detected %" value={`${kpi.pct_nbs_detected}%`} color={ACCENT4} />
            <KPI label="Asymptomatic %" value={`${kpi.pct_asymptomatic}%`} color={ACCENT2} />
            <KPI label="Liver Disease" value="0% ✓" color={ACCENT5} />
            <KPI label="Methionine NORMAL" value="100% ✓" color={ACCENT5} />
          </div>

          {/* SARDH vs GNMT Comparison Table — prominently placed */}
          <div className="card mb-3">
            <div className="card-header fw-bold text-white" style={{ background: ACCENT3 }}>
              SARDH vs GNMT — METABOLIC OPPOSITES (Single Most Important Comparison)
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
                  <thead className="table-dark">
                    <tr>
                      <th>Feature</th>
                      <th style={{ background: '#4a0e8f', color: '#fff' }}>SARDH ✦ (this dashboard)</th>
                      <th style={{ background: '#1b5e20', color: '#fff' }}>GNMT (opposite)</th>
                      <th>Clinical Significance</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ background: '#f3e5f5' }}>
                      <td className="fw-bold">Sarcosine</td>
                      <td style={{ fontWeight: 'bold', color: ACCENT }}>MARKEDLY ELEVATED (50–800 µmol/L) PATHOGNOMONIC</td>
                      <td style={{ fontWeight: 'bold', color: ACCENT2 }}>ABSENT (cannot be made)</td>
                      <td>COMPLETE OPPOSITE — single fastest discriminating test</td>
                    </tr>
                    <tr>
                      <td className="fw-bold">Methionine</td>
                      <td style={{ color: ACCENT5 }}>NORMAL (&lt;60 µmol/L) — KEY NEGATIVE</td>
                      <td style={{ color: ACCENT3 }}>HIGH (40–500 µmol/L)</td>
                      <td>SARDH does not affect methionine cycle</td>
                    </tr>
                    <tr style={{ background: '#fff3e0' }}>
                      <td className="fw-bold">SAM</td>
                      <td style={{ color: ACCENT5 }}>NORMAL (&lt;100 µmol/L) — KEY NEGATIVE</td>
                      <td style={{ color: '#b71c1c', fontWeight: 'bold' }}>MARKEDLY ELEVATED (800–2000+) PATHOGNOMONIC</td>
                      <td>SAM level immediately separates SARDH from GNMT</td>
                    </tr>
                    <tr>
                      <td className="fw-bold">SAM/SAH Ratio</td>
                      <td style={{ color: ACCENT5 }}>NORMAL (2–4)</td>
                      <td style={{ color: '#b71c1c', fontWeight: 'bold' }}>VERY HIGH (&gt;10–25) PATHOGNOMONIC</td>
                      <td>Ratio normal in SARDH; markedly high in GNMT</td>
                    </tr>
                    <tr>
                      <td className="fw-bold">tHcy</td>
                      <td style={{ color: ACCENT5 }}>NORMAL (&lt;15) — Shared KEY NEGATIVE</td>
                      <td style={{ color: ACCENT5 }}>NORMAL (&lt;20) — Shared KEY NEGATIVE</td>
                      <td>Both share normal tHcy — distinguish from CBS (100–500)</td>
                    </tr>
                    <tr style={{ background: '#e8f5e9' }}>
                      <td className="fw-bold">Liver Disease</td>
                      <td style={{ color: ACCENT2, fontWeight: 'bold' }}>ABSENT — no SAM hepatotoxicity</td>
                      <td style={{ color: ACCENT3 }}>PROMINENT 60–80% (SAM toxicity → NASH/cirrhosis)</td>
                      <td>Liver disease rules out SARDH → points to GNMT/AHCY</td>
                    </tr>
                    <tr>
                      <td className="fw-bold">Overall Prognosis</td>
                      <td style={{ color: ACCENT2, fontWeight: 'bold' }}>MOSTLY BENIGN — 50% asymptomatic</td>
                      <td style={{ color: ACCENT3 }}>Moderate-severe — liver disease dominant</td>
                      <td>SARDH far more benign than GNMT overall</td>
                    </tr>
                    <tr style={{ background: '#ffebee' }}>
                      <td className="fw-bold">SAM Supplement</td>
                      <td style={{ color: ACCENT6, fontWeight: 'bold' }}>ABSOLUTE CI ⛔ (drives GNMT → more sarcosine)</td>
                      <td style={{ color: ACCENT6, fontWeight: 'bold' }}>ABSOLUTE CI ⛔ (worsens SAM directly)</td>
                      <td>Both share SAM supplement CI but for different reasons</td>
                    </tr>
                    <tr>
                      <td className="fw-bold">Betaine/TMG</td>
                      <td style={{ color: ACCENT7, fontWeight: 'bold' }}>HIGH RISK (→ methionine → SAM → GNMT → sarcosine)</td>
                      <td style={{ color: ACCENT6 }}>ABSOLUTE CI ⛔</td>
                      <td>Both avoid betaine; GNMT stricter</td>
                    </tr>
                    <tr>
                      <td className="fw-bold">Gene / Locus</td>
                      <td>SARDH / 9q34.3 / 972 aa / FAD+THF / mitochondrial / homodimer</td>
                      <td>GNMT / 9q22.2 / 296 aa / SAM-dependent / PLP-independent / homotetramer</td>
                      <td>Both on chromosome 9; opposite metabolic functions</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* SARDH Reaction Pathway */}
          <div className="card mb-3">
            <div className="card-header fw-bold text-white" style={{ background: ACCENT }}>
              SARDH Reaction — Pathway Position &amp; Consequence of LOF
            </div>
            <div className="card-body">
              <div className="row g-3">
                <div className="col-md-6">
                  <h6 className="fw-bold" style={{ color: ACCENT2 }}>Normal Sarcosine Degradation Pathway</h6>
                  <ol style={{ fontSize: 13 }}>
                    <li>Choline → Betaine (CHDH)</li>
                    <li>Betaine → Dimethylglycine (DMG) + 5-methylTHF (BHMT)</li>
                    <li>DMG → <strong>Sarcosine</strong> (DMGDH — one step upstream of SARDH)</li>
                    <li>Sarcosine + THF → <strong>Glycine</strong> + 5,10-methyleneTHF (via <strong>SARDH</strong>)</li>
                    <li>5,10-methyleneTHF → used for serine biosynthesis + thymidylate synthesis</li>
                    <li>Glycine → many metabolic fates (TCA, heme, bile acids, glutathione)</li>
                    <li>GNMT (parallel): SAM + Glycine → SAH + Sarcosine (also feeds SARDH)</li>
                  </ol>
                </div>
                <div className="col-md-6">
                  <h6 className="fw-bold" style={{ color: ACCENT3 }}>SARDH LOF — Sarcosine Accumulation</h6>
                  <ol style={{ fontSize: 13 }}>
                    <li>DMG → Sarcosine (DMGDH INTACT — one step upstream OK)</li>
                    <li style={{ color: ACCENT }}>Sarcosine ✗→ SARDH BLOCKED — cannot convert to Glycine</li>
                    <li style={{ color: ACCENT }}>Sarcosine ACCUMULATES in plasma (50–800 µmol/L) + urine (sarcosinuria)</li>
                    <li>Glycine: mildly reduced production from this path; compensated by other sources</li>
                    <li style={{ color: ACCENT }}>5,10-methyleneTHF: mildly reduced (SARDH product not made)</li>
                    <li>SAM cycle: COMPLETELY INTACT (methionine, SAM, SAH, tHcy all NORMAL)</li>
                    <li style={{ color: ACCENT }}>GNMT still active → SAM + Glycine → Sarcosine (this feeds more substrate when SARDH absent)</li>
                  </ol>
                  <div className="alert alert-success py-1 mt-2" style={{ fontSize: 12 }}>
                    <strong>Key insight:</strong> SARDH deficiency does NOT disrupt the SAM cycle. SAM, SAH, methionine, and tHcy are ALL NORMAL. Only sarcosine accumulates. This is why SARDH is mostly benign — no SAM toxicity, no homocysteine excitotoxicity.
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Function/mechanism */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold text-white" style={{ background: ACCENT }}>
                  SARDH — FAD+THF Dual Cofactor Reaction
                </div>
                <div className="card-body" style={{ fontSize: 13 }}>
                  <p>{ov.function}</p>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold text-white" style={{ background: ACCENT3 }}>
                  Mechanism of Sarcosine Accumulation &amp; Mostly Benign Phenotype
                </div>
                <div className="card-body" style={{ fontSize: 13 }}>
                  <p>{ov.mechanism}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Key positive/negative features */}
          <div className="card mb-3">
            <div className="card-header fw-bold text-white" style={{ background: ACCENT }}>
              Key Positive &amp; Negative Features — SARDH Diagnostic Signature
            </div>
            <div className="card-body" style={{ fontSize: 13 }}>
              <p>{ov.key_positive_features}</p>
            </div>
          </div>

          {/* NBS */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold text-white" style={{ background: ACCENT4 }}>
                  NBS Primary — Sarcosine Elevation
                </div>
                <div className="card-body" style={{ fontSize: 13 }}>
                  <p>{ov.nbs_primary}</p>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold text-white" style={{ background: ACCENT4 }}>
                  NBS Secondary — Confirmatory Metabolomics
                </div>
                <div className="card-body" style={{ fontSize: 13 }}>
                  <p>{ov.nbs_secondary}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Phenotype distribution */}
          <div className="card mb-3">
            <div className="card-header fw-bold text-white" style={{ background: ACCENT2 }}>
              Phenotype Distribution — MOSTLY BENIGN (N={ov.cohort_n})
            </div>
            <div className="card-body">
              {Object.entries(ov.phenotype_distribution || {}).map(([ph, cnt]) => (
                <PctBar
                  key={ph}
                  label={`${ph} (n=${cnt})`}
                  pct={Math.round(cnt / ov.cohort_n * 100)}
                  color={
                    ph.includes('Asymptomatic') ? ACCENT2 :
                    ph.includes('Mild') ? ACCENT4 :
                    ACCENT3
                  }
                />
              ))}
              <div className="alert alert-success py-2 mt-2" style={{ fontSize: 12 }}>
                <strong>Ascertainment Bias Note:</strong> Historical case series (1960s–1980s) reported IDD in ~60–70%
                of SARDH cases — but only symptomatic patients reached diagnosis at that time.
                Modern NBS reveals 50% are asymptomatic. SARDH is one of the most benign metabolic IEMs.
              </div>
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
                <div className="card-header fw-bold text-white" style={{ background: ACCENT }}>
                  Clinical Feature Prevalence
                </div>
                <div className="card-body">
                  <PctBar label="Asymptomatic (NBS-only)" pct={kpiPct.pct_asymptomatic} color={ACCENT2} />
                  <PctBar label="Mild Neurodevelopmental" pct={kpiPct.pct_mild_neurodev} color={ACCENT4} />
                  <PctBar label="Classic Symptomatic" pct={kpiPct.pct_classic} color={ACCENT3} />
                  <PctBar label="IDD (any)" pct={kpiPct.pct_idd} color={ACCENT} />
                  <PctBar label="Seizures" pct={kpiPct.pct_seizures} color={ACCENT3} />
                  <PctBar label="Behavioral" pct={kpiPct.pct_behavioral} color={ACCENT7} />
                  <PctBar label="Attention issues" pct={kpiPct.pct_attention} color={ACCENT7} />
                  <PctBar label="NBS Detected" pct={kpiPct.pct_nbs} color={ACCENT4} />
                  <PctBar label="Liver Disease" pct={0} color={ACCENT5} />
                  <PctBar label="Myopathy" pct={0} color={ACCENT5} />
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold text-white" style={{ background: ACCENT3 }}>
                  Biomarker Ranges by Phenotype
                </div>
                <div className="card-body">
                  <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                    <thead className="table-dark">
                      <tr><th>Biomarker</th><th>Asymptomatic</th><th>Mild</th><th>Classic</th></tr>
                    </thead>
                    <tbody>
                      <tr style={{ background: '#f3e5f5' }}>
                        <td className="fw-bold" style={{ color: ACCENT }}>Sarcosine (µmol/L) ↑↑↑</td>
                        <td>50–200</td>
                        <td>150–400</td>
                        <td>350–800</td>
                      </tr>
                      <tr>
                        <td>Glycine (µmol/L) — mildly ↑</td>
                        <td>230–350</td>
                        <td>280–420</td>
                        <td>350–500</td>
                      </tr>
                      <tr style={{ background: '#e8f5e9' }}>
                        <td className="fw-bold" style={{ color: ACCENT5 }}>Methionine — NORMAL</td>
                        <td colSpan={3} className="text-center">25–55 µmol/L (all phenotypes)</td>
                      </tr>
                      <tr style={{ background: '#e8f5e9' }}>
                        <td className="fw-bold" style={{ color: ACCENT5 }}>tHcy — NORMAL</td>
                        <td colSpan={3} className="text-center">5–13 µmol/L (all phenotypes)</td>
                      </tr>
                      <tr style={{ background: '#e8f5e9' }}>
                        <td className="fw-bold" style={{ color: ACCENT5 }}>SAM — NORMAL</td>
                        <td colSpan={3} className="text-center">50–95 µmol/L (all phenotypes)</td>
                      </tr>
                      <tr style={{ background: '#e8f5e9' }}>
                        <td className="fw-bold" style={{ color: ACCENT5 }}>SAH — NORMAL</td>
                        <td colSpan={3} className="text-center">8–22 µmol/L (all phenotypes)</td>
                      </tr>
                      <tr style={{ background: '#e8f5e9' }}>
                        <td className="fw-bold" style={{ color: ACCENT5 }}>MMA</td>
                        <td colSpan={3} className="text-center fw-bold">NORMAL (all) ✓</td>
                      </tr>
                      <tr style={{ background: '#e8f5e9' }}>
                        <td className="fw-bold" style={{ color: ACCENT5 }}>Liver Function</td>
                        <td colSpan={3} className="text-center fw-bold" style={{ color: ACCENT2 }}>NORMAL (all) — no SAM hepatotoxicity ✓</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Variant breakdown */}
          <div className="card mb-3">
            <div className="card-header fw-bold text-white" style={{ background: ACCENT }}>
              SARDH Variant Distribution (Cohort N={br.cohort_n})
            </div>
            <div className="card-body">
              <div className="row g-2">
                {(br.variants || []).map(v => (
                  <div className="col-md-4 col-sm-6" key={v.variant}>
                    <div className="card h-100 shadow-sm">
                      <div className="card-body py-2">
                        <div className="fw-bold" style={{ fontSize: 13, color: ACCENT }}>{v.variant}</div>
                        <PctBar label={`n=${v.count}`} pct={v.pct} color={ACCENT} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-2 text-muted" style={{ fontSize: 12 }}>
                <strong>Most common:</strong> {br.variant_distribution?.most_common} |{' '}
                <strong>Most severe:</strong> {br.variant_distribution?.most_severe} |{' '}
                <strong>Most attenuated:</strong> {br.variant_distribution?.most_attenuated}
              </div>
            </div>
          </div>

          {/* Patient sample table */}
          <div className="card mb-3">
            <div className="card-header fw-bold text-white" style={{ background: ACCENT3 }}>
              Patient Sample — First 12 of 40 (seed 131)
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover table-bordered mb-0" style={{ fontSize: 11 }}>
                  <thead className="table-dark">
                    <tr>
                      <th>ID</th><th>Phenotype</th><th>Variant</th>
                      <th>Sarcosine (µmol/L)</th><th>Glycine</th>
                      <th>Met (NORMAL)</th><th>SAM (NORMAL)</th>
                      <th>SAM/SAH</th><th>tHcy</th>
                      <th>IDD</th><th>Sz</th><th>NBS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(br.patient_sample || []).map(p => (
                      <tr key={p.id}>
                        <td>{p.id}</td>
                        <td style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.phenotype}</td>
                        <td>{p.variant}</td>
                        <td style={{ fontWeight: 'bold', color: ACCENT }}>{p.sarcosine}</td>
                        <td>{p.glycine}</td>
                        <td style={{ color: ACCENT5 }}>{p.methionine}</td>
                        <td style={{ color: ACCENT5 }}>{p.sam}</td>
                        <td style={{ color: ACCENT5 }}>{p.sam_sah_ratio}</td>
                        <td style={{ color: ACCENT5 }}>{p.homocysteine}</td>
                        <td>{p.idd ? '✓' : '—'}</td>
                        <td>{p.seizures ? '✓' : '—'}</td>
                        <td>{p.nbs ? '✓' : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Phenotype comparison table (SARDH vs GNMT from breakdown) */}
          {br.phenotype_comparison?.sardh_vs_gnmt && (
            <div className="card mb-3">
              <div className="card-header fw-bold text-white" style={{ background: ACCENT3 }}>
                SARDH vs GNMT — Full Comparison Table
              </div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
                    <thead className="table-dark">
                      <tr><th>Feature</th><th>SARDH ✦</th><th>GNMT</th></tr>
                    </thead>
                    <tbody>
                      {br.phenotype_comparison.sardh_vs_gnmt.map((row, i) => (
                        <tr key={i} style={{ background: i % 2 === 0 ? '#fafafa' : '#f5f0ff' }}>
                          <td className="fw-bold">{row.feature}</td>
                          <td style={{ color: ACCENT }}>{row.sardh}</td>
                          <td style={{ color: ACCENT3 }}>{row.gnmt}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── TAB 2: Seizures & Triggers ── */}
      {tab === 2 && br && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold text-white" style={{ background: ACCENT3 }}>
                  Seizure Type Distribution (when seizures present)
                </div>
                <div className="card-body">
                  {(br.seizure_types || []).map(s => (
                    <PctBar key={s.type} label={s.type} pct={s.pct} color={ACCENT3} />
                  ))}
                  <div className="alert alert-success mt-2 py-2" style={{ fontSize: 12 }}>
                    <strong>Overall seizure prevalence:</strong> ~{kpiPct.pct_seizures}% (cohort). Much lower than GAMT (60–80%) or SLC6A8 (80–90%). Seizures are mild-moderate; drug-resistant epilepsy is rare. LEV first-line.
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold text-white" style={{ background: ACCENT }}>
                  Phenotype × Clinical Feature Matrix
                </div>
                <div className="card-body p-0">
                  <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
                    <thead className="table-dark">
                      <tr><th>Feature</th><th>Asymptomatic (50%)</th><th>Mild-Neurodev (35%)</th><th>Classic (15%)</th></tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Sarcosine (µmol/L)</td>
                        <td style={{ color: ACCENT }}>50–200</td>
                        <td style={{ color: ACCENT }}>150–400</td>
                        <td style={{ color: ACCENT, fontWeight: 'bold' }}>350–800</td>
                      </tr>
                      <tr>
                        <td>IDD</td>
                        <td>0%</td>
                        <td>~40%</td>
                        <td>~75%</td>
                      </tr>
                      <tr>
                        <td>Seizures</td>
                        <td>0%</td>
                        <td>~25%</td>
                        <td>~55%</td>
                      </tr>
                      <tr>
                        <td>Behavioral</td>
                        <td>~5%</td>
                        <td>~55%</td>
                        <td>~70%</td>
                      </tr>
                      <tr>
                        <td>Methionine</td>
                        <td colSpan={3} className="text-center" style={{ color: ACCENT5 }}>NORMAL (25–55 µmol/L) in ALL phenotypes ✓</td>
                      </tr>
                      <tr>
                        <td>SAM</td>
                        <td colSpan={3} className="text-center" style={{ color: ACCENT5 }}>NORMAL (50–95 µmol/L) in ALL phenotypes ✓</td>
                      </tr>
                      <tr>
                        <td>tHcy</td>
                        <td colSpan={3} className="text-center" style={{ color: ACCENT5 }}>NORMAL (&lt;15 µmol/L) in ALL phenotypes ✓</td>
                      </tr>
                      <tr>
                        <td>Liver Disease</td>
                        <td colSpan={3} className="text-center fw-bold" style={{ color: ACCENT2 }}>ABSENT in ALL ✓</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Metabolic triggers */}
          <div className="card mb-3">
            <div className="card-header fw-bold text-white" style={{ background: ACCENT7 }}>
              Metabolic Triggers &amp; Contraindications — SARDH Safety Profile
            </div>
            <div className="card-body">
              {(br.metabolic_triggers || []).map((t, i) => (
                <div key={i} className="card mb-2 border-warning">
                  <div className="card-body py-2">
                    <div className="d-flex justify-content-between align-items-start mb-1">
                      <span className="fw-bold" style={{ fontSize: 13 }}>{t.trigger}</span>
                      <span className="badge ms-2" style={{
                        background: t.pct >= 50 ? ACCENT7 : t.pct >= 30 ? ACCENT3 : ACCENT4,
                        color: '#fff', fontSize: 11, whiteSpace: 'nowrap'
                      }}>{t.pct}% risk</span>
                    </div>
                    <div className="text-muted" style={{ fontSize: 12 }}>{t.mechanism}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Treatment matrix */}
          {br.treatment_matrix && (
            <div className="card mb-3">
              <div className="card-header fw-bold text-white" style={{ background: ACCENT2 }}>
                Treatment Matrix by Phenotype
              </div>
              <div className="card-body p-0">
                <table className="table table-sm table-bordered mb-0" style={{ fontSize: 13 }}>
                  <thead className="table-dark">
                    <tr><th>Phenotype</th><th>Treatment Strategy</th></tr>
                  </thead>
                  <tbody>
                    <tr style={{ background: '#e8f5e9' }}>
                      <td className="fw-bold" style={{ color: ACCENT2 }}>Asymptomatic (50%)</td>
                      <td>{br.treatment_matrix.asymptomatic}</td>
                    </tr>
                    <tr>
                      <td className="fw-bold" style={{ color: ACCENT4 }}>Mild-Neurodevelopmental (35%)</td>
                      <td>{br.treatment_matrix.mild_neurodev}</td>
                    </tr>
                    <tr style={{ background: '#fff3e0' }}>
                      <td className="fw-bold" style={{ color: ACCENT3 }}>Classic-Symptomatic (15%)</td>
                      <td>{br.treatment_matrix.classic}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}
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
                    style={{
                      background: t.level.includes('ABSOLUTE') ? ACCENT6 :
                                  t.level.includes('Level A') ? ACCENT4 :
                                  t.level.includes('Level B') ? ACCENT2 :
                                  t.level.includes('HIGH RISK') ? ACCENT7 :
                                  ACCENT5,
                      color: '#fff', fontSize: 13
                    }}>
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
            <div className="card-header fw-bold text-white" style={{ background: ACCENT6 }}>
              Drug Risk Summary — SARDH Deficiency
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Drug / Supplement</th><th>Risk Level</th><th>Mechanism</th></tr>
                </thead>
                <tbody>
                  {(br.drug_risks || []).map((d, i) => (
                    <tr key={i} style={{
                      background: d.risk.includes('ABSOLUTE') ? '#ffebee' :
                                  d.risk.includes('HIGH') ? '#fff3e0' :
                                  d.risk.includes('MODERATE') ? '#fffde7' :
                                  '#f9fbe7'
                    }}>
                      <td className="fw-bold">{d.drug}</td>
                      <td style={{
                        color: d.risk.includes('ABSOLUTE') ? ACCENT6 :
                               d.risk.includes('HIGH') ? ACCENT7 :
                               d.risk.includes('MODERATE') ? ACCENT3 :
                               ACCENT5,
                        fontWeight: 'bold'
                      }}>{d.risk}</td>
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
              <div className="card-header fw-bold text-white" style={{ background: ACCENT2 }}>
                Treatment Summary — SARDH
              </div>
              <div className="card-body">
                <div className="mb-2"><strong>First-line:</strong> {df.treatment_summary.first_line}</div>
                <div className="mb-2">
                  <strong>Absolute CI:</strong>{' '}
                  {(df.treatment_summary.absolute_ci || []).map((ci, i) => (
                    <span key={i} className="badge me-1 mb-1" style={{ background: ACCENT6, color: '#fff', fontSize: 11 }}>⛔ {ci}</span>
                  ))}
                </div>
                <div className="mb-2">
                  <strong>High Risk:</strong>{' '}
                  {(df.treatment_summary.high_risk || []).map((h, i) => (
                    <span key={i} className="badge me-1 mb-1" style={{ background: ACCENT7, color: '#fff', fontSize: 11 }}>⚠ {h}</span>
                  ))}
                </div>
                <div className="mb-2">
                  <strong>Caution:</strong>{' '}
                  {(df.treatment_summary.caution || []).map((c, i) => (
                    <span key={i} className="badge me-1 mb-1" style={{ background: ACCENT4, color: '#fff', fontSize: 11 }}>⚠ {c}</span>
                  ))}
                </div>
                <div className="alert alert-success py-2 mt-2" style={{ fontSize: 13 }}>
                  <strong>Prognosis:</strong> {df.treatment_summary.prognosis}
                </div>
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
            <div className="card-header fw-bold text-white" style={{ background: ACCENT }}>
              Gene Card — SARDH
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
            <div className="card-header fw-bold text-white" style={{ background: ACCENT }}>
              Key Concepts — SARDH Biology &amp; Pathophysiology
            </div>
            <div className="card-body">
              {(df.key_concepts || []).map((c, i) => (
                <div key={i} className="mb-3 p-3 rounded border" style={{ background: i % 2 === 0 ? '#f3e5f5' : '#fff3e0' }}>
                  <div className="fw-bold mb-1" style={{ color: ACCENT }}>{c.term}</div>
                  <div style={{ fontSize: 13 }}>{c.definition}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Differential diagnosis */}
          <div className="card mb-3">
            <div className="card-header fw-bold text-white" style={{ background: ACCENT3 }}>
              Differential Diagnosis — SARDH vs Related Disorders
            </div>
            <div className="card-body">
              {(df.differential_diagnosis || []).map((d, i) => (
                <div key={i} className="mb-2 p-3 rounded border-start border-4" style={{ borderColor: ACCENT3, background: '#fff8f0' }}>
                  <div className="fw-bold mb-1">{d.disorder}</div>
                  <div style={{ fontSize: 13 }}>{d.key_distinction}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Epidemiology note */}
          <div className="card mb-3">
            <div className="card-header fw-bold text-white" style={{ background: ACCENT5 }}>
              Epidemiology, Rarity &amp; Benign Prognosis
            </div>
            <div className="card-body" style={{ fontSize: 13 }}>
              <p>{ov.prevalence}</p>
              <p>
                <strong>Worldwide:</strong> ~50–100 cases reported as of 2026. Rarer than GNMT (&lt;30 cases) is
                expected — but SARDH likely has many asymptomatic undetected carriers who never reach diagnosis
                in settings without comprehensive amino acid NBS. True population prevalence is uncertain.
              </p>
              <p>
                <strong>Chromosome 9:</strong> SARDH (9q34.3) and GNMT (9q22.2) are both located on chromosome 9,
                reflect opposite enzymatic functions in the sarcosine metabolic axis — one of the clearest
                examples of a metabolic "yin-yang" pair in inherited metabolic disease.
              </p>
              <p>
                <strong>FAD cofactor dependency:</strong> SARDH shares FAD-dependency with DMGDH (dimethylglycine
                dehydrogenase, one step upstream). Both are mitochondrial. FAD availability (riboflavin) does not
                typically rate-limit SARDH in normal nutrition — riboflavin supplementation is NOT a treatment
                (unlike MADD/GAII where riboflavin is Level A).
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Footer nav */}
      <div className="d-flex justify-content-between mt-4">
        <Link className="btn btn-outline-secondary btn-sm" href="/slc6a8">← SLC6A8</Link>
        <Link className="btn btn-outline-secondary btn-sm" href="/gnmt">GNMT →</Link>
      </div>
    </div>
  );
}

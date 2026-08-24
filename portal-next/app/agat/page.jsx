'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1b5e20';   // deep green — AGAT / creatine biosynthesis Step 1
const ACCENT2 = '#bf360c';   // deep orange-red — seizures / creatine energy deficit
const ACCENT3 = '#006064';   // deep teal — GAA VERY LOW / pathognomonic (opposite direction)
const ACCENT4 = '#1565c0';   // blue — Level A treatment / creatine
const ACCENT5 = '#37474f';   // slate — KEY NEGATIVES / normal values
const ACCENT6 = '#4e342e';   // brown — creatine absent / H-MRS
const ACCENT7 = '#4a148c';   // deep purple — IDD / speech absent
const ACCENT8 = '#e65100';   // amber-orange — drug risks / catabolic triggers

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

export default function AGATPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [br, setBr]       = useState(null);
  const [df, setDf]       = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/agat/overview`).then(r => r.json()),
      fetch(`${API}/api/agat/breakdown`).then(r => r.json()),
      fetch(`${API}/api/agat/definitions`).then(r => r.json()),
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
            <div>N = {ov.cohort_n} patients · Seed 117</div>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert
        variant="success"
        text="🧬 SINGLE PATHOLOGY: GAA VERY LOW / ABSENT (<0.5 µmol/L; normal 0.5–3) — AGAT cannot make GAA → GAMT has no substrate → Creatine ABSENT. No GAA neurotoxicity (unlike GAMT). Brain H-MRS creatine peak (3.0 ppm) ABSENT — PATHOGNOMONIC shared CCDS feature."
      />
      <Alert
        variant="info"
        text="✅ KEY vs GAMT: GAA VERY LOW (not high) · Seizures LESS drug-resistant (25–35% vs GAMT 60–80%) · Creatine monohydrate ONLY (Level A) — NO ornithine, NO arginine restriction needed. SLC6A8 intact → creatine supplementation IS EFFECTIVE (unlike SLC6A8 deficiency)."
      />
      <Alert
        variant="warning"
        text="⚠ KEY NEGATIVES: Methionine NORMAL · tHcy NORMAL · SAM NORMAL · MMA NORMAL — distinguishes AGAT from ALL hypermethioninemia / homocystinuria / MMA disorders. Single plasma GAA + creatine measurement immediately confirms diagnosis."
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
            <KPI label="Avg GAA (µmol/L) ↓↓" value={kpi.avg_gaa_umol_l} color={ACCENT3} />
            <KPI label="Avg Creatine (µmol/L) ↓" value={kpi.avg_creatine_umol_l} color={ACCENT2} />
            <KPI label="Avg Creatinine (µmol/L)" value={kpi.avg_creatinine_umol_l} color={ACCENT6} />
            <KPI label="Avg Methionine (µmol/L)" value={kpi.avg_methionine_umol_l} color={ACCENT5} />
            <KPI label="Avg tHcy (µmol/L)" value={kpi.avg_homocysteine_umol_l} color={ACCENT5} />
            <KPI label="Avg CK (U/L)" value={kpi.avg_ck_u_l} color={ACCENT5} />
            <KPI label="Seizures %" value={`${kpi.pct_seizures}%`} color={ACCENT2} />
            <KPI label="Drug-Resistant Sz %" value={`${kpi.pct_drug_resistant}%`} color={ACCENT2} />
            <KPI label="IDD %" value={`${kpi.pct_idd}%`} color={ACCENT7} />
            <KPI label="Speech Absent %" value={`${kpi.pct_speech_absent}%`} color={ACCENT7} />
            <KPI label="NBS Detected %" value={`${kpi.pct_nbs_detected}%`} color={ACCENT4} />
            <KPI label="Met NORMAL" value="100% ✓" color={ACCENT5} />
          </div>

          {/* AGAT vs All Creatine/Metabolic Comparison Table */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>
              AGAT vs Related Disorders — Biomarker Fingerprint (Creatine CCDS + SAM-Cycle)
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
                  <thead className="table-dark">
                    <tr>
                      <th>Feature</th>
                      <th style={{ background: '#1b5e20', color: '#fff' }}>AGAT ✦ (CCDS3)</th>
                      <th>GAMT (CCDS2)</th>
                      <th>SLC6A8 (CCDS1)</th>
                      <th>GNMT</th>
                      <th>MAT1A</th>
                      <th>AHCY</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="fw-semibold">Guanidinoacetate (GAA)</td>
                      <td style={{ background: '#e0f2f1', fontWeight: 'bold', color: ACCENT3 }}>↓ VERY LOW / ABSENT &lt;0.5 µmol/L</td>
                      <td style={{ color: '#b71c1c', fontWeight: 'bold' }}>↑↑↑ 50–300 µmol/L PATHOGNOMONIC</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">Creatine (plasma)</td>
                      <td style={{ background: '#ffebee', fontWeight: 'bold' }}>ABSENT / &lt;5 µmol/L</td>
                      <td>ABSENT / &lt;5 µmol/L</td>
                      <td style={{ color: ACCENT8 }}>↑ HIGH (cannot enter cells)</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">Brain H-MRS (3.0 ppm)</td>
                      <td style={{ fontWeight: 'bold', color: ACCENT2 }}>ABSENT ✗</td>
                      <td style={{ fontWeight: 'bold', color: ACCENT2 }}>ABSENT ✗</td>
                      <td style={{ fontWeight: 'bold', color: ACCENT2 }}>ABSENT ✗</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                      <td>NORMAL</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">Methionine</td>
                      <td style={{ background: '#e8f5e9' }}>NORMAL ✓</td>
                      <td>NORMAL ✓</td>
                      <td>NORMAL ✓</td>
                      <td style={{ color: ACCENT8 }}>↑ HIGH (40–500)</td>
                      <td style={{ color: ACCENT8 }}>↑↑↑ EXTREME (200–2000+)</td>
                      <td style={{ color: ACCENT8 }}>↑↑ VERY HIGH (200–600)</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">SAM</td>
                      <td style={{ background: '#e8f5e9' }}>NORMAL ✓ (AGAT not SAM-dependent)</td>
                      <td>NORMAL-LOW</td>
                      <td>NORMAL</td>
                      <td style={{ color: '#4a148c', fontWeight: 'bold' }}>↑↑↑ VERY HIGH PATHOGNOMONIC</td>
                      <td style={{ color: ACCENT3 }}>↓↓ VERY LOW</td>
                      <td>↑ ELEVATED</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">tHcy</td>
                      <td style={{ background: '#e8f5e9' }}>NORMAL ✓ (&lt;15)</td>
                      <td>NORMAL ✓</td>
                      <td>NORMAL ✓</td>
                      <td>NORMAL ✓ (&lt;20)</td>
                      <td>NORMAL ✓ (&lt;30)</td>
                      <td>↑ MODERATE (40–150)</td>
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
                      <td className="fw-semibold">Drug-Resistant Epilepsy</td>
                      <td style={{ color: ACCENT, fontWeight: 'bold' }}>25–35% (mild — no GAA toxicity)</td>
                      <td style={{ color: '#b71c1c', fontWeight: 'bold' }}>60–80% DRUG-RESISTANT (GAA)</td>
                      <td>60–70% (males)</td>
                      <td>25–35%</td>
                      <td>15–25%</td>
                      <td>30–40%</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">IDD severity</td>
                      <td style={{ color: ACCENT7, fontWeight: 'bold' }}>MODERATE-PROFOUND (100%)</td>
                      <td style={{ color: '#b71c1c', fontWeight: 'bold' }}>PROFOUND (90%+) dual mechanism</td>
                      <td>PROFOUND (males)</td>
                      <td>MODERATE 35–50%</td>
                      <td>MODERATE 30–40%</td>
                      <td>MODERATE 60–65%</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">Creatine Tx efficacy</td>
                      <td style={{ color: ACCENT4, fontWeight: 'bold' }}>Level A HIGHLY EFFECTIVE ✓</td>
                      <td style={{ color: ACCENT4 }}>Level A effective ✓</td>
                      <td style={{ color: '#b71c1c' }}>LARGELY INEFFECTIVE ✗</td>
                      <td>NOT INDICATED</td>
                      <td>Level B</td>
                      <td>Level B</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">Ornithine Tx</td>
                      <td style={{ color: ACCENT5, fontWeight: 'bold' }}>NOT NEEDED (no GAA) ✗</td>
                      <td style={{ color: ACCENT4, fontWeight: 'bold' }}>Level A ✓ (reduces GAA)</td>
                      <td>NOT NEEDED</td>
                      <td>NOT INDICATED</td>
                      <td>NOT INDICATED</td>
                      <td>NOT INDICATED</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">NBS detection</td>
                      <td>~40–60% (low GAA; harder than GAMT high GAA)</td>
                      <td>~70–75% (high GAA; easier)</td>
                      <td>~50% (Cr/Crn ratio)</td>
                      <td>~50–60% (met ↑)</td>
                      <td>~90% (extreme met)</td>
                      <td>~70% (met + SAH)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Creatine Pathway */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>
              Creatine Biosynthesis Pathway — AGAT Position (Step 1) &amp; Consequence of LOF
            </div>
            <div className="card-body">
              <div className="row g-3">
                <div className="col-md-6">
                  <h6 className="fw-bold text-success">Normal Creatine Biosynthesis (AGAT Intact)</h6>
                  <ol style={{ fontSize: 13 }}>
                    <li><strong>AGAT</strong> (kidney/pancreas): L-Arginine + Glycine → L-Ornithine + <strong>Guanidinoacetate (GAA)</strong></li>
                    <li><strong>GAMT</strong> (liver/pancreas): SAM + GAA → SAH + <strong>Creatine</strong></li>
                    <li><strong>SLC6A8</strong> (ubiquitous): Creatine transported into muscle &amp; brain (Na⁺/Cl⁻ dependent)</li>
                    <li><strong>Creatine kinase</strong>: Creatine + ATP ⇌ Phosphocreatine + ADP (energy buffer)</li>
                    <li>Phosphocreatine regenerates ATP in milliseconds during synaptic burst firing</li>
                  </ol>
                  <div className="alert alert-success py-1 mt-2" style={{ fontSize: 12 }}>
                    <strong>AGAT reaction type:</strong> Transamidination (NOT methylation). AGAT does NOT use SAM — it transfers the amidino group from arginine to glycine. Active site Cys performs the reaction.
                  </div>
                </div>
                <div className="col-md-6">
                  <h6 className="fw-bold" style={{ color: ACCENT2 }}>AGAT LOF — Single Pathology</h6>
                  <ol style={{ fontSize: 13 }}>
                    <li style={{ color: ACCENT2 }}>L-Arg + Gly → × AGAT BLOCKED → <strong>GAA = ABSENT</strong> (&lt;0.5 µmol/L)</li>
                    <li>GAMT (Step 2) has NO substrate — cannot make creatine</li>
                    <li style={{ color: ACCENT2 }}>Plasma creatine <strong>ABSENT</strong> (&lt;5 µmol/L; normal 20–80)</li>
                    <li style={{ color: ACCENT2 }}>Brain H-MRS: creatine peak (3.0 ppm) <strong>ABSENT</strong></li>
                    <li style={{ color: ACCENT }}>No GAA accumulation — <strong>NO GAA neurotoxicity</strong> (unlike GAMT)</li>
                    <li style={{ color: ACCENT }}>Seizures LESS drug-resistant than GAMT</li>
                    <li style={{ color: ACCENT4 }}>SLC6A8 (Step 3) intact → exogenous creatine IS transported normally</li>
                  </ol>
                  <div className="alert alert-info py-1 mt-2" style={{ fontSize: 12 }}>
                    <strong>Treatment:</strong> Creatine monohydrate ONLY (Level A). No ornithine needed (no GAA). Unlike SLC6A8 deficiency — creatine supplementation IS effective here.
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
                  AGAT — First Step of Creatine Biosynthesis (Transamidination)
                </div>
                <div className="card-body" style={{ fontSize: 13 }}>
                  <p>{ov.function}</p>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT2, color: '#fff' }}>
                  Mechanism — Absent GAA/Creatine, Single Pathology
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
              Key Positive &amp; Negative Features — AGAT Diagnostic Signature
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
                  NBS Primary — Low GAA Flag (Lower-Limit Cut-Off Required)
                </div>
                <div className="card-body" style={{ fontSize: 13 }}>
                  <p>{ov.nbs_primary}</p>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT4, color: '#fff' }}>
                  NBS Secondary — Creatine Absent + Brain H-MRS Confirmation
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
                  color={ph.includes('Classic') ? ACCENT2 : ph.includes('Moderate') ? ACCENT3 : ACCENT}
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
                  <PctBar label="Seizures" pct={kpiPct.pct_seizures} color={ACCENT2} />
                  <PctBar label="Drug-Resistant Seizures (LESS than GAMT — no GAA toxicity)" pct={kpiPct.pct_drug_resistant} color={ACCENT2} />
                  <PctBar label="IDD (moderate-profound)" pct={kpiPct.pct_idd} color={ACCENT7} />
                  <PctBar label="Speech Absent / Minimal" pct={kpiPct.pct_speech_absent} color={ACCENT7} />
                  <PctBar label="Autism-Like Features" pct={kpiPct.pct_autism_like} color={ACCENT5} />
                  <PctBar label="Hypotonia" pct={kpiPct.pct_hypotonia} color={ACCENT5} />
                  <PctBar label="NBS Detected (expanded NBS — low GAA flag)" pct={kpiPct.pct_nbs} color={ACCENT4} />
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
                      <tr><th>Biomarker</th><th>Mild</th><th>Moderate</th><th>Classic</th></tr>
                    </thead>
                    <tbody>
                      <tr style={{ background: '#e0f2f1' }}>
                        <td className="fw-bold">GAA (µmol/L) ↓↓</td>
                        <td>0.3–1.2</td>
                        <td>0.1–0.8</td>
                        <td style={{ color: ACCENT3, fontWeight: 'bold' }}>&lt;0.5 — ABSENT / PATHOGNOMONIC</td>
                      </tr>
                      <tr style={{ background: '#ffebee' }}>
                        <td className="fw-bold">Creatine (µmol/L) ↓↓↓</td>
                        <td>5–20</td>
                        <td>2–10</td>
                        <td style={{ color: ACCENT2, fontWeight: 'bold' }}>&lt;3.5 — ABSENT</td>
                      </tr>
                      <tr>
                        <td>Creatinine (µmol/L) — LOW</td>
                        <td>30–70</td>
                        <td>15–45</td>
                        <td>5–25 — CRITICALLY LOW</td>
                      </tr>
                      <tr style={{ background: '#e8f5e9' }}>
                        <td className="fw-bold">Methionine (µmol/L) — NORMAL ✓</td>
                        <td colSpan={3} className="text-center" style={{ color: ACCENT5 }}>18–42 (NORMAL in ALL phenotypes — KEY NEGATIVE vs all HHcy)</td>
                      </tr>
                      <tr style={{ background: '#e8f5e9' }}>
                        <td className="fw-bold">tHcy (µmol/L) — NORMAL ✓</td>
                        <td colSpan={3} className="text-center" style={{ color: ACCENT5 }}>5–14 (NORMAL in ALL — distinguishes from CBS/MTHFR/cblE/cblG)</td>
                      </tr>
                      <tr style={{ background: '#e8f5e9' }}>
                        <td className="fw-bold">SAM (µmol/L) — NORMAL ✓</td>
                        <td colSpan={3} className="text-center" style={{ color: ACCENT5 }}>60–125 (NORMAL — AGAT not SAM-dependent, KEY NEGATIVE vs GNMT)</td>
                      </tr>
                      <tr>
                        <td>CK (U/L)</td>
                        <td>25–80</td>
                        <td>35–120</td>
                        <td>50–180 (mild creatine-deficiency)</td>
                      </tr>
                      <tr>
                        <td>Brain H-MRS</td>
                        <td colSpan={3} className="text-center fw-bold" style={{ color: ACCENT2 }}>
                          Creatine peak (3.0 ppm) ABSENT — ALL phenotypes — PATHOGNOMONIC
                        </td>
                      </tr>
                      <tr>
                        <td>MMA</td>
                        <td colSpan={3} className="text-center">NORMAL (all)</td>
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
              GATM Variant Distribution (Cohort N={br.cohort_n})
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
              Patient Sample — First 12 of 40 (seed 117)
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover table-bordered mb-0" style={{ fontSize: 11 }}>
                  <thead className="table-dark">
                    <tr>
                      <th>ID</th><th>Phenotype</th><th>Variant</th>
                      <th>GAA (µmol/L) ↓</th><th>Creatine</th>
                      <th>Creatinine</th><th>Met</th>
                      <th>tHcy</th><th>CK</th>
                      <th>Sz</th><th>Drug-R</th><th>NBS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(br.patient_sample || []).map(p => (
                      <tr key={p.id}>
                        <td>{p.id}</td>
                        <td style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.phenotype}</td>
                        <td>{p.variant}</td>
                        <td style={{ fontWeight: 'bold', color: ACCENT3 }}>{p.gaa}</td>
                        <td style={{ fontWeight: 'bold', color: ACCENT2 }}>{p.creatine}</td>
                        <td>{p.creatinine}</td>
                        <td style={{ color: ACCENT5 }}>{p.methionine}</td>
                        <td style={{ color: ACCENT5 }}>{p.homocysteine}</td>
                        <td>{p.ck}</td>
                        <td>{p.seizures ? '✓' : '—'}</td>
                        <td style={{ color: p.drug_resistant ? ACCENT2 : 'inherit', fontWeight: p.drug_resistant ? 'bold' : 'normal' }}>{p.drug_resistant ? 'R' : '—'}</td>
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
                  <div className="alert alert-success mt-2 py-2" style={{ fontSize: 12 }}>
                    <strong>Better AED response than GAMT:</strong> ~25–35% drug-resistant (vs GAMT 60–80%).
                    No GAA neurotoxicity → AEDs are more effective once creatine is supplemented.
                    Creatine monohydrate + LEV achieves seizure freedom in most AGAT patients.
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
                      <tr><th>Feature</th><th>Mild (10%)</th><th>Moderate (35%)</th><th>Classic (55%)</th></tr>
                    </thead>
                    <tbody>
                      <tr><td>Seizures</td><td>15%</td><td>45%</td><td>70%</td></tr>
                      <tr><td>Drug-Resistant Sz</td><td>6%</td><td>22%</td><td>38%</td></tr>
                      <tr><td>IDD (moderate-profound)</td><td>50%</td><td>90%</td><td>99%</td></tr>
                      <tr><td>Speech Absent</td><td>10%</td><td>35%</td><td>65%</td></tr>
                      <tr><td>Autism-Like</td><td>10%</td><td>25%</td><td>42%</td></tr>
                      <tr><td>Hypotonia</td><td>20%</td><td>45%</td><td>65%</td></tr>
                      <tr><td>GAA (avg, µmol/L)</td><td>~0.7</td><td>~0.35</td><td>~0.15</td></tr>
                      <tr><td>Methionine</td><td colSpan={3} className="text-center" style={{ color: ACCENT5 }}>NORMAL in ALL phenotypes ✓</td></tr>
                      <tr><td>Creatine (H-MRS)</td><td colSpan={3} className="text-center" style={{ color: ACCENT2 }}>ABSENT in ALL phenotypes ✗</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Metabolic triggers */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT8, color: '#fff' }}>
              Metabolic Triggers &amp; Risk Factors — AGAT Safety Profile
            </div>
            <div className="card-body">
              {(br.metabolic_triggers || []).map((t, i) => (
                <div key={i} className="card mb-2 border-warning">
                  <div className="card-body py-2">
                    <div className="d-flex justify-content-between align-items-start mb-1">
                      <span className="fw-bold" style={{ fontSize: 13 }}>{t.trigger}</span>
                      <span className="badge ms-2" style={{
                        background: t.pct >= 85 ? ACCENT8 : t.pct >= 65 ? ACCENT2 : ACCENT,
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
                    style={{
                      background: t.level.includes('Level A') ? ACCENT4
                                : t.level.includes('Level B') ? ACCENT
                                : t.level.includes('NOT') ? ACCENT5
                                : ACCENT5,
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
            <div className="card-header fw-bold" style={{ background: ACCENT8, color: '#fff' }}>
              Drug Risk Summary — AGAT Deficiency
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Drug / Supplement</th><th>Risk Level</th><th>Mechanism</th></tr>
                </thead>
                <tbody>
                  {(br.drug_risks || []).map((d, i) => (
                    <tr key={i} style={{
                      background: d.risk.includes('HIGH') ? '#fff3e0' :
                                  d.risk.includes('MODERATE') ? '#fffde7' :
                                  d.risk.includes('NOT') ? '#f1f8e9' : '#f9fbe7'
                    }}>
                      <td className="fw-bold">{d.drug}</td>
                      <td style={{
                        color: d.risk.includes('HIGH') ? ACCENT8
                              : d.risk.includes('MODERATE') ? ACCENT3
                              : ACCENT5,
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
              <div className="card-header fw-bold" style={{ background: ACCENT4, color: '#fff' }}>
                Treatment Summary
              </div>
              <div className="card-body">
                <div className="mb-2"><strong>First-line regimen:</strong> {df.treatment_summary.first_line}</div>
                {(df.treatment_summary.absolute_ci || []).length > 0 && (
                  <div className="mb-2">
                    <strong>Absolute CI:</strong>{' '}
                    {(df.treatment_summary.absolute_ci || []).map((ci, i) => (
                      <span key={i} className="badge me-1 mb-1" style={{ background: '#b71c1c', color: '#fff', fontSize: 11 }}>⛔ {ci}</span>
                    ))}
                  </div>
                )}
                <div className="mb-2">
                  <strong>High Risk / Avoid:</strong>{' '}
                  {(df.treatment_summary.high_risk || []).map((h, i) => (
                    <span key={i} className="badge me-1 mb-1" style={{ background: ACCENT8, color: '#fff', fontSize: 11 }}>⚠ {h}</span>
                  ))}
                </div>
                <div>
                  <strong>Not indicated:</strong>{' '}
                  {(df.treatment_summary.not_indicated || []).map((ni, i) => (
                    <span key={i} className="badge me-1 mb-1" style={{ background: ACCENT5, color: '#fff', fontSize: 11 }}>{ni}</span>
                  ))}
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
            <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>
              Gene Card — GATM (AGAT Deficiency / CCDS3)
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
              Key Concepts — AGAT Biology, Pathophysiology &amp; CCDS3 Framework
            </div>
            <div className="card-body">
              {(df.key_concepts || []).map((c, i) => (
                <div key={i} className="mb-3 p-3 rounded border" style={{ background: i % 2 === 0 ? '#e0f2f1' : '#e8f5e9' }}>
                  <div className="fw-bold mb-1" style={{ color: ACCENT3 }}>{c.term}</div>
                  <div style={{ fontSize: 13 }}>{c.definition}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Differential diagnosis */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT2, color: '#fff' }}>
              Differential Diagnosis — AGAT vs Related Disorders
            </div>
            <div className="card-body">
              {(df.differential_diagnosis || []).map((d, i) => (
                <div key={i} className="mb-2 p-3 rounded border-start border-4" style={{ borderColor: ACCENT2, background: '#fff8f0' }}>
                  <div className="fw-bold mb-1">{d.disorder}</div>
                  <div style={{ fontSize: 13 }}>{d.key_distinction}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Epidemiology */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT5, color: '#fff' }}>
              Epidemiology, Rarity &amp; NBS Implications
            </div>
            <div className="card-body" style={{ fontSize: 13 }}>
              <p>{ov.prevalence}</p>
              <p>
                <strong>NBS challenge — the low-GAA problem:</strong> Unlike GAMT where GAA is
                dramatically elevated (50–300 µmol/L; easy to detect by upper-limit cut-off),
                AGAT deficiency produces a GAA below the normal lower reference range
                (&lt;0.5 µmol/L vs normal 0.5–3 µmol/L). Most NBS programs set upper-limit
                cut-offs for GAA (catching GAMT) but lack lower-limit GAA cut-offs.
                Urine creatine/creatinine ratio (very low in AGAT) is an accessible secondary screen.
                Programs must explicitly add lower-limit GAA flags to their GAMT-expanded panels.
              </p>
              <p>
                <strong>Treatment window:</strong> Pre-symptomatic creatine supplementation
                (started at NBS detection before IDD/seizures) can prevent or substantially
                reduce neurodevelopmental impact. AGAT has a better treatment prognosis than GAMT
                because there is no GAA neurotoxicity — creatine alone is curative of the
                metabolic defect. Late treatment reduces severity but cannot fully reverse
                established neurological damage from prolonged creatine deficiency.
              </p>
              <p>
                <strong>AGAT vs GAMT prognosis:</strong> AGAT patients treated early often
                achieve better outcomes than GAMT — the single-pathology mechanism (no GAA toxicity)
                means creatine replacement can fully address the biochemical deficit.
                GAMT patients, even on creatine + ornithine, carry the residual burden of
                pre-treatment GAA neurotoxicity; AGAT patients do not.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Footer nav */}
      <div className="d-flex justify-content-between mt-4">
        <Link className="btn btn-outline-secondary btn-sm" href="/gamt">← GAMT</Link>
        <Link className="btn btn-outline-secondary btn-sm" href="/ahcy">AHCY →</Link>
      </div>
    </div>
  );
}

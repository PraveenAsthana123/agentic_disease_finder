'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — AHCY / methionine cycle
const ACCENT2 = '#b71c1c';   // deep red — SAH elevated / severe myopathy
const ACCENT3 = '#4a148c';   // deep purple — SAM/SAH ratio / methylation failure
const ACCENT4 = '#1565c0';   // blue — Level A treatment
const ACCENT5 = '#37474f';   // slate — KEY NEGATIVES / normal values
const ACCENT6 = '#1b5e20';   // dark green — MeCbl NORMAL / cobalamin intact
const ACCENT7 = '#e65100';   // deep orange — cardiomyopathy / hepatomegaly
const ACCENT8 = '#006064';   // teal — methionine restriction / adenosine

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

export default function AHCYPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [br, setBr]       = useState(null);
  const [df, setDf]       = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ahcy/overview`).then(r => r.json()),
      fetch(`${API}/api/ahcy/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ahcy/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBr(b); setDf(d); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <div className="alert alert-danger m-4">{error}</div>;
  if (!ov)   return <div className="text-center mt-5"><div className="spinner-border" /></div>;

  const kpi = ov.kpis || {};
  const kpiPcts = br?.kpi_pcts || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT3} 100%)` }}>
        <div className="d-flex justify-content-between align-items-start flex-wrap gap-2">
          <div>
            <h4 className="mb-1 fw-bold">🧬 AHCY Epilepsy Dashboard</h4>
            <div style={{ fontSize: 13, opacity: 0.9 }}>
              Adenosylhomocysteinase Deficiency — Hypermethioninemia with Pathognomonic SAH Elevation
            </div>
            <div style={{ fontSize: 12, opacity: 0.8 }}>
              AHCY-432aa · NAD⁺-Homotetrameric · 20q11.22 · Autosomal Recessive ·
              SAH → Adenosine + Hcy (ONLY SAH-clearance enzyme) ·
              OMIM *180960 / #613752
            </div>
          </div>
          <div className="text-end">
            <span className="badge bg-light text-dark me-1">n={ov.cohort_n}</span>
            <span className="badge bg-warning text-dark me-1">Myopathy {kpiPcts.myopathy ?? kpi.pct_myopathy}%</span>
            <span className="badge bg-danger me-1">Cardio {kpiPcts.cardiomyopathy ?? kpi.pct_cardiomyopathy}%</span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert variant="danger" text="🚨 SAM SUPPLEMENTS ABSOLUTELY CONTRAINDICATED — SAM → SAH (stalled) → catastrophic methylation crisis. Avoid ALL SAM-containing products." />
      <Alert variant="warning" text="⚠️ Betaine (TMG) HIGH RISK in AHCY — Hcy → methionine (BHMT) → more SAM → more SAH → worsens. Use only with close SAH/SAM monitoring; many centers avoid entirely." />
      <Alert variant="info"    text="ℹ️ SAH NOT in standard NBS — SAH requires specialized HPLC for diagnosis. NBS detects AHCY only via elevated methionine (200–600 µmol/L)." />

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ── */}
      {tab === 0 && (
        <div>
          {/* KPI row 1 — biomarkers */}
          <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Cohort Biomarker KPIs (n={ov.cohort_n})</h6>
          <div className="row g-2 mb-3">
            <KPI label="Avg tHcy (µmol/L)" value={kpi.avg_homocysteine_umol_l} color={ACCENT2} />
            <KPI label="Avg Methionine (µmol/L)" value={kpi.avg_methionine_umol_l} color={ACCENT2} />
            <KPI label="Avg SAH (arb units)" value={kpi.avg_sah_arbitrary_units} color={ACCENT3} />
            <KPI label="Avg SAM/SAH Ratio" value={kpi.avg_sam_sah_ratio} color={ACCENT3} />
            <KPI label="Avg CK (U/L)" value={kpi.avg_ck_u_l?.toLocaleString()} color={ACCENT7} />
            <KPI label="Avg AST (U/L)" value={kpi.avg_ast_u_l} color={ACCENT7} />
          </div>
          {/* KPI row 2 — clinical */}
          <div className="row g-2 mb-4">
            <KPI label="Myopathy" value={`${kpi.pct_myopathy}%`} color={ACCENT2} />
            <KPI label="Cardiomyopathy" value={`${kpi.pct_cardiomyopathy}%`} color={ACCENT7} />
            <KPI label="Hepatomegaly" value={`${kpi.pct_hepatomegaly}%`} color={ACCENT7} />
            <KPI label="IDD" value={`${kpi.pct_idd}%`} color={ACCENT3} />
            <KPI label="Seizures" value={`${kpi.pct_seizures}%`} color={ACCENT} />
            <KPI label="NBS Detected" value={`${kpi.pct_nbs_detected}%`} color={ACCENT4} />
          </div>

          {/* Phenotype distribution */}
          <div className="row mb-4">
            <div className="col-md-6">
              <h6 className="fw-bold" style={{ color: ACCENT }}>Phenotypic Class Distribution</h6>
              {Object.entries(ov.phenotype_distribution || {}).map(([ph, cnt]) => (
                <PctBar
                  key={ph}
                  label={ph}
                  pct={Math.round(cnt / ov.cohort_n * 100)}
                  color={ph.includes('Severe') ? ACCENT2 : ph.includes('Classic') ? ACCENT : ACCENT6}
                />
              ))}
            </div>
            <div className="col-md-6">
              <h6 className="fw-bold" style={{ color: ACCENT2 }}>KEY METABOLIC FINGERPRINT</h6>
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Biomarker</th><th>AHCY</th><th>CBS (comparison)</th></tr>
                </thead>
                <tbody>
                  <tr><td>SAH</td><td className="text-danger fw-bold">↑↑↑ PATHOGNOMONIC</td><td>NORMAL</td></tr>
                  <tr><td>SAM</td><td className="text-danger fw-bold">↑↑ ELEVATED</td><td>Normal-high</td></tr>
                  <tr><td>SAM/SAH</td><td className="text-danger fw-bold">≪0.5 (↓↓↓)</td><td>Normal</td></tr>
                  <tr><td>Methionine</td><td className="text-danger fw-bold">200–600 µmol/L</td><td>60–500 µmol/L</td></tr>
                  <tr><td>tHcy</td><td className="text-warning fw-bold">40–150 (moderate)</td><td>100–500 (HIGHEST)</td></tr>
                  <tr><td>MMA</td><td className="text-success fw-bold">NORMAL ✓</td><td>NORMAL ✓</td></tr>
                  <tr><td>MeCbl</td><td className="text-success fw-bold">NORMAL ✓</td><td>NORMAL ✓</td></tr>
                  <tr><td>Ectopia lentis</td><td className="text-success fw-bold">ABSENT ✓</td><td className="text-danger">90% PATHOGN.</td></tr>
                  <tr><td>Myopathy</td><td className="text-danger fw-bold">85–90% HALLMARK</td><td className="text-success">ABSENT ✓</td></tr>
                  <tr><td>Cardiomyopathy</td><td className="text-danger fw-bold">60–70%</td><td className="text-success">ABSENT ✓</td></tr>
                  <tr><td>B6 response</td><td className="text-success fw-bold">ABSENT (NAD+, not PLP)</td><td>50% UNIQUE</td></tr>
                  <tr><td>NBS (methionine)</td><td>~70%</td><td>~60%</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Pathway summary */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT, color: 'white' }}>
              Methionine Cycle — AHCY Checkpoint
            </div>
            <div className="card-body">
              <pre style={{ fontSize: 12, background: '#f8f9fa', padding: 12, borderRadius: 6, whiteSpace: 'pre-wrap' }}>
{`  Methionine → SAM (via MAT1A/MAT2A)
                 │
           Methyltransferases (DNMT, COMT, HNMT, GAMT, PEMT, PRMT…)
           SAM donates CH₃ → ALL biological methylation
                 │
             SAH ← CHECKPOINT — product of ALL methylation reactions
                 │
          AHCY ← ONLY enzyme to hydrolyze SAH
           ↙ (AHCY LOF → SAH ACCUMULATES)
    ┌──────┴───────────┐
Adenosine          Homocysteine (Hcy)
(product          ┌──────────────────────┐
 deficient)       │ CBS → cystathionine  │
                  │ MTR → methionine     │
                  └──────────────────────┘

SAH ↑↑↑ → inhibits ALL methyltransferases → global methylation crisis
SAM ↑ → methionine ↑↑↑ → NBS detects via hypermethioninemia
SAM/SAH ratio ≪0.5 → ALL methylation reactions simultaneously impaired`}
              </pre>
            </div>
          </div>

          {/* Gene / disease summary */}
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT3, color: 'white' }}>Disease Summary</div>
                <div className="card-body small">
                  <p><strong>Gene:</strong> {ov.gene} (Adenosylhomocysteinase)</p>
                  <p><strong>Disease:</strong> {ov.disease_name}</p>
                  <p><strong>Protein:</strong> {ov.protein_size}</p>
                  <p><strong>Chromosome:</strong> {ov.chromosome} · {ov.inheritance}</p>
                  <p><strong>OMIM Gene:</strong> {ov.omim_gene} · <strong>Disease:</strong> {ov.omim_disease}</p>
                  <p><strong>Prevalence:</strong> {ov.prevalence}</p>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT4, color: 'white' }}>NBS Detection</div>
                <div className="card-body small">
                  <p><strong>Primary:</strong> {ov.nbs_primary}</p>
                  <p><strong>Confirmatory:</strong> {ov.nbs_secondary}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 1: Patients & Biomarkers ── */}
      {tab === 1 && br && (
        <div>
          <div className="row mb-4">
            <div className="col-md-6">
              <h6 className="fw-bold" style={{ color: ACCENT }}>Clinical Feature Prevalence</h6>
              <PctBar label="Myopathy (proximal; CK ↑↑↑)"              pct={kpiPcts.myopathy}       color={ACCENT2} />
              <PctBar label="Hepatomegaly / Liver disease"              pct={kpiPcts.hepatomegaly}   color={ACCENT7} />
              <PctBar label="IDD (intellectual disability)"             pct={kpiPcts.idd}            color={ACCENT3} />
              <PctBar label="Cardiomyopathy (hypertrophic/dilated)"     pct={kpiPcts.cardiomyopathy} color={ACCENT7} />
              <PctBar label="NBS detected (methionine)"                 pct={kpiPcts.nbs_detected}   color={ACCENT4} />
              <PctBar label="Seizures (focal/GTCS/IS/myoclonic)"        pct={kpiPcts.seizures}       color={ACCENT} />
              <PctBar label="Facial dysmorphism"                        pct={kpiPcts.dysmorphic}     color={ACCENT5} />
            </div>
            <div className="col-md-6">
              <h6 className="fw-bold" style={{ color: ACCENT2 }}>Biomarker Reference Ranges</h6>
              {br.biomarker_ranges && Object.entries(br.biomarker_ranges).map(([key, val]) => (
                <div key={key} className="mb-2 p-2 rounded" style={{ background: '#f8f9fa', fontSize: 12 }}>
                  <span className="fw-bold text-capitalize">{key.replace(/_/g, ' ')}: </span>
                  {typeof val === 'string' ? val :
                    Object.entries(val).map(([k, v]) => `${k}: ${v}`).join(' | ')}
                </div>
              ))}
            </div>
          </div>

          <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Patient Sample (first 12)</h6>
          <div className="table-responsive">
            <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
              <thead className="table-dark">
                <tr>
                  <th>ID</th><th>Phenotype</th><th>Variant</th><th>Onset (mo)</th>
                  <th>tHcy</th><th>Methionine</th><th>SAH</th><th>SAM/SAH</th>
                  <th>CK</th><th>Myopathy</th><th>Cardio</th><th>Liver</th><th>Sz</th><th>NBS</th>
                </tr>
              </thead>
              <tbody>
                {(br.patient_sample || []).map(p => (
                  <tr key={p.id}>
                    <td>{p.id}</td>
                    <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                        title={p.phenotype}>{p.phenotype.split(' (')[0]}</td>
                    <td><code>{p.variant}</code></td>
                    <td>{p.age_onset_months}</td>
                    <td><span className={p.homocysteine_umol_l > 80 ? 'text-danger fw-bold' : 'text-warning'}>{p.homocysteine_umol_l}</span></td>
                    <td><span className="text-danger fw-bold">{p.methionine_umol_l}</span></td>
                    <td><span className="text-danger fw-bold">{p.sah_arbitrary_units}</span></td>
                    <td><span className={p.sam_sah_ratio < 1 ? 'text-danger fw-bold' : 'text-warning'}>{p.sam_sah_ratio}</span></td>
                    <td><span className={p.ck_u_l > 500 ? 'text-danger fw-bold' : ''}>{p.ck_u_l?.toLocaleString()}</span></td>
                    <td>{p.myopathy ? '✓' : '—'}</td>
                    <td>{p.cardiomyopathy ? '✓' : '—'}</td>
                    <td>{p.hepatomegaly ? '✓' : '—'}</td>
                    <td>{p.seizures ? '✓' : '—'}</td>
                    <td>{p.nbs_detected ? '✓' : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h6 className="fw-bold mt-3 mb-2" style={{ color: ACCENT3 }}>Genetic Variants</h6>
          <div className="table-responsive">
            <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
              <thead className="table-dark">
                <tr><th>Variant</th><th>Domain</th><th>Prevalence</th><th>Severity</th></tr>
              </thead>
              <tbody>
                {(br.variant_breakdown || []).map((v, i) => (
                  <tr key={i}>
                    <td><code>{v.variant}</code></td>
                    <td>{v.domain}</td>
                    <td>{v.prevalence}</td>
                    <td>{v.severity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Tab 2: Seizures & Triggers ── */}
      {tab === 2 && br && (
        <div>
          <div className="row mb-4">
            <div className="col-md-5">
              <h6 className="fw-bold" style={{ color: ACCENT }}>Seizure Type Distribution</h6>
              {(br.seizure_types || []).map((s, i) => (
                <PctBar key={i} label={s.type} pct={s.pct}
                  color={i === 0 ? ACCENT2 : i === 1 ? ACCENT3 : i === 2 ? ACCENT7 : ACCENT} />
              ))}
              <div className="alert alert-secondary mt-2" style={{ fontSize: 12 }}>
                Seizures present in {kpiPcts.seizures}% of cohort — LESS PROMINENT than myopathy/cardiomyopathy.
                Mechanism: neurotransmitter methylation failure (COMT/HNMT impaired) +
                HHcy-mediated NMDA excitotoxicity + cerebral white matter disease.
              </div>
            </div>
            <div className="col-md-7">
              <h6 className="fw-bold" style={{ color: ACCENT2 }}>Metabolic Triggers / Risk Factors</h6>
              {(br.metabolic_triggers || []).map((t, i) => (
                <div key={i} className="mb-2 p-2 rounded border" style={{ fontSize: 12 }}>
                  <div className="d-flex justify-content-between mb-1">
                    <span className="fw-bold">{t.trigger}</span>
                    <span className="badge" style={{ backgroundColor: ACCENT2 }}>{t.pct}%</span>
                  </div>
                  <div className="text-muted">{t.mechanism}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 3: Treatments ── */}
      {tab === 3 && br && (
        <div>
          <Alert variant="danger" text="🚨 SAM ABSOLUTELY CONTRAINDICATED — SAM supplements directly worsen SAH accumulation." />
          <Alert variant="warning" text="⚠️ BETAINE HIGH RISK — BHMT-driven methionine → SAM → more SAH. Use cautiously or avoid entirely." />
          <Alert variant="success" text="✅ First-line AED: Levetiracetam — no impact on methionine/SAH pathway; safe with liver disease." />

          <h6 className="fw-bold mb-3" style={{ color: ACCENT4 }}>Treatment Protocols</h6>
          {(br.treatments || []).map((t, i) => (
            <div key={i} className="card mb-3">
              <div className="card-header d-flex justify-content-between align-items-center py-2">
                <span className="fw-bold" style={{ fontSize: 14 }}>{t.treatment}</span>
                <span className="badge" style={{
                  backgroundColor: t.level.includes('Level A') ? '#1b5e20' : t.level.includes('Level B') ? '#1565c0' : '#37474f'
                }}>{t.level}</span>
              </div>
              <div className="card-body py-2 small text-muted">{t.mechanism}</div>
            </div>
          ))}

          <h6 className="fw-bold mb-3 mt-4" style={{ color: ACCENT2 }}>Drug Risks & Contraindications</h6>
          {(br.drug_risks || []).map((d, i) => (
            <div key={i} className="card mb-3 border-danger">
              <div className="card-header d-flex justify-content-between align-items-center py-2"
                   style={{ background: d.risk.includes('ABSOLUTE') ? '#ffebee' : '#fff8e1' }}>
                <span className="fw-bold" style={{ fontSize: 14 }}>{d.agent}</span>
                <span className="badge" style={{
                  backgroundColor: d.risk.includes('ABSOLUTE') ? '#b71c1c' : d.risk.includes('HIGH') ? '#e65100' : '#f57f17'
                }}>{d.risk.split(' —')[0]}</span>
              </div>
              <div className="card-body py-2 small text-muted">{d.mechanism}</div>
            </div>
          ))}
        </div>
      )}

      {/* ── Tab 4: Definitions ── */}
      {tab === 4 && df && (
        <div>
          {/* Gene card */}
          {df.gene_card && (
            <div className="card mb-4">
              <div className="card-header fw-bold" style={{ background: ACCENT, color: 'white' }}>Gene Card — AHCY</div>
              <div className="card-body">
                <div className="row g-2">
                  {Object.entries(df.gene_card).map(([k, v]) => (
                    <div key={k} className="col-12 col-md-6">
                      <div className="p-2 rounded" style={{ background: '#f8f9fa', fontSize: 12 }}>
                        <span className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}: </span>
                        <span className="text-muted">{v}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Key concepts */}
          <h6 className="fw-bold mb-3" style={{ color: ACCENT3 }}>Key Concepts</h6>
          {(df.key_concepts || []).map((c, i) => (
            <div key={i} className="card mb-3">
              <div className="card-header fw-bold py-2" style={{ background: '#e8eaf6', fontSize: 13 }}>
                {c.concept}
              </div>
              <div className="card-body py-2 small text-muted">{c.explanation}</div>
            </div>
          ))}

          {/* Differential */}
          <h6 className="fw-bold mb-3 mt-4" style={{ color: ACCENT2 }}>Differential Diagnosis</h6>
          {(df.differential_diagnosis || []).map((d, i) => (
            <div key={i} className="card mb-3">
              <div className="card-header fw-bold py-2" style={{ background: '#fce4ec', fontSize: 13 }}>
                {d.disease}
              </div>
              <div className="card-body py-2 small text-muted">{d.distinguishing}</div>
            </div>
          ))}
        </div>
      )}

      {/* Footer nav */}
      <div className="mt-4 pt-3 border-top text-center small text-muted">
        <Link href="/cbs" className="me-3">← CBS (Classical Homocystinuria)</Link>
        <Link href="/" className="me-3">Home</Link>
        <span>AHCY · OMIM *180960 / #613752 · 20q11.22 · AR · NAD⁺-dependent</span>
      </div>
    </div>
  );
}

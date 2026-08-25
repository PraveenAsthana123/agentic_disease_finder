'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// ASS1 color scheme — citrulline CRITICALLY HIGH / arginine conditionally essential / step 3 cytoplasmic
const ACCENT  = '#e65100';   // deep orange — citrulline VERY HIGH / PATHOGNOMONIC
const ACCENT2 = '#1a237e';   // deep navy — ammonia CRITICALLY HIGH / urea cycle block
const ACCENT3 = '#2e7d32';   // deep forest green — arginine PRIMARY therapy / conditionally essential
const ACCENT4 = '#4a148c';   // deep purple — cytoplasmic enzyme / step 3 / pathway context
const ACCENT5 = '#006064';   // teal — key negatives / orotic NORMAL (vs OTC) / differentials
const ACCENT6 = '#37474f';   // blue-grey — secondary features / variants
const ACCENT7 = '#b71c1c';   // deep crimson — VPA ABSOLUTE CI / drug risks
const ACCENT8 = '#880e4f';   // dark pink — ASL distinction / argininosuccinate absent

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

export default function ASS1Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/ass1/overview`).then(r => r.json()),
      fetch(`${API}/api/ass1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ass1/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading ASS1 dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}18, ${ACCENT4}18)`, borderLeft: `5px solid ${ACCENT}` }}>
        <div className="d-flex align-items-center gap-2 mb-1 flex-wrap">
          <span style={{ fontSize: 24 }}>🧪</span>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>ASS1 Deficiency — Citrullinemia Type 1 Dashboard</h4>
          <span className="badge" style={{ background: ACCENT4, fontSize: 11 }}>Urea Cycle Step 3 — Cytoplasmic</span>
          <span className="badge" style={{ background: ACCENT, fontSize: 11 }}>Citrulline CRITICALLY HIGH — Hallmark</span>
          <span className="badge" style={{ background: ACCENT3, fontSize: 11 }}>Arginine Conditionally Essential</span>
          <span className="badge" style={{ background: ACCENT5, fontSize: 11 }}>Orotic NORMAL (≠ OTC)</span>
        </div>
        <div className="small text-muted">{ov?.subtitle}</div>
        <div className="d-flex gap-3 mt-2 flex-wrap">
          {[
            ['Gene', ov?.gene],
            ['Chr', ov?.chromosome],
            ['Protein', ov?.protein_size?.split(';')[0]],
            ['OMIM Disease', ov?.omim_disease],
            ['Inheritance', 'Autosomal Recessive'],
            ['Cohort', `n=${ov?.cohort_n}`],
          ].map(([k, v]) => (
            <span key={k} className="badge bg-light text-dark border small">{k}: {v}</span>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div className="row g-2 mb-3">
            <KPI label="Avg Plasma Citrulline (µmol/L)" value={kpi.avg_plasma_citrulline_umol_l} color={ACCENT} />
            <KPI label="Avg Plasma Ammonia (µmol/L)" value={kpi.avg_plasma_ammonia_umol_l} color={ACCENT2} />
            <KPI label="Avg Arginine (µmol/L)" value={kpi.avg_plasma_arginine_umol_l} color={ACCENT3} />
            <KPI label="Avg Urine Orotic (µmol/mol Cr)" value={kpi.avg_urine_orotic_acid_umol_mol} color={ACCENT5} />
            <KPI label="Avg Ornithine (µmol/L)" value={kpi.avg_plasma_ornithine_umol_l} color={ACCENT6} />
            <KPI label="Avg Glutamine (µmol/L)" value={kpi.avg_plasma_glutamine_umol_l} color={ACCENT4} />
            <KPI label="Seizures %" value={`${kpi.pct_seizures}%`} color={ACCENT2} />
            <KPI label="Status Epilepticus %" value={`${kpi.pct_status_epilepticus}%`} color={ACCENT7} />
            <KPI label="DRE %" value={`${kpi.pct_dre}%`} color={ACCENT6} />
            <KPI label="IDD %" value={`${kpi.pct_idd}%`} color={ACCENT2} />
          </div>

          {/* Phenotype + pathway */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>Phenotype Distribution (n={ov?.cohort_n})</div>
                <div className="card-body">
                  {Object.entries(ov?.phenotype_distribution || {}).map(([ph, v]) => (
                    <PctBar
                      key={ph} label={`${ph} (n=${v.n})`} pct={v.pct}
                      color={ph.includes('Classic') ? ACCENT : ph.includes('Late') ? ACCENT2 : ACCENT3}
                    />
                  ))}
                  <div className="small text-muted mt-2">
                    <strong style={{ color: ACCENT }}>Classic Neonatal (50%): NULL ASS1</strong> — citrulline {'>'} 2000 µmol/L; ammonia {'>'} 1000; CRRT mandatory.<br/>
                    Late-Onset Partial (30%): residual ASS1; episodic crises; protein aversion clue.<br/>
                    Mild/NBS-Detected (20%): citrulline 150-500; often asymptomatic; monitoring essential.
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT4 }}>Pathway Role — ASS1 as Step 3 Cytoplasmic Condensation Enzyme</div>
                <div className="card-body small text-muted">
                  <div><strong style={{ color: ACCENT5 }}>Reaction:</strong> {def?.reaction}</div>
                  <div className="mt-2"><strong>ASS1 in the urea cycle:</strong></div>
                  <ul className="mt-1 mb-2" style={{ paddingLeft: 18 }}>
                    <li>NAGS: Glutamate + Acetyl-CoA → NAG [cofactor]</li>
                    <li>Step 1: NH₃ + CO₂ → Carbamoyl-P [CPS1] (mitochondrial)</li>
                    <li>Step 2: Carbamoyl-P + Ornithine → Citrulline [OTC] (mitochondrial)</li>
                    <li style={{ color: ACCENT, fontWeight: 700 }}>Step 3: Citrulline + Asp + ATP → Argininosuccinate [ASS1, BLOCKED] (cytoplasmic)</li>
                    <li>Step 4: Argininosuccinate → Arginine + Fumarate [ASL]</li>
                    <li>Step 5: Arginine → Ornithine + Urea [ARG1]</li>
                  </ul>
                  <div className="alert alert-warning py-1 px-2 small mb-2">
                    <strong>KEY: OTC works → citrulline produced continuously → CANNOT enter ASS1 → ACCUMULATES MASSIVELY</strong>
                  </div>
                  <div><strong style={{ color: ACCENT3 }}>Arginine axis:</strong> Not produced downstream → CONDITIONALLY ESSENTIAL → high-dose arginine = PRIMARY therapy</div>
                </div>
              </div>
            </div>
          </div>

          {/* Key biomarkers */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>🏆 Citrulline — The ASS1 Pathognomonic Hallmark</div>
                <div className="card-body small">
                  {Object.entries(ov?.key_positives || {}).map(([k, v]) => (
                    <div key={k} className="mb-2 p-2 rounded" style={{
                      background: k.includes('citrulline') ? '#fff3e0' : k.includes('ammonia') ? '#fce4ec' : '#e8f5e9'
                    }}>
                      <div className="fw-bold" style={{
                        color: k.includes('citrulline') ? ACCENT : k.includes('ammonia') ? ACCENT2 : ACCENT3
                      }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                      <div className="text-muted">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT5 }}>🟢 KEY NEGATIVE Biomarkers (NORMAL in ASS1)</div>
                <div className="card-body small">
                  {Object.entries(ov?.biomarker_normals || {}).map(([k, v]) => (
                    <div key={k} className="mb-1">
                      <span className="badge me-1" style={{
                        background: k.includes('orotic') ? ACCENT5 : k.includes('argininosucc') ? ACCENT8 : ACCENT3,
                        fontSize: 10
                      }}>
                        {k.includes('orotic') ? '★ NORMAL/MILD (≠ OTC)' : k.includes('argininosucc') ? 'ABSENT' : 'NORMAL'}
                      </span>
                      <strong>{k.replace(/_/g, ' ').replace('NORMAL', '').replace('ABSENT', '').trim()}: </strong>
                      <span className="text-muted">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* ASS1 vs OTC/CPS1/ASL alerts */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="alert alert-warning border-start border-5 border-warning mb-0">
                <div className="fw-bold mb-1">🟠 ASS1 vs OTC: Citrulline REVERSAL</div>
                <div className="small">
                  <strong>ASS1: citrulline VERY HIGH ({'>'} 500 µmol/L)</strong> — OTC works, produces citrulline, ASS1 cannot consume it.<br/>
                  <strong>OTC: citrulline CRITICALLY LOW ({'<'} 5 µmol/L)</strong> — OTC blocked, citrulline CANNOT be made.<br/>
                  OPPOSITE DIRECTION — single citrulline level completely differentiates. Orotic HIGH in OTC; NORMAL in ASS1.
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="alert alert-info border-start border-5 border-info mb-0">
                <div className="fw-bold mb-1">🔵 ASS1 vs ASL: Argininosuccinate is the Key</div>
                <div className="small">
                  <strong>Both: citrulline HIGH, arginine LOW, AR.</strong><br/>
                  <strong>ASL:</strong> argininosuccinate VERY HIGH in urine/plasma (ASL cannot cleave it).<br/>
                  <strong>ASS1:</strong> argininosuccinate <em>ABSENT</em> — cannot be made (block is BEFORE ASL step).<br/>
                  Urine argininosuccinate = ASL, not ASS1. Gene panel confirms.
                </div>
              </div>
            </div>
          </div>

          {/* Arginine + VPA */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${ACCENT3}` }}>
                <div className="card-header fw-bold small" style={{ color: ACCENT3 }}>Arginine — Conditionally Essential in ASS1 (PRIMARY Therapy)</div>
                <div className="card-body small text-muted">
                  {def?.arginine_conditional_essential}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="alert alert-danger border-start border-5 border-danger mb-0">
                <div className="fw-bold mb-1">⚠️ ABSOLUTE CI: VPA — Doubly Blocks Urea Cycle</div>
                <div className="small">
                  VPA inhibits NAGS → CPS1 INACTIVE (step 1 block).<br/>
                  <strong>In ASS1: existing block at step 3. VPA adds complete block at step 1 → DOUBLY BLOCKED.</strong><br/>
                  Catastrophic hyperammonemia. Multiple fatalities. ABSOLUTE CI in all UCD.<br/>
                  <strong>Use LEV, LTG, CLB instead.</strong>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 1: PATIENTS & BIOMARKERS ── */}
      {tab === 1 && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>Phenotype Detail</div>
                <div className="card-body">
                  {Object.entries(bd?.phenotype_detail || {}).map(([ph, v]) => (
                    <div key={ph} className="mb-3 p-2 rounded" style={{ background: '#fafafa', border: '1px solid #eee' }}>
                      <div className="fw-bold" style={{ color: ph.includes('Classic') ? ACCENT : ph.includes('Late') ? ACCENT2 : ACCENT3 }}>
                        {ph} — {v.pct}% (n={v.n})
                      </div>
                      <div className="small text-muted mt-1">{v.description}</div>
                      <div className="small mt-1">
                        <span className="badge me-2" style={{ background: ACCENT, color: '#fff' }}>Peak NH₃ avg: {v.avg_peak_ammonia_umol_l} µmol/L</span>
                        <span className="badge me-2" style={{ background: ACCENT2, color: '#fff' }}>Citrulline avg: {v.avg_plasma_citrulline_umol_l} µmol/L</span>
                        <span className="badge" style={{ background: ACCENT3, color: '#fff' }}>Seizures: {v.seizure_rate_pct}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT4 }}>Variant Distribution (ASS1 gene, 9q34.11)</div>
                <div className="card-body">
                  {(bd?.variants || []).map(v => (
                    <div key={v.name} className="mb-2 p-2 rounded" style={{
                      background: v.ncg_analogue ? '#e8f5e9' : '#fff3e0'
                    }}>
                      <div className="d-flex justify-content-between">
                        <strong style={{ color: v.ncg_analogue ? ACCENT3 : ACCENT }}>{v.name}</strong>
                        <span className="badge" style={{ background: v.ncg_analogue ? ACCENT3 : ACCENT }}>{v.pct}%</span>
                      </div>
                      <div className="small text-muted">{v.domain}</div>
                      <div className="small text-muted">{v.severity}</div>
                    </div>
                  ))}
                  <div className="small text-muted mt-2">
                    <span style={{ color: ACCENT3 }}>Green</span> = attenuated (partial residual function); <span style={{ color: ACCENT }}>Orange</span> = null/classic severe.<br/>
                    AR inheritance: biallelic LOF; males and females equally affected.
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Differentials */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT5 }}>Differential Diagnoses</div>
            <div className="card-body">
              <div className="row g-2">
                {(bd?.differentials || []).map(d => (
                  <div key={d.disease} className="col-md-6">
                    <div className="p-2 rounded" style={{ background: '#f5f5f5', borderLeft: `3px solid ${ACCENT5}` }}>
                      <div className="fw-bold small" style={{ color: ACCENT5 }}>{d.disease}</div>
                      <div className="small text-muted">{d.key_diff}</div>
                      <div className="small mt-1"><strong>Key test:</strong> {d.distinguishing}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: SEIZURES & TREATMENTS ── */}
      {tab === 2 && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT2 }}>Seizure Types (of those with seizures)</div>
                <div className="card-body">
                  {Object.entries(bd?.seizure_types || {}).map(([st, v]) => (
                    <PctBar key={st} label={`${st} (n≈${v.n})`} pct={v.pct}
                      color={st.includes('Status') ? ACCENT7 : st.includes('GTCS') ? ACCENT2 : ACCENT} />
                  ))}
                  <div className="small text-muted mt-2">
                    GTCS modal in acute hyperammonemic crisis.<br/>
                    Status epilepticus risk in classic neonatal especially severe.<br/>
                    <strong>Control ammonia FIRST (arginine + scavengers + CRRT); AED second. NEVER VPA.</strong>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>Crisis Triggers</div>
                <div className="card-body">
                  {Object.entries(bd?.triggers || {}).map(([t, v]) => (
                    <div key={t} className="mb-2">
                      <PctBar label={t} pct={v.pct} color={t.includes('Valproate') ? ACCENT7 : ACCENT} />
                      <div className="text-muted small ms-1" style={{ marginTop: -4 }}>{v.detail}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Treatments */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT3 }}>Treatment Ladder (High-Dose Arginine = PRIMARY for ASS1)</div>
            <div className="card-body">
              <div className="row g-2">
                {Object.entries(bd?.treatments || {}).map(([tx, v]) => (
                  <div key={tx} className="col-md-6">
                    <div className="p-2 rounded" style={{
                      background: tx.includes('Arginine') && tx.includes('High') ? '#e8f5e9' : v.level === 'A' ? '#f3e5f5' : '#e3f2fd',
                      borderLeft: `3px solid ${tx.includes('Arginine') && tx.includes('High') ? ACCENT3 : v.level === 'A' ? ACCENT4 : ACCENT2}`
                    }}>
                      <div className="d-flex justify-content-between">
                        <div className="fw-bold small">{tx}</div>
                        <span className="badge" style={{ background: tx.includes('Arginine') && tx.includes('High') ? ACCENT3 : v.level === 'A' ? ACCENT4 : ACCENT2 }}>Level {v.level}</span>
                      </div>
                      <div className="small text-muted">{v.category}</div>
                      <div className="small mt-1">{v.mechanism}</div>
                      {v.efficacy_pct && (
                        <div className="progress mt-1" style={{ height: 6 }}>
                          <div className="progress-bar" style={{ width: `${v.efficacy_pct}%`, backgroundColor: tx.includes('Arginine') && tx.includes('High') ? ACCENT3 : v.level === 'A' ? ACCENT4 : ACCENT2 }} />
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Drug risks */}
          <div className="card shadow-sm">
            <div className="card-header fw-bold small" style={{ color: ACCENT7 }}>Drug Risk Profile (VPA = Doubly Blocks Urea Cycle in ASS1)</div>
            <div className="card-body">
              <div className="row g-2">
                {Object.entries(bd?.drug_risks || {}).map(([drug, v]) => (
                  <div key={drug} className="col-md-6">
                    <div className="p-2 rounded" style={{
                      background: v.risk === 'ABSOLUTE CI' ? '#ffebee' : v.risk === 'HIGH RISK' ? '#fff3e0' : '#fff9c4',
                      borderLeft: `3px solid ${v.risk === 'ABSOLUTE CI' ? ACCENT7 : v.risk === 'HIGH RISK' ? ACCENT : ACCENT5}`
                    }}>
                      <div className="d-flex justify-content-between">
                        <div className="fw-bold small">{drug}</div>
                        <span className="badge" style={{
                          background: v.risk === 'ABSOLUTE CI' ? ACCENT7 : v.risk === 'HIGH RISK' ? ACCENT : ACCENT5,
                          fontSize: 10
                        }}>{v.risk}</span>
                      </div>
                      <div className="small text-muted mt-1">{v.detail}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: DEFINITIONS ── */}
      {tab === 3 && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <InfoBox title="Disease" color={ACCENT}>{def?.disease}</InfoBox>
              <InfoBox title="Gene / OMIM" color={ACCENT2}>Gene: {def?.gene} · OMIM gene: {def?.omim_gene} · OMIM disease: {def?.omim_disease}</InfoBox>
              <InfoBox title="Chromosome" color={ACCENT2}>{def?.chromosome}</InfoBox>
              <InfoBox title="Inheritance" color={ACCENT4}>{def?.inheritance}</InfoBox>
              <InfoBox title="Prevalence" color={ACCENT5}>{def?.prevalence}</InfoBox>
              <InfoBox title="Enzyme" color={ACCENT}>{def?.enzyme}</InfoBox>
              <InfoBox title="Reaction" color={ACCENT}>{def?.reaction}</InfoBox>
              <InfoBox title="Pathway Role" color={ACCENT4}>{def?.pathway_role}</InfoBox>
              <InfoBox title="Mechanism of Disease" color={ACCENT2}>{def?.mechanism_of_disease}</InfoBox>
              <InfoBox title="Arginine — Conditionally Essential (PRIMARY Therapy)" color={ACCENT3}>{def?.arginine_conditional_essential}</InfoBox>
              <InfoBox title="Citrulline — Pathognomonic Hallmark" color={ACCENT}>{def?.citrulline_pathognomonic}</InfoBox>
              <InfoBox title="VPA Mechanism (Doubly Blocks in ASS1)" color={ACCENT7}>{def?.vpa_mechanism}</InfoBox>
            </div>
            <div className="col-md-6">
              <InfoBox title="ASS1 vs OTC Critical Distinction" color={ACCENT}>{def?.critical_ass1_vs_otc_distinction}</InfoBox>
              <InfoBox title="ASS1 vs CPS1/NAGS Critical Distinction" color={ACCENT2}>{def?.critical_ass1_vs_cps1_nags_distinction}</InfoBox>
              <InfoBox title="ASS1 vs ASL (Closest Relative)" color={ACCENT8}>{def?.asl_distinction}</InfoBox>
              <InfoBox title="Unique Features vs Other UCDs" color={ACCENT4}>{def?.unique_features_vs_other_ucd}</InfoBox>
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>Key Positive Biomarkers</div>
                <div className="card-body small">
                  {Object.entries(def?.key_biomarker_positives || {}).map(([k, v]) => (
                    <div key={k} className="mb-1">
                      <span className="badge me-1" style={{ background: k.includes('citrulline') ? ACCENT : ACCENT2, fontSize: 10 }}>
                        {k.includes('citrulline') ? '★ CRITICALLY HIGH — HALLMARK' : 'HIGH/LOW'}
                      </span>
                      <strong>{k.replace(/_/g,' ')}:</strong> <span className="text-muted">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-bold small" style={{ color: ACCENT5 }}>Key Negative Biomarkers (NORMAL/ABSENT in ASS1)</div>
                <div className="card-body small">
                  {Object.entries(def?.key_biomarker_negatives || {}).map(([k, v]) => (
                    <div key={k} className="mb-1">
                      <span className="badge me-1" style={{
                        background: k === 'urine_orotic_acid' ? ACCENT5 : k === 'argininosuccinate' ? ACCENT8 : ACCENT3,
                        fontSize: 10
                      }}>
                        {k === 'urine_orotic_acid' ? '★ NORMAL/MILD (≠ OTC)' : k === 'argininosuccinate' ? 'ABSENT' : 'NORMAL'}
                      </span>
                      <strong>{k.replace(/_/g,' ')}:</strong> <span className="text-muted">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="card shadow-sm mb-3 border-danger">
                <div className="card-header fw-bold small text-danger">Absolute Contraindications</div>
                <div className="card-body small">
                  {Object.entries(def?.absolute_contraindications || {}).map(([k, v]) => (
                    <div key={k} className="mb-2 p-2 rounded" style={{ background: '#ffebee' }}>
                      <div className="fw-bold text-danger">{k.replace(/_/g,' ').toUpperCase()}</div>
                      <div className="text-muted">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
              <InfoBox title="AR Inheritance Note" color={ACCENT4}>{def?.ar_inheritance_note}</InfoBox>
              <InfoBox title="Seizure Mechanism" color={ACCENT2}>{def?.seizure_mechanism}</InfoBox>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="mt-4 pt-3 border-top small text-muted d-flex justify-content-between">
        <span>ASS1 Deficiency · Citrullinemia Type 1 (CTLN1) · 9q34.11 · OMIM #215700 · AR · Seed {ov?.seed} · n={ov?.cohort_n}</span>
        <Link href="/" className="text-muted">← Back to Portal</Link>
      </div>
    </div>
  );
}

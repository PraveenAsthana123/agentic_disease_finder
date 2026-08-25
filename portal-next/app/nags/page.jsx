'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// NAGS color scheme — upstream cofactor / NCG-responsive hallmark / VPA direct target
const ACCENT  = '#b71c1c';   // deep crimson — ammonia CRITICALLY HIGH / neonatal crisis
const ACCENT2 = '#1a237e';   // deep navy — citrulline CRITICALLY LOW / urea cycle block
const ACCENT3 = '#2e7d32';   // deep forest green — NCG COMPLETE response / unique NAGS hallmark
const ACCENT4 = '#4a148c';   // deep purple — upstream cofactor / NAGS enzyme / pathway context
const ACCENT5 = '#e65100';   // dark orange — NCG therapy / arginine axis / nitrogen scavengers
const ACCENT6 = '#006064';   // teal — key negatives / differentials / NAGS-specific glutamate
const ACCENT7 = '#880e4f';   // dark pink — VPA ABSOLUTE CI (most direct mechanism in any UCD)
const ACCENT8 = '#37474f';   // blue-grey — secondary features

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

export default function NAGSPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nags/overview`).then(r => r.json()),
      fetch(`${API}/api/nags/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nags/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT3 }} /><p className="mt-3 text-muted">Loading NAGS dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT3}18, ${ACCENT4}18)`, borderLeft: `5px solid ${ACCENT3}` }}>
        <div className="d-flex align-items-center gap-2 mb-1 flex-wrap">
          <span style={{ fontSize: 24 }}>⚗️</span>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT3 }}>NAGS Deficiency Dashboard</h4>
          <span className="badge" style={{ background: ACCENT4, fontSize: 11 }}>Upstream UCD Cofactor Enzyme</span>
          <span className="badge" style={{ background: ACCENT3, fontSize: 11 }}>NCG COMPLETE Response — Hallmark</span>
          <span className="badge" style={{ background: ACCENT, fontSize: 11 }}>Orotic Acid NORMAL (= CPS1)</span>
          <span className="badge" style={{ background: ACCENT7, fontSize: 11 }}>VPA ABSOLUTE CI (Most Direct)</span>
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
            <KPI label="Avg Plasma Ammonia (µmol/L)" value={kpi.avg_plasma_ammonia_umol_l} color={ACCENT} />
            <KPI label="Avg Citrulline (µmol/L)" value={kpi.avg_plasma_citrulline_umol_l} color={ACCENT2} />
            <KPI label="Avg Urine Orotic Acid (µmol/mol Cr)" value={kpi.avg_urine_orotic_acid_umol_mol} color={ACCENT3} />
            <KPI label="Avg Arginine (µmol/L)" value={kpi.avg_plasma_arginine_umol_l} color={ACCENT4} />
            <KPI label="Avg Glutamate (µmol/L)" value={kpi.avg_plasma_glutamate_umol_l} color={ACCENT6} />
            <KPI label="Seizures %" value={`${kpi.pct_seizures}%`} color={ACCENT} />
            <KPI label="NCG-Responsive %" value={`${kpi.pct_ncg_responsive}%`} color={ACCENT3} />
            <KPI label="Status Epilepticus %" value={`${kpi.pct_status_epilepticus}%`} color={ACCENT7} />
            <KPI label="DRE %" value={`${kpi.pct_dre}%`} color={ACCENT8} />
            <KPI label="IDD %" value={`${kpi.pct_idd}%`} color={ACCENT2} />
          </div>

          {/* Phenotype + pathway */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT3 }}>Phenotype Distribution (n={ov?.cohort_n})</div>
                <div className="card-body">
                  {Object.entries(ov?.phenotype_distribution || {}).map(([ph, v]) => (
                    <PctBar
                      key={ph} label={`${ph} (n=${v.n})`} pct={v.pct}
                      color={ph.includes('NCG') ? ACCENT3 : ph.includes('Neonatal') ? ACCENT : ACCENT4}
                    />
                  ))}
                  <div className="small text-muted mt-2">
                    <strong style={{ color: ACCENT3 }}>NCG-Responsive (65%): HALLMARK</strong> — complete NH₃ normalisation with NCG/Carbaglu.<br/>
                    Neonatal-Onset (20%): null NAGS; severe; CRRT + liver transplant if NCG-non-responsive.<br/>
                    Late-Onset Episodic (15%): residual NAGS; protein aversion clue; NCG trial mandatory.
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT4 }}>Pathway Role — NAGS as Upstream Cofactor Generator</div>
                <div className="card-body small text-muted">
                  <div><strong style={{ color: ACCENT6 }}>Reaction:</strong> {def?.reaction}</div>
                  <div className="mt-2"><strong>NAGS in the urea cycle:</strong></div>
                  <ul className="mt-1 mb-2" style={{ paddingLeft: 18 }}>
                    <li style={{ color: ACCENT4, fontWeight: 700 }}>NAGS: Glutamate + Acetyl-CoA → NAG [cofactor, BLOCKED]</li>
                    <li style={{ color: ACCENT, fontWeight: 600 }}>Step 1: NH₃ + CO₂ → Carbamoyl-P [CPS1, requires NAG — INACTIVE]</li>
                    <li>Step 2: Carbamoyl-P + Ornithine → Citrulline [OTC]</li>
                    <li>Step 3: Citrulline + Asp → Argininosuccinate [ASS1]</li>
                    <li>Step 4: Argininosuccinate → Arginine + Fumarate [ASL]</li>
                    <li>Step 5: Arginine → Ornithine + Urea [ARG1]</li>
                  </ul>
                  <div className="alert alert-success py-1 px-2 small mb-2">
                    <strong>KEY: No NAGS → No NAG → CPS1 INACTIVE → No Carbamoyl-P → No Overflow → Orotic NORMAL</strong>
                  </div>
                  <div><strong style={{ color: ACCENT5 }}>Arginine axis:</strong> Arginine allosterically activates NAGS → more NAG → CPS1 more active (positive feedback)</div>
                </div>
              </div>
            </div>
          </div>

          {/* NCG hallmark + key biomarkers */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT3 }}>🏆 NCG HALLMARK — Only UCD Treatable by Single Oral Drug</div>
                <div className="card-body small">
                  {Object.entries(ov?.key_positives || {}).map(([k, v]) => (
                    <div key={k} className="mb-2 p-2 rounded" style={{
                      background: k.includes('ncg') ? '#e8f5e9' : k.includes('NORMAL') ? '#e3f2fd' : '#fff3e0'
                    }}>
                      <div className="fw-bold" style={{
                        color: k.includes('ncg') ? ACCENT3 : k.includes('NORMAL') ? ACCENT2 : ACCENT
                      }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                      <div className="text-muted">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT6 }}>🟢 KEY NEGATIVE Biomarkers (NORMAL in NAGS)</div>
                <div className="card-body small">
                  {Object.entries(ov?.biomarker_normals || {}).filter(([k]) => k !== 'description').map(([k, v]) => (
                    <div key={k} className="mb-1">
                      <span className="badge me-1" style={{ background: k.includes('orotic') ? ACCENT : ACCENT3, fontSize: 10 }}>
                        {k.includes('orotic') ? '★ NORMAL' : 'NORMAL'}
                      </span>
                      <strong>{k.replace(/_/g, ' ').replace('normal', '').trim()}: </strong>
                      <span className="text-muted">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* NAGS vs CPS1 vs OTC alerts */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="alert alert-success border-start border-5 border-success mb-0">
                <div className="fw-bold mb-1">🟢 NAGS vs OTC: Single Key Test</div>
                <div className="small">
                  <strong>Orotic acid NORMAL in NAGS</strong> — no carbamoyl-P produced → no overflow to pyrimidines.<br/>
                  <strong>Orotic acid HIGH in OTC</strong> — carbamoyl-P made but blocked at step 2 → overflows.<br/>
                  Both have: ammonia HIGH, citrulline LOW. Orotic acid = single test to separate OTC from NAGS/CPS1.
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="alert alert-primary border-start border-5 border-primary mb-0">
                <div className="fw-bold mb-1">⚗️ NAGS vs CPS1: NCG Trial Mandatory</div>
                <div className="small">
                  <strong>Biochemically IDENTICAL</strong> (both: NH₃ HIGH, citrulline LOW, orotic NORMAL).<br/>
                  <strong>NCG trial distinguishes:</strong> COMPLETE NH₃ normalisation = <span style={{ color: ACCENT3 }}>NAGS</span>; partial/no response = CPS1.<br/>
                  NAGS is the ONLY UCD where NCG can fully restore ammonia control long-term.
                </div>
              </div>
            </div>
          </div>

          {/* Arginine + VPA */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${ACCENT5}` }}>
                <div className="card-header fw-bold small" style={{ color: ACCENT5 }}>Arginine → NAGS Activation (NAGS-Specific)</div>
                <div className="card-body small text-muted">
                  {def?.arginine_nags_axis || ov?.pathway_context?.arginine_feedback}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="alert alert-danger border-start border-5 border-danger mb-0">
                <div className="fw-bold mb-1">⚠️ ABSOLUTE CI: VPA — Most Direct Mechanism in Any UCD</div>
                <div className="small">
                  VPA is a <strong>competitive inhibitor of NAGS active site</strong> — directly occupies the NAGS substrate-binding site.<br/>
                  NAGS is the PRIMARY VPA target (CPS1 and OTC are secondary). Even single dose → catastrophic hyperammonemia.<br/>
                  <strong>CI in ALL urea cycle disorders. Use LEV, LTG, CLB instead.</strong>
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
                <div className="card-header fw-bold small" style={{ color: ACCENT3 }}>Phenotype Detail</div>
                <div className="card-body">
                  {Object.entries(bd?.phenotype_detail || {}).map(([ph, v]) => (
                    <div key={ph} className="mb-3 p-2 rounded" style={{ background: '#fafafa', border: '1px solid #eee' }}>
                      <div className="fw-bold" style={{ color: ph.includes('NCG') ? ACCENT3 : ph.includes('Neonatal') ? ACCENT : ACCENT4 }}>
                        {ph} — {v.pct}% (n={v.n})
                      </div>
                      <div className="small text-muted mt-1">{v.description}</div>
                      <div className="small mt-1">
                        <span className="badge me-2" style={{ background: ACCENT, color: '#fff' }}>Peak NH₃ avg: {v.avg_peak_ammonia_umol_l} µmol/L</span>
                        <span className="badge me-2" style={{ background: ACCENT2, color: '#fff' }}>Seizures: {v.seizure_rate_pct}%</span>
                        {v.ncg_complete_response_pct > 0 && (
                          <span className="badge" style={{ background: ACCENT3, color: '#fff' }}>NCG complete: {v.ncg_complete_response_pct}%</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT4 }}>Variant Distribution (NAGS gene, 17q21.31)</div>
                <div className="card-body">
                  {(bd?.variants || []).map(v => (
                    <div key={v.name} className="mb-2 p-2 rounded" style={{ background: v.responsive ? '#e8f5e9' : '#fce4ec' }}>
                      <div className="d-flex justify-content-between">
                        <strong style={{ color: v.responsive ? ACCENT3 : ACCENT }}>{v.name}</strong>
                        <span className="badge" style={{ background: v.responsive ? ACCENT3 : ACCENT }}>{v.pct}%</span>
                      </div>
                      <div className="small text-muted">{v.domain} — {v.severity}</div>
                    </div>
                  ))}
                  <div className="small text-muted mt-2">
                    <span style={{ color: ACCENT3 }}>Green</span> = NCG-responsive (catalytic/binding domain); <span style={{ color: ACCENT }}>Red</span> = null/severe (not NCG-responsive).<br/>
                    AR inheritance: biallelic LOF required; males and females equally affected.
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Differentials */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT6 }}>Differential Diagnoses</div>
            <div className="card-body">
              <div className="row g-2">
                {(bd?.differentials || []).map(d => (
                  <div key={d.disease} className="col-md-6">
                    <div className="p-2 rounded" style={{ background: '#f5f5f5', borderLeft: `3px solid ${ACCENT6}` }}>
                      <div className="fw-bold small" style={{ color: ACCENT6 }}>{d.disease}</div>
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
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>Seizure Types (of those with seizures)</div>
                <div className="card-body">
                  {Object.entries(bd?.seizure_types || {}).map(([st, v]) => (
                    <PctBar key={st} label={`${st} (n≈${v.n})`} pct={v.pct}
                      color={st.includes('Status') ? ACCENT7 : st.includes('GTCS') ? ACCENT : ACCENT2} />
                  ))}
                  <div className="small text-muted mt-2">
                    GTCS modal in acute hyperammonemic crisis.<br/>
                    Status epilepticus rate significant — neonatal onset especially severe.<br/>
                    <strong>Treat ammonia FIRST with NCG + nitrogen scavengers; AED second.</strong>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT5 }}>Crisis Triggers</div>
                <div className="card-body">
                  {Object.entries(bd?.triggers || {}).map(([t, v]) => (
                    <div key={t} className="mb-2">
                      <PctBar label={t} pct={v.pct} color={t.includes('Valproate') ? ACCENT7 : ACCENT5} />
                      <div className="text-muted small ms-1" style={{ marginTop: -4 }}>{v.detail}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Treatments */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT3 }}>Treatment Ladder (NCG is FIRST-LINE SPECIFIC for NAGS)</div>
            <div className="card-body">
              <div className="row g-2">
                {Object.entries(bd?.treatments || {}).map(([tx, v]) => (
                  <div key={tx} className="col-md-6">
                    <div className="p-2 rounded" style={{
                      background: tx.includes('NCG') ? '#e8f5e9' : v.level === 'A' ? '#e8f5e9' : '#e3f2fd',
                      borderLeft: `3px solid ${tx.includes('NCG') ? ACCENT3 : v.level === 'A' ? ACCENT3 : ACCENT2}`
                    }}>
                      <div className="d-flex justify-content-between">
                        <div className="fw-bold small">{tx}</div>
                        <span className="badge" style={{ background: tx.includes('NCG') ? ACCENT3 : v.level === 'A' ? ACCENT3 : ACCENT2 }}>Level {v.level}</span>
                      </div>
                      <div className="small text-muted">{v.category}</div>
                      <div className="small mt-1">{v.mechanism}</div>
                      {v.efficacy_pct && (
                        <div className="progress mt-1" style={{ height: 6 }}>
                          <div className="progress-bar" style={{ width: `${v.efficacy_pct}%`, backgroundColor: tx.includes('NCG') ? ACCENT3 : v.level === 'A' ? ACCENT3 : ACCENT2 }} />
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
            <div className="card-header fw-bold small" style={{ color: ACCENT7 }}>Drug Risk Profile (VPA = Most Direct NAGS Inhibitor)</div>
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
              <InfoBox title="Disease" color={ACCENT3}>{def?.disease}</InfoBox>
              <InfoBox title="Gene / OMIM" color={ACCENT2}>Gene: {def?.gene} · OMIM gene: {def?.omim_gene} · OMIM disease: {def?.omim_disease}</InfoBox>
              <InfoBox title="Chromosome" color={ACCENT2}>{def?.chromosome}</InfoBox>
              <InfoBox title="Inheritance" color={ACCENT4}>{def?.inheritance}</InfoBox>
              <InfoBox title="Prevalence" color={ACCENT5}>{def?.prevalence}</InfoBox>
              <InfoBox title="Enzyme" color={ACCENT3}>{def?.enzyme}</InfoBox>
              <InfoBox title="Reaction" color={ACCENT3}>{def?.reaction}</InfoBox>
              <InfoBox title="Pathway Role" color={ACCENT4}>{def?.pathway_role}</InfoBox>
              <InfoBox title="Mechanism of Disease" color={ACCENT}>{def?.mechanism_of_disease}</InfoBox>
              <InfoBox title="NCG Mechanism — Specific NAGS Therapy" color={ACCENT3}>{def?.ncg_mechanism}</InfoBox>
              <InfoBox title="Arginine → NAGS Activation (Positive Feedback)" color={ACCENT5}>{def?.arginine_nags_axis}</InfoBox>
              <InfoBox title="VPA → NAGS Inhibition (Most Direct in Any UCD)" color={ACCENT7}>{def?.vpa_nags_mechanism}</InfoBox>
            </div>
            <div className="col-md-6">
              <InfoBox title="NAGS vs CPS1 Critical Distinction" color={ACCENT3}>{def?.critical_nags_vs_cps1_distinction}</InfoBox>
              <InfoBox title="NAGS vs OTC Critical Distinction" color={ACCENT2}>{def?.critical_nags_vs_otc_distinction}</InfoBox>
              <InfoBox title="Unique Features vs Other UCDs" color={ACCENT4}>{def?.unique_features_vs_other_ucd}</InfoBox>
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>Key Positive Biomarkers</div>
                <div className="card-body small">
                  {Object.entries(def?.key_biomarker_positives || {}).map(([k, v]) => (
                    <div key={k} className="mb-1">
                      <span className="badge me-1" style={{ background: k.includes('glut') ? ACCENT6 : ACCENT, fontSize: 10 }}>
                        {k.includes('glut') ? 'HIGH (NAGS-specific)' : 'HIGH'}
                      </span>
                      <strong>{k.replace(/_/g,' ')}:</strong> <span className="text-muted">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-bold small" style={{ color: ACCENT6 }}>Key Negative Biomarkers (NORMAL in NAGS)</div>
                <div className="card-body small">
                  {Object.entries(def?.key_biomarker_negatives || {}).map(([k, v]) => (
                    <div key={k} className="mb-1">
                      <span className="badge me-1" style={{ background: k === 'urine_orotic_acid' ? ACCENT : ACCENT3, fontSize: 10 }}>
                        {k === 'urine_orotic_acid' ? '★ NORMAL (= CPS1)' : 'NORMAL'}
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
        <span>NAGS Deficiency · 17q21.31 · OMIM #237310 · AR · Seed {ov?.seed} · n={ov?.cohort_n}</span>
        <Link href="/" className="text-muted">← Back to Portal</Link>
      </div>
    </div>
  );
}

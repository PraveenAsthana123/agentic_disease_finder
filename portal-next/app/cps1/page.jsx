'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// CPS1 color scheme — step 1 urea cycle / carbamoyl-phosphate deficiency / NAG axis
const ACCENT  = '#b71c1c';   // deep crimson — ammonia CRITICALLY HIGH / neonatal crisis
const ACCENT2 = '#1a237e';   // deep navy — citrulline CRITICALLY LOW / urea cycle block
const ACCENT3 = '#2e7d32';   // deep forest green — NORMAL orotic acid (KEY NEGATIVE vs OTC)
const ACCENT4 = '#4a148c';   // deep purple — pathway / step 1 / NAG axis
const ACCENT5 = '#e65100';   // dark orange — protein restriction / nitrogen scavengers / NCG
const ACCENT6 = '#006064';   // teal — key negatives / differentials
const ACCENT7 = '#880e4f';   // dark pink — absolute CI (VPA!)
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

export default function CPS1Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/cps1/overview`).then(r => r.json()),
      fetch(`${API}/api/cps1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/cps1/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading CPS1 dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}18, ${ACCENT4}18)`, borderLeft: `5px solid ${ACCENT}` }}>
        <div className="d-flex align-items-center gap-2 mb-1 flex-wrap">
          <span style={{ fontSize: 24 }}>⚗️</span>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>CPS1 Deficiency Dashboard</h4>
          <span className="badge" style={{ background: ACCENT4, fontSize: 11 }}>Urea Cycle Step 1 (Most Proximal)</span>
          <span className="badge" style={{ background: ACCENT3, fontSize: 11 }}>Orotic Acid NORMAL</span>
          <span className="badge" style={{ background: ACCENT7, fontSize: 11 }}>VPA ABSOLUTE CI</span>
          <span className="badge bg-secondary" style={{ fontSize: 11 }}>NCG Trial Mandatory</span>
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
            <KPI label="Avg Ornithine (µmol/L)" value={kpi.avg_plasma_ornithine_umol_l} color={ACCENT5} />
            <KPI label="Seizures %" value={`${kpi.pct_seizures}%`} color={ACCENT} />
            <KPI label="Status Epilepticus %" value={`${kpi.pct_status_epilepticus}%`} color={ACCENT7} />
            <KPI label="DRE %" value={`${kpi.pct_dre}%`} color={ACCENT8} />
            <KPI label="IDD %" value={`${kpi.pct_idd}%`} color={ACCENT2} />
            <KPI label="Cerebral Oedema %" value={`${kpi.pct_cerebral_oedema}%`} color={ACCENT} />
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
                      color={ph.includes('Neonatal') ? ACCENT : ph.includes('NCG') ? ACCENT5 : ACCENT4}
                    />
                  ))}
                  <div className="small text-muted mt-2">
                    Neonatal-Onset: null/near-null CPS1; day 1–3 presentation; ammonia &gt;1000; most severe proximal UCD<br/>
                    Late-Onset: residual 5–20% CPS1; episodic crisis; protein aversion clue<br/>
                    NCG-Responsive: NAG-binding domain variants; NCG activates residual CPS1; milder; may avoid liver Tx
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT4 }}>Pathway Position — Urea Cycle Step 1 (Entry)</div>
                <div className="card-body small text-muted">
                  <div><strong style={{ color: ACCENT2 }}>Reaction:</strong> {def?.reaction}</div>
                  <div className="mt-2"><strong>Urea cycle (5 steps):</strong></div>
                  <ol className="mt-1 mb-2" style={{ paddingLeft: 18 }}>
                    <li style={{ color: ACCENT, fontWeight: 700 }}>NH₃ + CO₂ → Carbamoyl-P [CPS1 ← BLOCKED; needs NAG]</li>
                    <li>Carbamoyl-P + Ornithine → Citrulline [OTC]</li>
                    <li>Citrulline + Asp → Argininosuccinate [ASS1]</li>
                    <li>Argininosuccinate → Arginine + Fumarate [ASL]</li>
                    <li>Arginine → Ornithine + Urea [ARG1]</li>
                  </ol>
                  <div className="alert alert-success py-1 px-2 small mb-2">
                    <strong>KEY: No Carbamoyl-P → No Overflow → Orotic Acid NORMAL</strong><br/>
                    (Unlike OTC where carbamoyl-P made but overflows → orotic HIGH)
                  </div>
                  <div><strong style={{ color: ACCENT5 }}>NAG axis:</strong> NAGS makes NAG → activates CPS1. VPA blocks NAGS → no NAG → CPS1 OFF</div>
                </div>
              </div>
            </div>
          </div>

          {/* NCG + key biomarkers */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT5 }}>🔑 KEY POSITIVE NEGATIVES & KEY POSITIVES</div>
                <div className="card-body small">
                  {Object.entries(ov?.key_positives || {}).map(([k, v]) => (
                    <div key={k} className="mb-2 p-2 rounded" style={{ background: k.includes('NORMAL') || k.includes('orotic') ? '#e8f5e9' : '#fff3e0' }}>
                      <div className="fw-bold" style={{ color: k.includes('NORMAL') || k.includes('orotic') ? ACCENT3 : ACCENT }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                      <div className="text-muted">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT6 }}>🟢 KEY NEGATIVE Biomarkers (NORMAL in CPS1)</div>
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

          {/* CPS1 vs OTC vs NAGS alert */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="alert alert-success border-start border-5 border-success mb-0">
                <div className="fw-bold mb-1">🟢 CPS1 vs OTC: Single Key Test</div>
                <div className="small">
                  <strong>Orotic acid NORMAL in CPS1</strong> — no carbamoyl-P produced → no overflow to pyrimidines.<br/>
                  <strong>Orotic acid HIGH in OTC</strong> — carbamoyl-P made at step 1 but blocked at step 2 → overflows.<br/>
                  Both have: ammonia HIGH, citrulline LOW. Orotic acid is the SINGLE differentiating test.
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="alert alert-warning border-start border-5 border-warning mb-0">
                <div className="fw-bold mb-1">⚗️ CPS1 vs NAGS: NCG Trial Mandatory</div>
                <div className="small">
                  <strong>Biochemically identical</strong> (both: ammonia HIGH, citrulline LOW, orotic NORMAL).<br/>
                  <strong>NCG trial distinguishes:</strong> complete NH₃ normalisation = NAGS; partial/no response = CPS1.<br/>
                  NCG trial is MANDATORY in all newly diagnosed CPS1/NAGS — safe even if non-responsive.
                </div>
              </div>
            </div>
          </div>

          {/* VPA Warning */}
          <div className="alert alert-danger border-start border-5 border-danger mb-3">
            <div className="fw-bold mb-1">⚠️ ABSOLUTE CONTRAINDICATION: Valproate / VPA</div>
            <div className="small">
              VPA inhibits NAGS → no NAG production → CPS1 CANNOT be activated at all.
              In CPS1 deficiency, mechanism is MORE DIRECT than in OTC: abolishes the obligate CPS1 activator.
              Even single therapeutic dose → catastrophic hyperammonemia. Multiple fatalities.
              <strong> CI in ALL urea cycle disorders.</strong> Use LEV, LTG, or CLB instead.
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
                      <div className="fw-bold" style={{ color: ph.includes('Neonatal') ? ACCENT : ph.includes('NCG') ? ACCENT5 : ACCENT4 }}>
                        {ph} — {v.pct}% (n={v.n})
                      </div>
                      <div className="small text-muted mt-1">{v.description}</div>
                      <div className="small mt-1">
                        <span className="badge me-2" style={{ background: ACCENT, color: '#fff' }}>Peak NH₃ avg: {v.avg_peak_ammonia_umol_l} µmol/L</span>
                        <span className="badge" style={{ background: ACCENT2, color: '#fff' }}>Seizures: {v.seizure_rate_pct}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT4 }}>Variant Distribution (CPS1 gene, 2q34)</div>
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
                    Green = NCG-responsive (NAG-binding domain); Red = null/severe (neonatal-onset).<br/>
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
                    GTCS most common in acute hyperammonemic crisis.<br/>
                    Status epilepticus rate higher in CPS1 than OTC (step 1 block = complete urea cycle arrest).<br/>
                    Seizures secondary to ammonia — treat ammonia FIRST, AED second.
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
                      <PctBar label={t} pct={v.pct} color={ACCENT5} />
                      <div className="text-muted small ms-1" style={{ marginTop: -4 }}>{v.detail}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Treatments */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT3 }}>Treatment Ladder (NCG is CPS1-specific)</div>
            <div className="card-body">
              <div className="row g-2">
                {Object.entries(bd?.treatments || {}).map(([tx, v]) => (
                  <div key={tx} className="col-md-6">
                    <div className="p-2 rounded" style={{
                      background: v.level === 'A' ? '#e8f5e9' : '#e3f2fd',
                      borderLeft: `3px solid ${v.level === 'A' ? ACCENT3 : ACCENT2}`
                    }}>
                      <div className="d-flex justify-content-between">
                        <div className="fw-bold small">{tx}</div>
                        <span className="badge" style={{ background: v.level === 'A' ? ACCENT3 : ACCENT2 }}>Level {v.level}</span>
                      </div>
                      <div className="small text-muted">{v.category}</div>
                      <div className="small mt-1">{v.mechanism}</div>
                      {v.efficacy_pct && (
                        <div className="progress mt-1" style={{ height: 6 }}>
                          <div className="progress-bar" style={{ width: `${v.efficacy_pct}%`, backgroundColor: v.level === 'A' ? ACCENT3 : ACCENT2 }} />
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
            <div className="card-header fw-bold small" style={{ color: ACCENT7 }}>Drug Risk Profile</div>
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
              <InfoBox title="Enzyme" color={ACCENT3}>{def?.enzyme}</InfoBox>
              <InfoBox title="Reaction" color={ACCENT3}>{def?.reaction}</InfoBox>
              <InfoBox title="Pathway Position" color={ACCENT4}>{def?.pathway_position}</InfoBox>
              <InfoBox title="Mechanism of Disease" color={ACCENT}>{def?.mechanism_of_disease}</InfoBox>
              <InfoBox title="NAG Allosteric Axis (CPS1-specific)" color={ACCENT4}>{def?.nag_axis}</InfoBox>
            </div>
            <div className="col-md-6">
              <InfoBox title="CPS1 vs OTC Critical Distinction" color={ACCENT3}>{def?.critical_cps1_vs_otc_distinction}</InfoBox>
              <InfoBox title="CPS1 vs NAGS Critical Distinction" color={ACCENT5}>{def?.critical_cps1_vs_nags_distinction}</InfoBox>
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>Key Positive Biomarkers</div>
                <div className="card-body small">
                  {Object.entries(def?.key_biomarker_positives || {}).map(([k, v]) => (
                    <div key={k} className="mb-1"><span className="badge me-1" style={{ background: ACCENT, fontSize: 10 }}>HIGH</span><strong>{k.replace(/_/g,' ')}:</strong> <span className="text-muted">{v}</span></div>
                  ))}
                </div>
              </div>
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-bold small" style={{ color: ACCENT6 }}>Key Negative Biomarkers (NORMAL in CPS1)</div>
                <div className="card-body small">
                  {Object.entries(def?.key_biomarker_negatives || {}).map(([k, v]) => (
                    <div key={k} className="mb-1">
                      <span className="badge me-1" style={{ background: k === 'urine_orotic_acid' ? ACCENT : ACCENT3, fontSize: 10 }}>
                        {k === 'urine_orotic_acid' ? '★ NORMAL' : 'NORMAL'}
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
        <span>CPS1 Deficiency · 2q34 · OMIM #237300 · AR · Seed {ov?.seed} · n={ov?.cohort_n}</span>
        <Link href="/" className="text-muted">← Back to Portal</Link>
      </div>
    </div>
  );
}

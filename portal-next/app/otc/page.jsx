'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// OTC color scheme — hyperammonemia / urea cycle block / citrulline deficiency
const ACCENT  = '#b71c1c';   // deep crimson — ammonia CRITICALLY HIGH / neonatal crisis
const ACCENT2 = '#1a237e';   // deep navy — citrulline CRITICALLY LOW / urea cycle block
const ACCENT3 = '#1b5e20';   // deep forest green — orotic acid HIGH / carbamoyl phosphate overflow
const ACCENT4 = '#4a148c';   // deep purple — pathway position / urea cycle step 2
const ACCENT5 = '#e65100';   // dark orange — protein restriction / nitrogen scavengers
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

export default function OTCPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/otc/overview`).then(r => r.json()),
      fetch(`${API}/api/otc/breakdown`).then(r => r.json()),
      fetch(`${API}/api/otc/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading OTC dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}18, ${ACCENT2}18)`, borderLeft: `5px solid ${ACCENT}` }}>
        <div className="d-flex align-items-center gap-2 mb-1">
          <span style={{ fontSize: 24 }}>⚡</span>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>OTC Deficiency Dashboard</h4>
          <span className="badge" style={{ background: ACCENT2, fontSize: 11 }}>Most Common Urea Cycle Disorder</span>
          <span className="badge" style={{ background: ACCENT7, fontSize: 11 }}>VPA ABSOLUTE CI</span>
        </div>
        <div className="small text-muted">{ov?.subtitle}</div>
        <div className="d-flex gap-3 mt-2 flex-wrap">
          {[
            ['Gene', ov?.gene],
            ['Chr', ov?.chromosome],
            ['Protein', ov?.protein_size?.split(';')[0]],
            ['OMIM Disease', ov?.omim_disease],
            ['Inheritance', 'X-linked'],
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

          {/* Phenotype distribution */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>Phenotype Distribution (n={ov?.cohort_n})</div>
                <div className="card-body">
                  {Object.entries(ov?.phenotype_distribution || {}).map(([ph, v]) => (
                    <PctBar
                      key={ph} label={`${ph} (n=${v.n})`} pct={v.pct}
                      color={ph === 'Neonatal-Onset Males' ? ACCENT : ph === 'Late-Onset Males' ? ACCENT5 : ACCENT4}
                    />
                  ))}
                  <div className="small text-muted mt-2">
                    Neonatal-Onset Males: hemizygous null; crisis day 1–5; ammonia &gt;500 µmol/L; often lethal untreated<br/>
                    Late-Onset Males: residual 5–25% OTC activity; triggered by illness/protein/fasting<br/>
                    Symptomatic Females: skewed X-inactivation (Lyon effect); late-onset; protein aversion clue
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT3 }}>Pathway Position — Urea Cycle Step 2</div>
                <div className="card-body small text-muted">
                  <div><strong style={{ color: ACCENT2 }}>Reaction:</strong> {def?.reaction}</div>
                  <div className="mt-2"><strong>Urea cycle (5 steps):</strong></div>
                  <ol className="mt-1 mb-2" style={{ paddingLeft: 18 }}>
                    <li>NH₃ + CO₂ → <strong>Carbamoyl-P</strong> [CPS1]</li>
                    <li style={{ color: ACCENT, fontWeight: 700 }}>Carbamoyl-P + Ornithine → Citrulline [OTC ← BLOCKED]</li>
                    <li>Citrulline + Asp → Argininosuccinate [ASS1]</li>
                    <li>Argininosuccinate → Arginine + Fumarate [ASL]</li>
                    <li>Arginine → Ornithine + Urea [ARG1]</li>
                  </ol>
                  <div className="alert alert-danger py-1 px-2 small mb-2">
                    <strong>CARBAMOYL-P OVERFLOW:</strong> Excess carbamoyl-P → cytoplasm → pyrimidine synthesis → OROTIC ACID ↑ (urine)
                  </div>
                  <div><strong style={{ color: ACCENT2 }}>OTC vs CPS1 KEY TEST:</strong> Orotic acid HIGH in OTC, NORMAL in CPS1</div>
                </div>
              </div>
            </div>
          </div>

          {/* Key biomarkers */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>🔴 KEY POSITIVE Biomarkers</div>
                <div className="card-body small">
                  {Object.entries(ov?.key_positives || {}).map(([k, v]) => (
                    <div key={k} className="mb-2 p-2 rounded" style={{ background: '#fff3e0' }}>
                      <div className="fw-bold" style={{ color: ACCENT }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                      <div className="text-muted">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT6 }}>🟢 KEY NEGATIVE Biomarkers (NORMAL in OTC)</div>
                <div className="card-body small">
                  {Object.entries(ov?.biomarker_normals || {}).filter(([k]) => k !== 'description').map(([k, v]) => (
                    <div key={k} className="mb-1">
                      <span className="badge bg-success me-1" style={{ fontSize: 10 }}>NORMAL</span>
                      <strong>{k.replace(/_/g, ' ').replace('normal', '').trim()}: </strong>
                      <span className="text-muted">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Critical VPA warning */}
          <div className="alert alert-danger border-start border-5 border-danger mb-3">
            <div className="fw-bold mb-1">⚠️ ABSOLUTE CONTRAINDICATION: Valproate / VPA</div>
            <div className="small">
              VPA inhibits CPS1 and NAGS (N-acetylglutamate synthase — the CPS1 activator) + impairs mitochondrial beta-oxidation.
              Even therapeutic VPA doses → hyperammonemic crisis in OTC deficiency. Multiple fatalities reported.
              <strong> CI in ALL urea cycle disorders.</strong> Never use VPA in OTC — use LEV, LTG, or TPM (with monitoring) instead.
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
                      <div className="fw-bold" style={{ color: ph === 'Neonatal-Onset Males' ? ACCENT : ph === 'Late-Onset Males' ? ACCENT5 : ACCENT4 }}>
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
                <div className="card-header fw-bold small" style={{ color: ACCENT3 }}>Variant Distribution (OTC gene, Xp21.1)</div>
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
                    Green = partial function (late-onset); Red = null/severe (neonatal-onset).<br/>
                    X-linked: males hemizygous; females carrier (heterozygous).
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
                    Status epilepticus constitutes a neurological emergency in OTC — emergent ammonia removal required.
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
            <div className="card-header fw-bold small" style={{ color: ACCENT3 }}>Treatment Ladder</div>
            <div className="card-body">
              <div className="row g-2">
                {Object.entries(bd?.treatments || {}).map(([tx, v]) => (
                  <div key={tx} className="col-md-6">
                    <div className="p-2 rounded" style={{ background: v.level === 'A' ? '#e8f5e9' : v.level === 'B' ? '#e3f2fd' : '#fff9c4', borderLeft: `3px solid ${v.level === 'A' ? ACCENT3 : v.level === 'B' ? ACCENT2 : ACCENT5}` }}>
                      <div className="d-flex justify-content-between">
                        <div className="fw-bold small">{tx}</div>
                        <span className="badge" style={{ background: v.level === 'A' ? ACCENT3 : v.level === 'B' ? ACCENT2 : ACCENT5 }}>Level {v.level}</span>
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
            </div>
            <div className="col-md-6">
              <InfoBox title="Critical OTC vs CPS1 Distinction" color={ACCENT3}>{def?.critical_otc_vs_cps1_distinction}</InfoBox>
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>Key Positive Biomarkers</div>
                <div className="card-body small">
                  {Object.entries(def?.key_biomarker_positives || {}).map(([k, v]) => (
                    <div key={k} className="mb-1"><span className="badge me-1" style={{ background: ACCENT, fontSize: 10 }}>HIGH</span><strong>{k.replace(/_/g,' ')}:</strong> <span className="text-muted">{v}</span></div>
                  ))}
                </div>
              </div>
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-bold small" style={{ color: ACCENT6 }}>Key Negative Biomarkers (NORMAL in OTC)</div>
                <div className="card-body small">
                  {Object.entries(def?.key_biomarker_negatives || {}).map(([k, v]) => (
                    <div key={k} className="mb-1"><span className="badge bg-success me-1" style={{ fontSize: 10 }}>NORMAL</span><strong>{k.replace(/_/g,' ')}:</strong> <span className="text-muted">{v}</span></div>
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
              <InfoBox title="X-Linkage Note" color={ACCENT4}>{def?.x_linkage_note}</InfoBox>
              <InfoBox title="Seizure Mechanism" color={ACCENT2}>{def?.seizure_mechanism}</InfoBox>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="mt-4 pt-3 border-top small text-muted d-flex justify-content-between">
        <span>OTC Deficiency · Xp21.1 · OMIM #311250 · Seed {ov?.seed} · n={ov?.cohort_n}</span>
        <Link href="/" className="text-muted">← Back to Portal</Link>
      </div>
    </div>
  );
}

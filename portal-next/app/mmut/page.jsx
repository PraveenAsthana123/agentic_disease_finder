'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — MMUT / MMA
const ACCENT2 = '#b71c1c';   // dark red — metabolic crisis / hyperammonemia DANGER
const ACCENT3 = '#e65100';   // deep orange — MMA accumulation / PATHOGNOMONIC
const ACCENT4 = '#1565c0';   // deep blue — carnitine / treatment / level A
const ACCENT5 = '#880e4f';   // dark magenta — CKD / cardiomyopathy / optic atrophy
const ACCENT6 = '#1b5e20';   // dark green — KEY NEGATIVES / normal markers
const ACCENT7 = '#37474f';   // dark slate — variant data / gene card
const ACCENT8 = '#4a148c';   // deep purple — AdoCbl / cobalamin / B12 mechanism

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

function Section({ title, children, color = ACCENT }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

export default function MMUTPage() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mmut/overview`).then(r => r.json()),
      fetch(`${API}/api/mmut/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mmut/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!ov)  return <div className="text-center p-5 text-muted">Loading MMUT Dashboard…</div>;

  const kpis = ov.kpis || {};

  return (
    <div className="container-fluid px-3 py-3">
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🧬 MMUT Epilepsy — Methylmalonic Acidemia (Methylmalonyl-CoA Mutase Deficiency)
        </h4>
        <div className="text-muted small">
          MMUT · 6p12.3 · AR · OMIM *609058 / #251000 · 750 aa · mitochondrial homodimer · AdoCbl (adenosylcobalamin) cofactor ·
          L-methylmalonyl-CoA → succinyl-CoA BLOCKED · MMA PATHOGNOMONIC (urine 200–10,000+) ·
          NO methylcitrate (KEY NEG vs PA) · Homocysteine NORMAL (KEY NEG vs cblC) · CKD unique ·
          VPA ABSOLUTE CI · n={ov.cohort_n}
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          <Alert
            variant="danger"
            text="⚠️ METHYLMALONIC ACIDEMIA (MMUT) — ACUTE DECOMPENSATION: MMA >5,000 mmol/mol Cr + hyperammonemia (NH₃ >300 = EMERGENT) + metabolic acidosis. FIRST response: IV glucose GIR 8–12 + STOP PROTEIN + ammonia scavengers. RENAL monitoring mandatory (CKD progressive — unique to MMA, not PA). VPA ABSOLUTE CI — inhibits AdoCbl metabolism + depletes carnitine + worsens MMA: FATAL. Fasting = EXTREME HAZARD. NBS: C3 elevated IDENTICAL to PA — urine organic acids MANDATORY to distinguish."
          />

          {/* Gene & Disease Overview */}
          <Section title="Gene & Disease Overview" color={ACCENT}>
            <div className="row g-2 mb-2">
              {[
                ['Gene',          ov.gene || 'MMUT'],
                ['Full Name',     ov.full_name || 'Methylmalonyl-CoA Mutase'],
                ['Chromosome',    ov.chromosome || '6p12.3'],
                ['Inheritance',   ov.inheritance || 'AR'],
                ['OMIM Gene',     ov.omim_gene || '*609058'],
                ['OMIM Disease',  ov.omim_disease || '#251000'],
                ['Protein',       ov.protein_size || '750 aa; mitochondrial; AdoCbl cofactor'],
                ['Prevalence',    ov.prevalence || '~1:50,000–80,000 (all MMA)'],
                ['NBS Primary',   ov.nbs_primary || 'C3 (propionylcarnitine) elevated — triggers MMA workup'],
                ['NBS Secondary', ov.nbs_secondary || 'Urine MMA (PATHOGNOMONIC); plasma MMA; C3/C2 ratio'],
              ].map(([k, v]) => (
                <div key={k} className="col-12 col-md-6 col-lg-4">
                  <span className="fw-semibold text-muted small">{k}: </span>
                  <span className="small">{v}</span>
                </div>
              ))}
            </div>
            <div className="alert alert-info py-2" style={{ fontSize: 12 }}>
              <strong>Function:</strong> {ov.function}<br />
              <strong>Mechanism:</strong> {ov.mechanism}<br />
              <strong>Key Negative:</strong> {ov.key_negative}
            </div>
          </Section>

          {/* KPIs */}
          <Section title={`Cohort KPIs — ${ov.cohort_n} Patients`} color={ACCENT3}>
            <div className="row g-2">
              <KPI label="Avg MMA (mmol/mol Cr)" value={kpis.avg_mma_urine?.toLocaleString()} color={ACCENT3} />
              <KPI label="Avg C3 (µmol/L)" value={kpis.avg_c3_umol_l} color={ACCENT3} />
              <KPI label="Avg NH₃ (µmol/L)" value={kpis.avg_ammonia_umol_l} color={ACCENT2} />
              <KPI label="Avg Carnitine" value={kpis.avg_free_carnitine} color={ACCENT4} />
              <KPI label="Avg eGFR" value={kpis.avg_egfr} color={ACCENT5} />
              <KPI label="Seizures %" value={kpis.seizure_pct + '%'} color={ACCENT2} />
              <KPI label="Cardiomyopathy %" value={kpis.cardiomyopathy_pct + '%'} color={ACCENT5} />
              <KPI label="Optic Atrophy %" value={kpis.optic_atrophy_pct + '%'} color={ACCENT5} />
              <KPI label="CKD %" value={kpis.ckd_pct + '%'} color={ACCENT5} />
              <KPI label="NBS Detected %" value={kpis.nbs_detected_pct + '%'} color={ACCENT4} />
              <KPI label="AdoCbl Response %" value={kpis.adocbl_response_pct + '%'} color={ACCENT8} />
              <KPI label="Transplant %" value={kpis.transplant_pct + '%'} color={ACCENT6} />
            </div>
          </Section>

          {/* Phenotype Distribution */}
          <Section title="Phenotype Distribution (mut0 vs mut-)" color={ACCENT2}>
            {(ov.phenotype_distribution || []).map((p, i) => (
              <PctBar key={i} label={p.phenotype} pct={p.pct}
                color={i === 0 ? ACCENT2 : i === 1 ? ACCENT8 : i === 2 ? ACCENT3 : ACCENT4} />
            ))}
          </Section>

          {/* MMUT Pathway */}
          <Section title="MMUT Metabolic Pathway — Propionyl-CoA → Succinyl-CoA (4 Steps)" color={ACCENT8}>
            <div className="row g-2">
              {(ov.mmut_pathway || []).map((s, i) => (
                <div key={i} className="col-12 col-md-6">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-body py-2 px-3">
                      <div className="fw-bold small" style={{ color: i === 2 ? ACCENT2 : ACCENT8 }}>{s.step}</div>
                      <div className="font-monospace" style={{ fontSize: 11, color: ACCENT }}>{s.reaction}</div>
                      <div className="text-muted" style={{ fontSize: 11 }}>Enzyme: {s.enzyme}</div>
                      {s.cofactor && s.cofactor !== 'None' && (
                        <div className="text-muted" style={{ fontSize: 11 }}>Cofactor: {s.cofactor}</div>
                      )}
                      <div className="mt-1 small" style={{ color: i === 2 ? ACCENT2 : ACCENT6 }}>
                        <strong>MMUT LOF:</strong> {s.consequence_mmut_lof}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* MMUT vs PA */}
          {ov.mmut_vs_pa && (
            <Section title={ov.mmut_vs_pa.title} color={ACCENT6}>
              <div className="text-muted small mb-2">{ov.mmut_vs_pa.note}</div>
              <div className="table-responsive">
                <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                  <thead className="table-dark">
                    <tr>
                      <th>Feature</th>
                      <th style={{ color: '#80cbc4' }}>MMA — MMUT (this disease)</th>
                      <th style={{ color: '#ce93d8' }}>PA — PCCA/PCCB (comparator)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.mmut_vs_pa.comparison || []).map((row, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{row.feature}</td>
                        <td style={{ color: row.MMA_MMUT?.includes('PATHOGNOMONIC') ? ACCENT3 : row.MMA_MMUT?.includes('NORMAL') ? ACCENT6 : 'inherit' }}>{row.MMA_MMUT}</td>
                        <td style={{ color: row.PA_PCCAPCCB?.includes('PATHOGNOMONIC') ? ACCENT3 : row.PA_PCCAPCCB?.includes('ABSENT') ? ACCENT6 : 'inherit' }}>{row.PA_PCCAPCCB}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          )}

          {/* High-Risk Situations */}
          <Section title="High-Risk Situations" color={ACCENT2}>
            {(ov.high_risk_situations || []).map((r, i) => (
              <div key={i} className="d-flex align-items-start mb-2">
                <span className={`badge me-2 ${
                  r.risk === 'ABSOLUTE CI' ? 'bg-danger' :
                  r.risk === 'EXTREME HAZARD' ? 'bg-warning text-dark' :
                  r.risk === 'NOT EFFECTIVE' ? 'bg-secondary' :
                  r.risk === 'AVOID' ? 'bg-warning text-dark' :
                  'bg-secondary'
                }`} style={{ minWidth: 130, fontSize: 10 }}>
                  {r.risk}
                </span>
                <div className="small"><strong>{r.situation}:</strong> {r.detail}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── TAB 1: PATIENTS & BIOMARKERS ── */}
      {tab === 1 && bk && (
        <div>
          {/* Biomarkers */}
          <Section title="Biomarkers (11 key measures)" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr>
                    <th>Biomarker</th><th>Normal Range</th><th>MMA Range</th>
                    <th>Significance</th><th>Method</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.biomarkers || []).map((b, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{b.name}</td>
                      <td>{b.normal}</td>
                      <td style={{ color: b.mma_range?.includes('NORMAL') ? ACCENT6 : b.mma_range?.includes('PATHOGNOMONIC') ? ACCENT3 : ACCENT3 }}>
                        {b.mma_range}
                      </td>
                      <td>{b.significance}</td>
                      <td className="text-muted">{b.method}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Key Variants */}
          <Section title="Key Pathogenic Variants in MMUT" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Variant</th><th>cDNA</th><th>Domain</th><th>Class</th><th>Note</th></tr>
                </thead>
                <tbody>
                  {(bk.key_variants || []).map((v, i) => (
                    <tr key={i}>
                      <td className="fw-semibold font-monospace">{v.variant}</td>
                      <td className="font-monospace small">{v.cdna}</td>
                      <td>{v.domain}</td>
                      <td>
                        <span className={`badge ${v.severity?.includes('mut0') ? 'bg-danger' : v.severity?.includes('mut-') ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                          {v.severity}
                        </span>
                      </td>
                      <td className="text-muted small">{v.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Patient Cohort Sample */}
          <Section title="Patient Cohort Sample (15 of 40)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 11 }}>
                <thead className="table-dark">
                  <tr>
                    <th>ID</th><th>Sex</th><th>Phenotype</th><th>Onset (mo)</th>
                    <th>MMA (mmol/mol Cr)</th><th>C3 (µmol/L)</th>
                    <th>NH₃ (µmol/L)</th><th>Carnitine</th><th>eGFR</th>
                    <th>Seizures</th><th>Cardio</th><th>Optic</th>
                    <th>NBS</th><th>B12 Resp</th><th>Genotype</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.patient_sample || []).map((p, i) => (
                    <tr key={i}>
                      <td className="font-monospace">{p.id}</td>
                      <td>{p.sex}</td>
                      <td style={{ fontSize: 10 }}>{p.phenotype}</td>
                      <td>{p.onset_age_months}</td>
                      <td style={{ color: p.mma_urine_mmol_molCr > 500 ? ACCENT3 : 'inherit', fontWeight: 'bold' }}>{p.mma_urine_mmol_molCr?.toLocaleString()}</td>
                      <td style={{ color: p.c3_umol_l > 5 ? ACCENT3 : 'inherit' }}>{p.c3_umol_l}</td>
                      <td style={{ color: p.ammonia_umol_l > 150 ? ACCENT2 : 'inherit' }}>{p.ammonia_umol_l}</td>
                      <td style={{ color: p.free_carnitine_umol_l < 20 ? ACCENT5 : 'inherit' }}>{p.free_carnitine_umol_l}</td>
                      <td style={{ color: p.egfr_ml_min_1_73m2 < 60 ? ACCENT5 : 'inherit' }}>{p.egfr_ml_min_1_73m2}</td>
                      <td>{p.seizures ? '✓' : '—'}</td>
                      <td style={{ color: p.cardiomyopathy ? ACCENT5 : 'inherit' }}>{p.cardiomyopathy ? '✓ DCM' : '—'}</td>
                      <td style={{ color: p.optic_atrophy ? ACCENT5 : 'inherit' }}>{p.optic_atrophy ? '✓ OA' : '—'}</td>
                      <td>{p.nbs_detected ? '✓' : '✗'}</td>
                      <td style={{ color: p.adobcl_response ? ACCENT8 : 'inherit' }}>{p.adobcl_response ? '✓ resp' : '—'}</td>
                      <td className="font-monospace" style={{ fontSize: 10 }}>{p.genotype}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 2: SEIZURES & TRIGGERS ── */}
      {tab === 2 && bk && (
        <div>
          {/* Seizure Types */}
          <Section title="Seizure Types in Methylmalonic Acidemia (MMUT)" color={ACCENT2}>
            <div className="row g-2">
              {(bk.seizure_types || []).map((s, i) => (
                <div key={i} className="col-12 col-md-6">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-body py-2 px-3">
                      <div className="d-flex justify-content-between align-items-center mb-1">
                        <span className="fw-semibold small">{s.type}</span>
                        <span className="badge" style={{ backgroundColor: ACCENT2 }}>{s.pct}%</span>
                      </div>
                      <div className="progress mb-1" style={{ height: 6 }}>
                        <div className="progress-bar" style={{ width: `${s.pct}%`, backgroundColor: ACCENT2 }} />
                      </div>
                      <div className="text-muted" style={{ fontSize: 11 }}>{s.note}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Metabolic Triggers */}
          <Section title="Metabolic Crisis Triggers" color={ACCENT3}>
            {(bk.metabolic_triggers || []).map((t, i) => (
              <div key={i} className="mb-2 d-flex align-items-start">
                <span className="badge me-2 bg-warning text-dark" style={{ minWidth: 40, fontSize: 10 }}>
                  {t.pct}%
                </span>
                <div className="small"><strong>{t.trigger}:</strong> {t.mechanism}</div>
              </div>
            ))}
          </Section>

          {/* High-Risk Drugs */}
          <Section title="High-Risk Drugs / Substances" color={ACCENT2}>
            {(bk.high_risk_drugs || []).map((d, i) => (
              <div key={i} className="d-flex align-items-start mb-2">
                <span className={`badge me-2 ${
                  d.risk === 'ABSOLUTE CI' ? 'bg-danger' :
                  d.risk === 'EXTREME HAZARD' ? 'bg-warning text-dark' :
                  d.risk === 'AVOID' ? 'bg-warning text-dark' :
                  'bg-secondary'
                }`} style={{ minWidth: 130, fontSize: 10 }}>{d.risk}</span>
                <div className="small"><strong>{d.drug}:</strong> {d.mechanism}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── TAB 3: TREATMENTS ── */}
      {tab === 3 && bk && (
        <div>
          <Alert
            variant="info"
            text="ℹ️ MMUT TREATMENT KEY: Hydroxocobalamin trial (Level A) MANDATORY for ALL patients — only mut- with residual enzyme respond (~20%). L-Carnitine (Level A) for secondary depletion. RENAL monitoring (CKD is progressive and unique to MMA). Combined liver-kidney transplant is best long-term option for mut0 (liver provides MMUT enzyme + kidney corrects CKD). VPA ABSOLUTE CI — use LEV as first-line AED."
          />
          <div className="row g-2">
            {(bk.treatments || []).map((t, i) => (
              <div key={i} className="col-12 col-md-6">
                <div className="card border-0 shadow-sm h-100">
                  <div className="card-body py-2 px-3">
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <span className="fw-semibold small">{t.treatment}</span>
                      <span className={`badge ${
                        t.evidence === 'Level A' ? 'bg-success' :
                        t.evidence === 'Level B' ? 'bg-primary' :
                        t.evidence === 'NOT EFFECTIVE' ? 'bg-secondary' :
                        t.evidence === 'ABSOLUTE CI' ? 'bg-danger' : 'bg-secondary'
                      }`}>{t.evidence}</span>
                    </div>
                    {t.response_pct > 0 && (
                      <div className="mb-1">
                        <div className="progress" style={{ height: 6 }}>
                          <div className="progress-bar bg-success" style={{ width: `${t.response_pct}%` }} />
                        </div>
                        <div className="text-muted" style={{ fontSize: 10 }}>{t.response_pct}% response</div>
                      </div>
                    )}
                    <div className="text-muted" style={{ fontSize: 11 }}>{t.note}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── TAB 4: DEFINITIONS ── */}
      {tab === 4 && df && (
        <div>
          {/* Gene Card */}
          <Section title="Gene Card — MMUT" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <tbody>
                  {Object.entries(df.gene_card || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-semibold text-muted" style={{ width: '30%' }}>{k}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Key Concepts */}
          <Section title="Key Concepts" color={ACCENT}>
            {(df.key_concepts || []).map((c, i) => (
              <div key={i} className="card border-0 shadow-sm mb-2">
                <div className="card-body py-2 px-3">
                  <div className="fw-semibold small mb-1" style={{ color: ACCENT }}>{c.concept}</div>
                  <div className="text-muted" style={{ fontSize: 12 }}>{c.explanation}</div>
                </div>
              </div>
            ))}
          </Section>

          {/* Diagnostic Thresholds */}
          <Section title="Diagnostic Thresholds" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr>
                </thead>
                <tbody>
                  {(df.diagnostic_thresholds || []).map((d, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{d.parameter}</td>
                      <td style={{ color: ACCENT3 }}>{d.threshold}</td>
                      <td>{d.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Differential Diagnosis */}
          <Section title="Differential Diagnosis" color={ACCENT6}>
            <div className="row g-2">
              {(df.differential_diagnosis || []).map((d, i) => (
                <div key={i} className="col-12 col-md-6">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-body py-2 px-3">
                      <div className="fw-semibold small mb-1" style={{ color: ACCENT6 }}>{d.disease}</div>
                      <div className="text-muted" style={{ fontSize: 11 }}>{d.distinguishing}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}
    </div>
  );
}

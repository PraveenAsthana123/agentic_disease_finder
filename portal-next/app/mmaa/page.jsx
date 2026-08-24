'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — MMAA / MMA cblA
const ACCENT2 = '#006064';   // dark teal — OHCbl response / cobalamin / AdoCbl delivery
const ACCENT3 = '#e65100';   // deep orange — MMA accumulation / PATHOGNOMONIC
const ACCENT4 = '#1565c0';   // blue — treatment / level A
const ACCENT5 = '#880e4f';   // dark magenta — CKD / cardiomyopathy
const ACCENT6 = '#1b5e20';   // dark green — KEY NEGATIVES / normal markers / OHCbl response positive
const ACCENT7 = '#37474f';   // slate — variant data / gene card
const ACCENT8 = '#4a148c';   // purple — cobalamin mechanism / GTPase chaperone / AdoCbl pathway

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

export default function MMAAPage() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mmaa/overview`).then(r => r.json()),
      fetch(`${API}/api/mmaa/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mmaa/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!ov)  return <div className="text-center p-5 text-muted">Loading MMAA Dashboard…</div>;

  const kpis = ov.kpis || {};

  return (
    <div className="container-fluid px-3 py-3">
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🧬 MMAA Epilepsy — Methylmalonic Acidemia (cblA / Cobalamin-A Chaperone Deficiency)
        </h4>
        <div className="text-muted small">
          MMAA · 4p14 · AR · OMIM *607481 / #251100 · 418 aa · G3E P-loop GTPase chaperone ·
          AdoCbl delivery from MMAB to MMUT BLOCKED · MMUT apoenzyme INTACT · MMA PATHOGNOMONIC (urine 200–5,000) ·
          OHCbl response STRONG &gt;60% (HALLMARK cblA) · Homocysteine NORMAL (KEY NEG vs cblC) ·
          CKD less severe than MMUT · VPA ABSOLUTE CI · n={ov.cohort_n}
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
            text="⚠️ MMAA (cblA) DECOMPENSATION: MMA >3,000 mmol/mol Cr + hyperammonemia (NH₃ >200 = ACTION) + metabolic acidosis. FIRST response: IV glucose GIR 8–12 + STOP PROTEIN + continue IM OHCbl. OHCbl response STRONG: >60% of cblA patients respond (vs MMUT ~20%). VPA ABSOLUTE CI — inhibits cobalamin (MMAA-MMAB) pathway + depletes carnitine: FATAL. MMUT apoenzyme INTACT — AdoCbl delivery defect only. Fasting = EXTREME HAZARD. Missed OHCbl doses = HIGH RISK (MMA rebounds days)."
          />

          {/* Gene & Disease Overview */}
          <Section title="Gene & Disease Overview" color={ACCENT}>
            <div className="row g-2 mb-2">
              {[
                ['Gene',          ov.gene || 'MMAA'],
                ['Full Name',     ov.full_name || 'Methylmalonyl-CoA Mutase-Associated GTPase'],
                ['Chromosome',    ov.chromosome || '4p14'],
                ['Inheritance',   ov.inheritance || 'AR'],
                ['OMIM Gene',     ov.omim_gene || '*607481'],
                ['OMIM Disease',  ov.omim_disease || '#251100'],
                ['Protein',       ov.protein_size || '418 aa; G3E P-loop GTPase; mitochondrial matrix'],
                ['Prevalence',    ov.prevalence || '~1:100,000–200,000 (~25–30% cbl-responsive MMA)'],
                ['NBS Primary',   ov.nbs_primary || 'C3 (propionylcarnitine) elevated — triggers MMA workup'],
                ['NBS Secondary', ov.nbs_secondary || 'Urine MMA (PATHOGNOMONIC); OHCbl trial; plasma MMA'],
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
          <Section title={`Cohort KPIs — ${ov.cohort_n} Patients (cblA / MMAA)`} color={ACCENT3}>
            <div className="row g-2">
              <KPI label="Avg MMA (mmol/mol Cr)" value={kpis.avg_mma_urine?.toLocaleString()} color={ACCENT3} />
              <KPI label="Avg C3 (µmol/L)" value={kpis.avg_c3_umol_l} color={ACCENT3} />
              <KPI label="Avg NH₃ (µmol/L)" value={kpis.avg_ammonia_umol_l} color={ACCENT2} />
              <KPI label="Avg Carnitine" value={kpis.avg_free_carnitine} color={ACCENT4} />
              <KPI label="Avg eGFR" value={kpis.avg_egfr} color={ACCENT5} />
              <KPI label="Seizures %" value={kpis.seizure_pct + '%'} color={ACCENT3} />
              <KPI label="Cardiomyopathy %" value={kpis.cardiomyopathy_pct + '%'} color={ACCENT5} />
              <KPI label="CKD (eGFR<60) %" value={kpis.ckd_pct + '%'} color={ACCENT5} />
              <KPI label="NBS Detected %" value={kpis.nbs_detected_pct + '%'} color={ACCENT4} />
              <KPI label="OHCbl Response %" value={kpis.ohcbl_response_pct + '%'} color={ACCENT2} />
              <KPI label="B12 Responsive %" value={kpis.b12_responsive_pct + '%'} color={ACCENT6} />
            </div>
            <div className="alert py-2 mt-2" style={{ backgroundColor: '#e0f7fa', fontSize: 12, borderLeft: `4px solid ${ACCENT2}` }}>
              <strong style={{ color: ACCENT2 }}>KEY HALLMARK cblA:</strong> OHCbl response rate {kpis.ohcbl_response_pct}% in this cohort —
              substantially higher than MMUT (~20% mut- only). This is the pharmacological signature of MMAA:
              MMUT apoenzyme is INTACT; flooding the cobalamin pathway partially overcomes the AdoCbl delivery defect.
            </div>
          </Section>

          {/* Phenotype Distribution */}
          <Section title="Phenotype Distribution (cblA Types)" color={ACCENT3}>
            {(ov.phenotype_distribution || []).map((p, i) => (
              <PctBar key={i} label={p.phenotype} pct={p.pct}
                color={i === 0 ? ACCENT3 : i === 1 ? ACCENT2 : i === 2 ? ACCENT4 : ACCENT6} />
            ))}
          </Section>

          {/* MMAA AdoCbl Delivery Pathway */}
          <Section title="MMAA — AdoCbl Delivery Pathway (4 Steps; MMAA is Step 3 — the Block)" color={ACCENT8}>
            <div className="row g-2">
              {(ov.mmaa_pathway || []).map((s, i) => (
                <div key={i} className="col-12 col-md-6">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-body py-2 px-3">
                      <div className="fw-bold small" style={{ color: i === 2 ? '#b71c1c' : ACCENT8 }}>{s.step}</div>
                      <div className="font-monospace" style={{ fontSize: 11, color: ACCENT }}>{s.reaction}</div>
                      <div className="text-muted" style={{ fontSize: 11 }}>Enzyme: {s.enzyme}</div>
                      {s.cofactor && s.cofactor !== 'None' && (
                        <div className="text-muted" style={{ fontSize: 11 }}>Cofactor/Cargo: {s.cofactor}</div>
                      )}
                      <div className="mt-1 small" style={{ color: i === 2 ? '#b71c1c' : ACCENT6 }}>
                        <strong>MMAA LOF:</strong> {s.consequence_mmaa_lof}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* MMAA vs MMUT Comparison */}
          {ov.mmaa_vs_mmut && (
            <Section title={ov.mmaa_vs_mmut.title} color={ACCENT6}>
              <div className="text-muted small mb-2">{ov.mmaa_vs_mmut.note}</div>
              <div className="table-responsive">
                <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                  <thead className="table-dark">
                    <tr>
                      <th>Feature</th>
                      <th style={{ color: '#80cbc4' }}>MMAA — cblA (this disease)</th>
                      <th style={{ color: '#ce93d8' }}>MMUT (apoenzyme defect)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.mmaa_vs_mmut.comparison || []).map((row, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{row.feature}</td>
                        <td style={{
                          color: row.MMAA_cblA?.includes('STRONG') ? ACCENT2 :
                                 row.MMAA_cblA?.includes('NORMAL') || row.MMAA_cblA?.includes('INTACT') ? ACCENT6 :
                                 row.MMAA_cblA?.includes('BLOCKED') ? '#b71c1c' :
                                 'inherit'
                        }}>{row.MMAA_cblA}</td>
                        <td style={{
                          color: row.MMUT?.includes('DEFICIENT') || row.MMUT?.includes('NONE') ? '#b71c1c' :
                                 row.MMUT?.includes('~20%') ? ACCENT3 :
                                 'inherit'
                        }}>{row.MMUT}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          )}

          {/* High-Risk Situations */}
          <Section title="High-Risk Situations" color="#b71c1c">
            {(ov.high_risk_situations || []).map((r, i) => (
              <div key={i} className="d-flex align-items-start mb-2">
                <span className={`badge me-2 ${
                  r.risk === 'ABSOLUTE CI' ? 'bg-danger' :
                  r.risk === 'EXTREME HAZARD' ? 'bg-warning text-dark' :
                  r.risk === 'HIGH RISK' ? 'bg-danger' :
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
          <Section title="Biomarkers (11 key measures — MMAA / cblA)" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr>
                    <th>Biomarker</th><th>Normal Range</th><th>MMAA Range</th>
                    <th>Significance</th><th>Method</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.biomarkers || []).map((b, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{b.name}</td>
                      <td>{b.normal}</td>
                      <td style={{
                        color: b.mmaa_range?.includes('NORMAL') ? ACCENT6 :
                               b.mmaa_range?.includes('PATHOGNOMONIC') ? ACCENT3 :
                               b.mmaa_range?.includes('CORRECTABLE') ? ACCENT2 :
                               ACCENT3
                      }}>
                        {b.mmaa_range}
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
          <Section title="Key Pathogenic Variants in MMAA (GTPase Chaperone)" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Variant</th><th>cDNA</th><th>Domain</th><th>Severity</th><th>Note</th></tr>
                </thead>
                <tbody>
                  {(bk.key_variants || []).map((v, i) => (
                    <tr key={i}>
                      <td className="fw-semibold font-monospace">{v.variant}</td>
                      <td className="font-monospace small">{v.cdna}</td>
                      <td>{v.domain}</td>
                      <td>
                        <span className={`badge ${
                          v.severity?.includes('non-responsive') || v.severity?.includes('null') ? 'bg-danger' :
                          v.severity?.includes('Severe') ? 'bg-danger' :
                          v.severity?.includes('Moderate-Severe') ? 'bg-warning text-dark' :
                          v.severity?.includes('Moderate') ? 'bg-warning text-dark' :
                          'bg-secondary'
                        }`}>
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
          <Section title="Patient Cohort Sample (15 of 40 — MMAA cblA)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 11 }}>
                <thead className="table-dark">
                  <tr>
                    <th>ID</th><th>Sex</th><th>Phenotype</th><th>Diag (mo)</th>
                    <th>MMA Urine (mmol/mol Cr)</th><th>Plasma MMA (µmol/L)</th>
                    <th>C3 (µmol/L)</th><th>NH₃ (µmol/L)</th>
                    <th>Carnitine</th><th>eGFR</th>
                    <th>Seizures</th><th>Cardio</th>
                    <th>NBS</th><th>OHCbl Resp</th><th>AED</th><th>Genotype</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.patient_sample || []).map((p, i) => (
                    <tr key={i}>
                      <td className="font-monospace">{p.patient_id}</td>
                      <td>{p.sex}</td>
                      <td style={{ fontSize: 10 }}>{p.phenotype}</td>
                      <td>{p.age_at_diagnosis_months}</td>
                      <td style={{ color: p.mma_urine_mmol_molCr > 500 ? ACCENT3 : 'inherit', fontWeight: 'bold' }}>{p.mma_urine_mmol_molCr?.toLocaleString()}</td>
                      <td style={{ color: p.plasma_mma_umol_l > 200 ? ACCENT3 : 'inherit' }}>{p.plasma_mma_umol_l}</td>
                      <td style={{ color: p.c3_umol_l > 5 ? ACCENT3 : 'inherit' }}>{p.c3_umol_l}</td>
                      <td style={{ color: p.ammonia_umol_l > 150 ? '#b71c1c' : 'inherit' }}>{p.ammonia_umol_l}</td>
                      <td style={{ color: p.free_carnitine_umol_l < 20 ? ACCENT5 : 'inherit' }}>{p.free_carnitine_umol_l}</td>
                      <td style={{ color: p.egfr_ml_min_1_73m2 < 60 ? ACCENT5 : 'inherit' }}>{p.egfr_ml_min_1_73m2}</td>
                      <td>{p.seizures ? '✓' : '—'}</td>
                      <td style={{ color: p.cardiomyopathy ? ACCENT5 : 'inherit' }}>{p.cardiomyopathy ? '✓ DCM' : '—'}</td>
                      <td>{p.nbs_detected ? '✓' : '✗'}</td>
                      <td style={{ color: p.ohcbl_response ? ACCENT2 : 'inherit', fontWeight: p.ohcbl_response ? 'bold' : 'normal' }}>{p.ohcbl_response ? '✓ B12 resp' : '—'}</td>
                      <td className="text-muted" style={{ fontSize: 10 }}>{p.aed}</td>
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
          <Section title="Seizure Types in MMAA (cblA Methylmalonic Acidemia)" color="#b71c1c">
            <div className="row g-2">
              {(bk.seizure_types || []).map((s, i) => (
                <div key={i} className="col-12 col-md-6">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-body py-2 px-3">
                      <div className="d-flex justify-content-between align-items-center mb-1">
                        <span className="fw-semibold small">{s.type}</span>
                        <span className="badge" style={{ backgroundColor: '#b71c1c' }}>{s.pct}%</span>
                      </div>
                      <div className="progress mb-1" style={{ height: 6 }}>
                        <div className="progress-bar" style={{ width: `${s.pct}%`, backgroundColor: '#b71c1c' }} />
                      </div>
                      <div className="text-muted" style={{ fontSize: 11 }}>{s.note}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Metabolic Triggers */}
          <Section title="Metabolic Crisis Triggers in MMAA" color={ACCENT3}>
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
          <Section title="High-Risk Drugs / Substances in MMAA" color="#b71c1c">
            {(bk.high_risk_drugs || []).map((d, i) => (
              <div key={i} className="d-flex align-items-start mb-2">
                <span className={`badge me-2 ${
                  d.risk === 'ABSOLUTE CI' ? 'bg-danger' :
                  d.risk === 'EXTREME HAZARD' ? 'bg-warning text-dark' :
                  d.risk === 'HIGH RISK' ? 'bg-danger' :
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
            text="ℹ️ MMAA (cblA) TREATMENT KEY: Hydroxocobalamin (OHCbl) 1–2 mg/day IM (Level A) — trial ALL patients; >60–80% respond (HALLMARK cblA; strongest B12 response of all isolated MMA). MMUT apoenzyme is INTACT — OHCbl overcomes the AdoCbl delivery defect. L-Carnitine (Level A) for secondary depletion. Liver transplant RARELY needed (unlike MMUT mut0). VPA ABSOLUTE CI — inhibits cobalamin pathway + carnitine depletion: use LEV as first-line AED. Missed OHCbl doses = HIGH RISK."
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
          <Section title="Gene Card — MMAA (cblA Chaperone)" color={ACCENT7}>
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
          <Section title="Key Concepts — MMAA / cblA Cobalamin-A Chaperone" color={ACCENT}>
            {(df.key_concepts || []).map((c, i) => (
              <div key={i} className="card border-0 shadow-sm mb-2">
                <div className="card-body py-2 px-3">
                  <div className="fw-semibold small mb-1" style={{ color: i === 1 ? ACCENT2 : i === 4 ? '#b71c1c' : ACCENT }}>{c.concept}</div>
                  <div className="text-muted" style={{ fontSize: 12 }}>{c.explanation}</div>
                </div>
              </div>
            ))}
          </Section>

          {/* Diagnostic Thresholds */}
          <Section title="Diagnostic Thresholds — MMAA (cblA)" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr>
                </thead>
                <tbody>
                  {(df.diagnostic_thresholds || []).map((d, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{d.parameter}</td>
                      <td style={{ color: d.parameter?.includes('OHCbl') ? ACCENT2 : ACCENT3 }}>{d.threshold}</td>
                      <td>{d.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Differential Diagnosis */}
          <Section title="Differential Diagnosis — vs Other Isolated MMA and Combined Disorders" color={ACCENT6}>
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

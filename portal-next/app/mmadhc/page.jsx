'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — MMADHC / cblD distributor
const ACCENT2 = '#004d40';   // dark teal — cytoplasmic arm B / HHcy treatment
const ACCENT3 = '#b71c1c';   // deep red — MMA accumulation / mitochondrial arm A
const ACCENT4 = '#1565c0';   // blue — treatment / Level A
const ACCENT5 = '#e65100';   // deep orange — NBS miss / cblD-HHcy invisible
const ACCENT6 = '#1b5e20';   // dark green — KEY NEGATIVES
const ACCENT7 = '#37474f';   // slate — variant data / gene card
const ACCENT8 = '#4a0072';   // deep purple — cobalamin distributor / branch point

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

export default function MMADHCPage() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mmadhc/overview`).then(r => r.json()),
      fetch(`${API}/api/mmadhc/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mmadhc/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!ov)  return <div className="text-center p-5 text-muted">Loading MMADHC Dashboard…</div>;

  const kpis = ov.kpis || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT8} 100%)`, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">🧬 MMADHC (cblD) Epilepsy Dashboard</h4>
        <div style={{ fontSize: 13 }}>
          {ov.full_name} &nbsp;·&nbsp; {ov.chromosome} &nbsp;·&nbsp; {ov.inheritance} &nbsp;·&nbsp; OMIM {ov.omim_gene} / {ov.omim_disease}
        </div>
        <div className="mt-1" style={{ fontSize: 12, opacity: 0.85 }}>
          {ov.protein_size} &nbsp;·&nbsp; Prevalence: {ov.prevalence}
        </div>
      </div>

      {/* Unique cblD alert */}
      <div className="alert alert-info py-2 mb-3" style={{ fontSize: 13, borderLeft: `4px solid ${ACCENT8}` }}>
        <strong>UNIQUE cblD Feature:</strong> MMADHC is the ONLY cobalamin gene where the variant's domain location
        PREDICTS which metabolic arm is blocked: C-terminus variants → isolated MMA only (Hcy NORMAL);
        N-terminus variants → isolated HHcy only (MMA NORMAL); global variants → combined MMA+HHcy (like cblC).
        <br />
        <strong>NBS WARNING:</strong> cblD-HHcy subtype has NORMAL C3 on NBS — HIGH MISS RATE by standard newborn screening.
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          {/* KPIs */}
          <div className="row g-2 mb-4">
            <KPI label="Cohort (N)" value={ov.cohort_n} color={ACCENT} />
            <KPI label="cblD-MMA only" value={`${kpis.mma_subtype_pct}%`} color={ACCENT3} />
            <KPI label="cblD-HHcy only" value={`${kpis.hhcy_subtype_pct}%`} color={ACCENT2} />
            <KPI label="cblD-Combined" value={`${kpis.combined_subtype_pct}%`} color={ACCENT8} />
            <KPI label="Avg MMA (urine)" value={`${kpis.avg_mma_urine} mmol/mol`} color={ACCENT3} />
            <KPI label="Avg tHcy (µmol/L)" value={kpis.avg_homocysteine_umol_l} color={ACCENT2} />
            <KPI label="Avg Methionine" value={`${kpis.avg_methionine_umol_l} µmol/L`} color={ACCENT4} />
            <KPI label="Avg C3 (µmol/L)" value={kpis.avg_c3_umol_l} color={ACCENT3} />
            <KPI label="Avg NH3 (µmol/L)" value={kpis.avg_ammonia_umol_l} color={ACCENT5} />
            <KPI label="Seizures" value={`${kpis.seizure_pct}%`} color={ACCENT3} />
            <KPI label="OHCbl Response" value={`${kpis.ohcbl_response_pct}%`} color={ACCENT4} />
            <KPI label="NBS Detected" value={`${kpis.nbs_detected_pct}%`} color={ACCENT5} />
          </div>

          {/* Subtype distribution */}
          <Section title="cblD Subtype Distribution — Genotype Predicts Biochemical Arm" color={ACCENT8}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered">
                <thead className="table-dark">
                  <tr><th>Subtype</th><th>Prevalence</th><th>NBS</th><th>MMA</th><th>tHcy</th></tr>
                </thead>
                <tbody>
                  {(ov.subtype_distribution || []).map((s, i) => (
                    <tr key={i}>
                      <td style={{ fontSize: 12 }}>{s.subtype}</td>
                      <td className="text-center"><span className="badge" style={{ background: ACCENT8 }}>{s.pct}%</span></td>
                      <td style={{ fontSize: 12, color: s.nbs_detected.includes('MISSED') ? ACCENT5 : ACCENT6 }}>{s.nbs_detected}</td>
                      <td className="text-center"><span className="badge" style={{ background: s.mma === 'ELEVATED' ? ACCENT3 : ACCENT6 }}>{s.mma}</span></td>
                      <td className="text-center"><span className="badge" style={{ background: s.hcy === 'ELEVATED' ? ACCENT2 : ACCENT6 }}>{s.hcy}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Pathway */}
          <Section title="Intracellular Cobalamin Pathway — MMADHC Position" color={ACCENT}>
            {(ov.mmadhc_pathway || []).map((step, i) => (
              <div key={i} className="mb-2 p-2 border rounded" style={{ fontSize: 12, borderLeft: `3px solid ${i === 1 ? ACCENT3 : ACCENT}` }}>
                <div className="fw-bold">{step.step}</div>
                <div className="text-muted">{step.reaction}</div>
                <div><em>Enzyme:</em> {step.enzyme}</div>
                {step.consequence_lof && (
                  <div style={{ color: ACCENT3, fontWeight: 500 }}>LOF: {step.consequence_lof}</div>
                )}
              </div>
            ))}
          </Section>

          {/* Function + Mechanism */}
          <Section title="Function & Mechanism" color={ACCENT4}>
            <div className="p-2 border rounded mb-2" style={{ fontSize: 13 }}>
              <strong>Function:</strong> {ov.function}
            </div>
            <div className="p-2 border rounded mb-2" style={{ fontSize: 13 }}>
              <strong>Mechanism:</strong> {ov.mechanism}
            </div>
            <div className="p-2 border rounded" style={{ fontSize: 13 }}>
              <strong>Key Negatives:</strong> {ov.key_negative}
            </div>
          </Section>

          {/* High-risk situations */}
          <Section title="High-Risk Situations & Contraindications" color={ACCENT3}>
            {(ov.high_risk_situations || []).map((h, i) => (
              <div key={i} className="mb-2 p-2 border rounded" style={{ fontSize: 12, borderLeft: `3px solid ${h.risk.includes('ABSOLUTE') ? '#b71c1c' : '#e65100'}` }}>
                <div className="fw-bold">{h.situation} — <span style={{ color: ACCENT3 }}>{h.risk}</span></div>
                <div className="text-muted">{h.detail}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── TAB 1: PATIENTS & BIOMARKERS ── */}
      {tab === 1 && bk && (
        <div>
          {/* Biomarker table */}
          <Section title="Biomarkers by MMADHC Subtype" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 11 }}>
                <thead className="table-dark">
                  <tr><th>Biomarker</th><th>Normal</th><th>MMADHC Range</th><th>Significance</th></tr>
                </thead>
                <tbody>
                  {(bk.biomarkers || []).map((b, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{b.name}</td>
                      <td>{b.normal}</td>
                      <td style={{ color: ACCENT3 }}>{b.mmadhc_range}</td>
                      <td>{b.significance}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Key variants */}
          <Section title="Key Variants — Domain Predicts Subtype" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 11 }}>
                <thead className="table-dark">
                  <tr><th>Variant</th><th>cDNA</th><th>Domain</th><th>Subtype</th><th>Severity</th><th>Note</th></tr>
                </thead>
                <tbody>
                  {(bk.key_variants || []).map((v, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{v.variant}</td>
                      <td>{v.cdna}</td>
                      <td style={{ color: ACCENT8, fontSize: 11 }}>{v.domain}</td>
                      <td><span className="badge" style={{
                        background: v.subtype?.includes('MMA only') ? ACCENT3 : v.subtype?.includes('HHcy only') ? ACCENT2 : ACCENT8,
                        fontSize: 10
                      }}>{v.subtype}</span></td>
                      <td>{v.severity}</td>
                      <td style={{ fontSize: 10 }}>{v.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Patient sample */}
          <Section title={`Patient Sample (first 15 of ${ov.cohort_n})`} color={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered table-striped" style={{ fontSize: 11 }}>
                <thead className="table-dark">
                  <tr>
                    <th>ID</th><th>Sex</th><th>Subtype</th><th>Phenotype</th>
                    <th>Onset (mo)</th><th>MMA (mmol/mol)</th><th>tHcy (µmol)</th>
                    <th>Met (µmol)</th><th>C3</th><th>Seizures</th><th>NBS</th><th>OHCbl Resp</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.patient_sample || []).map((p, i) => (
                    <tr key={i}>
                      <td>{p.patient_id}</td>
                      <td>{p.sex}</td>
                      <td><span className="badge" style={{
                        background: p.subtype?.includes('MMA only') ? ACCENT3 : p.subtype?.includes('HHcy only') ? ACCENT2 : ACCENT8,
                        fontSize: 9
                      }}>{p.subtype?.replace('cblD-', '')}</span></td>
                      <td style={{ fontSize: 10 }}>{p.phenotype}</td>
                      <td className="text-center">{p.age_at_onset_months}</td>
                      <td className="text-center" style={{ color: p.mma_urine_mmol_molCr > 10 ? ACCENT3 : ACCENT6 }}>{p.mma_urine_mmol_molCr}</td>
                      <td className="text-center" style={{ color: p.total_homocysteine_umol_l > 15 ? ACCENT2 : ACCENT6 }}>{p.total_homocysteine_umol_l}</td>
                      <td className="text-center" style={{ color: p.methionine_umol_l < 18 ? ACCENT2 : '' }}>{p.methionine_umol_l}</td>
                      <td className="text-center">{p.c3_umol_l}</td>
                      <td className="text-center">{p.seizures ? '✓' : '—'}</td>
                      <td className="text-center">{p.nbs_detected ? '✓' : <span style={{ color: ACCENT5 }}>✗</span>}</td>
                      <td className="text-center">{p.ohcbl_response ? '✓' : '—'}</td>
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
          <Section title="Seizure Types in MMADHC (cblD)" color={ACCENT3}>
            {(bk.seizure_types || []).map((s, i) => (
              <div key={i} className="mb-2">
                <PctBar label={s.type} pct={s.pct} color={ACCENT3} />
                <div className="text-muted small ms-2">{s.note}</div>
              </div>
            ))}
          </Section>

          <Section title="Metabolic Triggers" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Trigger</th><th>% Patients</th><th>Mechanism</th></tr>
                </thead>
                <tbody>
                  {(bk.metabolic_triggers || []).map((t, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{t.trigger}</td>
                      <td className="text-center">{t.pct}%</td>
                      <td style={{ fontSize: 11 }}>{t.mechanism}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="High-Risk Drugs" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Drug</th><th>Risk</th><th>Mechanism</th></tr>
                </thead>
                <tbody>
                  {(bk.high_risk_drugs || []).map((d, i) => (
                    <tr key={i} style={{ backgroundColor: d.risk.includes('ABSOLUTE') ? '#fff3f3' : '' }}>
                      <td className="fw-bold">{d.drug}</td>
                      <td style={{ color: ACCENT3, fontWeight: 600 }}>{d.risk}</td>
                      <td style={{ fontSize: 11 }}>{d.mechanism}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 3: TREATMENTS ── */}
      {tab === 3 && bk && (
        <div>
          <Alert text="OHCbl trial ALL subtypes (Level A). Betaine Level A ONLY if HHcy component present (cblD-HHcy or cblD-Combined) — NOT for cblD-MMA only. Low protein + carnitine for MMA subtypes. VPA ABSOLUTE CI (MMA subtypes) / HIGH RISK (HHcy subtypes). LEV first-line AED for ALL cblD seizure types." variant="info" />
          <Section title="Treatments — Subtype-Specific Management" color={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Treatment</th><th>Evidence</th><th>Response</th><th>Note</th></tr>
                </thead>
                <tbody>
                  {(bk.treatments || []).map((t, i) => (
                    <tr key={i} style={{ backgroundColor: t.evidence === 'AVOID' ? '#fff3f3' : '' }}>
                      <td className="fw-bold">{t.treatment}</td>
                      <td>
                        <span className="badge" style={{
                          background: t.evidence === 'Level A' ? ACCENT4
                            : t.evidence === 'Level B' ? ACCENT2
                            : t.evidence === 'AVOID' ? ACCENT3
                            : ACCENT7
                        }}>{t.evidence}</span>
                      </td>
                      <td className="text-center">
                        {t.response_pct > 0
                          ? <><div className="progress" style={{ height: 8 }}><div className="progress-bar" style={{ width: `${t.response_pct}%`, backgroundColor: ACCENT4 }} /></div><small>{t.response_pct}%</small></>
                          : <span className="text-danger fw-bold">AVOID</span>}
                      </td>
                      <td style={{ fontSize: 11 }}>{t.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 4: DEFINITIONS ── */}
      {tab === 4 && df && (
        <div>
          {/* Gene card */}
          <Section title="Gene Card — MMADHC (cblD)" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <tbody>
                  {Object.entries(df.gene_card || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold text-nowrap" style={{ width: '28%' }}>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Key concepts */}
          <Section title="Key Concepts" color={ACCENT4}>
            {(df.key_concepts || []).map((c, i) => (
              <div key={i} className="mb-3 p-3 border rounded" style={{ borderLeft: `3px solid ${ACCENT4}` }}>
                <div className="fw-bold mb-1" style={{ color: ACCENT4, fontSize: 13 }}>{c.concept}</div>
                <div style={{ fontSize: 12, lineHeight: 1.6 }}>{c.explanation}</div>
              </div>
            ))}
          </Section>

          {/* Diagnostic thresholds */}
          <Section title="Diagnostic Thresholds & Actions" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr>
                </thead>
                <tbody>
                  {(df.diagnostic_thresholds || []).map((d, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{d.parameter}</td>
                      <td style={{ color: ACCENT3 }}>{d.threshold}</td>
                      <td style={{ fontSize: 11 }}>{d.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Differential diagnosis */}
          <Section title="Differential Diagnosis" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Disease</th><th>Key Distinguishing Feature</th></tr>
                </thead>
                <tbody>
                  {(df.differential_diagnosis || []).map((d, i) => (
                    <tr key={i}>
                      <td className="fw-bold" style={{ width: '30%' }}>{d.disease}</td>
                      <td style={{ color: ACCENT6, fontSize: 11 }}>{d.distinguishing}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}
    </div>
  );
}

'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#0d47a1';   // deep blue — SLC6A8 / X-linked / transport
const ACCENT2 = '#b71c1c';   // deep red — seizures / creatine tx failure
const ACCENT3 = '#1b5e20';   // deep green — NORMAL values (GAA, tHcy, MMA) / key negatives
const ACCENT4 = '#e65100';   // amber-orange — urine Cr/CrCr ELEVATED / warning
const ACCENT5 = '#4a148c';   // deep purple — IDD / speech absent
const ACCENT6 = '#006064';   // teal — X-linked / male vs female
const ACCENT7 = '#37474f';   // slate — drug risks
const ACCENT8 = '#880e4f';   // dark pink — creatine TX fails (distinctive warning)

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

export default function SLC6A8Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [br, setBr]       = useState(null);
  const [df, setDf]       = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/slc6a8/overview`).then(r => r.json()),
      fetch(`${API}/api/slc6a8/breakdown`).then(r => r.json()),
      fetch(`${API}/api/slc6a8/definitions`).then(r => r.json()),
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
      <div className="mb-3 p-3 rounded text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT6} 100%)` }}>
        <div className="d-flex justify-content-between align-items-start flex-wrap gap-2">
          <div>
            <h4 className="mb-1 fw-bold">🧬 SLC6A8 — Creatine Transporter Deficiency (CCDS1)</h4>
            <div style={{ fontSize: 13, opacity: 0.92 }}>{ov.subtitle}</div>
          </div>
          <div className="text-end" style={{ fontSize: 12, opacity: 0.85 }}>
            <div>X-linked · Xq28 · OMIM 300036/300352</div>
            <div>Most common of 3 CCDS · ~300–500 cases worldwide</div>
            <div className="mt-1"><strong>CCDS1</strong> · Transport failure (not biosynthesis)</div>
          </div>
        </div>
      </div>

      {/* CRITICAL ALERT — creatine tx fails */}
      <div className="alert alert-danger py-2 mb-3" style={{ fontSize: 13, borderLeft: `5px solid ${ACCENT8}` }}>
        <strong>⚠️ CREATINE Tx FAILS IN HEMIZYGOUS MALES:</strong> Creatine monohydrate is largely ineffective —
        the transporter (SLC6A8) required to import creatine into brain/muscle is absent.
        Plasma creatine rises further without brain benefit. <strong>Do NOT prescribe standard creatine in males.</strong>{' '}
        Emerging therapy: GAA supplementation (enters via SLC6A6/BGT-1 — bypasses absent SLC6A8).
      </div>

      {/* Nav tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button className={`nav-link${tab === i ? ' active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Patients"          value={kpi.n_patients}         color={ACCENT} />
            <KPI label="Males (severe)"    value={kpi.n_males}            color={ACCENT6} />
            <KPI label="Females (mosaic)"  value={kpi.n_females}          color={ACCENT4} />
            <KPI label="Plasma Creatine ↑" value={`${kpi.avg_creatine_plasma} µM`} color={ACCENT2} />
            <KPI label="Urine Cr/CrCr ↑↑" value={kpi.avg_cr_ratio_males} color={ACCENT8} />
            <KPI label="GAA → Normal"      value={`${kpi.avg_gaa} µM`}    color={ACCENT3} />
            <KPI label="Seizures"          value={`${kpi.pct_seizures}%`} color={ACCENT2} />
            <KPI label="Drug-Resistant"    value={`${kpi.pct_drug_resistant}%`} color={ACCENT4} />
            <KPI label="IDD"               value={`${kpi.pct_idd}%`}      color={ACCENT5} />
            <KPI label="Cr Tx Fails (♂)"   value={`${kpi.pct_creatine_tx_fail_male}%`} color={ACCENT8} />
          </div>

          {/* Pathway */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>
              Creatine Biosynthesis & Transport Pathway — CCDS1 Blocks at Step 3
            </div>
            <div className="card-body">
              {ov.pathway_diagram && (
                <div className="row g-3">
                  {[
                    { step: 'step1', label: 'Step 1 (AGAT)', icon: '✅' },
                    { step: 'step2', label: 'Step 2 (GAMT)', icon: '✅' },
                    { step: 'step3', label: 'Step 3 (SLC6A8)', icon: '✗' },
                  ].map(({ step, label, icon }) => (
                    <div className="col-md-4" key={step}>
                      <div className={`card h-100 border-${step === 'step3' ? 'danger' : 'success'}`}>
                        <div className={`card-header small fw-bold text-${step === 'step3' ? 'danger' : 'success'}`}>
                          {icon} {label}
                        </div>
                        <div className="card-body py-2 small">{ov.pathway_diagram[step]}</div>
                      </div>
                    </div>
                  ))}
                  <div className="col-12">
                    <div className="alert alert-warning py-2 mb-1 small">
                      <strong>Consequence:</strong> {ov.pathway_diagram.consequence}
                    </div>
                    <div className="alert alert-info py-2 mb-0 small">
                      <strong>Brain H-MRS:</strong> {ov.pathway_diagram.h_mrs}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Biomarker signature */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT4 }}>
              Biomarker Signature — Transport Failure Profile
            </div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-bordered">
                  <thead className="table-dark">
                    <tr>
                      <th>Biomarker</th><th>Direction</th><th>Significance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ov.biomarker_signature && Object.entries(ov.biomarker_signature).map(([k, v]) => (
                      <tr key={k}>
                        <td><strong>{k.replace(/_/g,' ')}</strong></td>
                        <td>
                          <span style={{
                            color: v.direction?.includes('↑') ? ACCENT2 :
                                   v.direction?.includes('↓') ? ACCENT4 :
                                   ACCENT3,
                            fontWeight: 'bold',
                          }}>
                            {v.direction}
                          </span>
                        </td>
                        <td className="small">{v.significance || v.value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* CCDS Triad Comparison */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>
              CCDS Triad Comparison — All 3 Cerebral Creatine Deficiency Syndromes
            </div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-bordered">
                  <thead className="table-dark">
                    <tr>
                      <th>Disease</th><th>Step</th><th>Plasma Creatine</th>
                      <th>Urine Creatine</th><th>GAA</th><th>Creatine Tx</th><th>Inheritance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.ccds_triad_comparison || []).map((row, i) => (
                      <tr key={i} style={{ background: i === 0 ? '#e3f2fd' : 'inherit' }}>
                        <td><strong>{row.disease}</strong></td>
                        <td className="small">{row.step}</td>
                        <td style={{ color: row.plasma_creatine?.includes('↑') ? ACCENT2 : ACCENT3 }}>
                          <strong>{row.plasma_creatine}</strong>
                        </td>
                        <td style={{ color: row.urine_creatine?.includes('↑') ? ACCENT2 : ACCENT3 }}>
                          <strong>{row.urine_creatine}</strong>
                        </td>
                        <td className="small">{row.gaa}</td>
                        <td className="small" style={{ color: row.tx_creatine?.includes('FAIL') ? ACCENT8 : ACCENT3 }}>
                          <strong>{row.tx_creatine}</strong>
                        </td>
                        <td>{row.inheritance}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Key clinical distinctions */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>
              Key Clinical Distinctions
            </div>
            <div className="card-body">
              <ul className="mb-0 small">
                {(ov.key_clinical_distinctions || []).map((d, i) => (
                  <li key={i} className="mb-1">{d}</li>
                ))}
              </ul>
            </div>
          </div>

          {/* NBS */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT7 }}>
              Newborn Screening (NBS) Status
            </div>
            <div className="card-body small">
              {ov.nbs_status && Object.entries(ov.nbs_status).map(([k, v]) => (
                <div key={k} className="mb-1">
                  <strong>{k.replace(/_/g, ' ')}:</strong> {v}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 1: PATIENTS & BIOMARKERS ── */}
      {tab === 1 && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>
                  Clinical Features — All Patients (n=40)
                </div>
                <div className="card-body">
                  <PctBar label="Seizures"            pct={kpiPct.seizures}       color={ACCENT2} />
                  <PctBar label="Drug-Resistant Sz"   pct={kpiPct.drug_resistant} color={ACCENT4} />
                  <PctBar label="IDD"                 pct={kpiPct.idd}            color={ACCENT5} />
                  <PctBar label="Speech Absent"       pct={kpiPct.speech_absent}  color={ACCENT5} />
                  <PctBar label="Autism-Like"         pct={kpiPct.autism_like}    color={ACCENT7} />
                  <PctBar label="Behavioral / ADHD-like" pct={kpiPct.behavioral}  color={ACCENT6} />
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT4 }}>
                  Phenotype Distribution
                </div>
                <div className="card-body">
                  {(ov.phenotype_distribution || []).map((pd, i) => (
                    <PctBar key={i} label={pd.name} pct={pd.pct}
                      color={i === 0 ? ACCENT : i === 1 ? ACCENT6 : ACCENT3} />
                  ))}
                  <div className="alert alert-info py-2 mt-2 small">
                    <strong>X-linked:</strong> Hemizygous males = severe; carrier females = variable (Lyon mosaicism)
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Variant distribution */}
          <div className="card mb-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT }}>
              Variant Distribution (40 patients, seed 127)
            </div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead className="table-dark"><tr><th>Variant</th><th>N</th></tr></thead>
                  <tbody>
                    {(br?.variant_distribution || []).map((v, i) => (
                      <tr key={i}>
                        <td><code>{v.variant}</code></td>
                        <td><strong>{v.n}</strong></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Patient table */}
          <div className="card mb-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT }}>
              Patient Cohort — Synthetic Data (seed 127)
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0" style={{ fontSize: 11 }}>
                  <thead className="table-dark">
                    <tr>
                      <th>ID</th><th>Sex</th><th>Phenotype</th><th>Variant</th>
                      <th>Onset (mo)</th><th>Plasma Cr ↑</th><th>Urine Cr/CrCr ↑↑</th>
                      <th>GAA →</th><th>Sz</th><th>DR</th><th>IDD</th><th>Speech-</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(br?.patients || []).map(p => (
                      <tr key={p.id} style={{ background: p.drug_resistant ? '#fff3f3' : 'inherit' }}>
                        <td>{p.id}</td>
                        <td style={{ color: p.sex === 'M' ? ACCENT : ACCENT6 }}>
                          <strong>{p.sex}</strong>
                        </td>
                        <td style={{ fontSize: 10 }}>{p.phenotype}</td>
                        <td><code style={{ fontSize: 10 }}>{p.variant}</code></td>
                        <td>{p.age_onset_months}</td>
                        <td style={{ color: ACCENT2 }}><strong>{p.creatine_plasma}</strong></td>
                        <td style={{ color: ACCENT8 }}><strong>{p.urine_cr_ratio}</strong></td>
                        <td style={{ color: ACCENT3 }}>{p.gaa}</td>
                        <td>{p.seizures ? '✓' : '–'}</td>
                        <td style={{ color: p.drug_resistant ? ACCENT2 : 'inherit' }}>
                          {p.drug_resistant ? '✓' : '–'}
                        </td>
                        <td>{p.idd ? '✓' : '–'}</td>
                        <td>{p.speech_absent ? '✓' : '–'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: SEIZURES & TRIGGERS ── */}
      {tab === 2 && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT2 }}>
                  Seizure Types
                </div>
                <div className="card-body">
                  {(br?.seizure_types || []).map((s, i) => (
                    <div key={i} className="mb-2 d-flex justify-content-between align-items-center">
                      <span className="small">{s.type}</span>
                      <span className="badge ms-2" style={{ background: ACCENT2 }}>n={s.n}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT4 }}>
                  Metabolic Triggers & Precipitants
                </div>
                <div className="card-body">
                  {(br?.metabolic_triggers || []).map((t, i) => (
                    <div key={i} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <strong>{t.trigger}</strong>
                        <span className="text-muted">{t.pct}%</span>
                      </div>
                      <div className="progress mb-1" style={{ height: 8 }}>
                        <div className="progress-bar" style={{ width: `${t.pct}%`, backgroundColor: ACCENT4 }} />
                      </div>
                      <div style={{ fontSize: 11, color: '#666' }}>{t.mechanism}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="card mb-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT }}>
              Epilepsy Burden — CCDS1 vs Other CCDS
            </div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-bordered">
                  <thead className="table-dark">
                    <tr><th>Syndrome</th><th>Seizure Rate</th><th>Drug-Resistant</th><th>Mechanism</th></tr>
                  </thead>
                  <tbody>
                    <tr style={{ background: '#e3f2fd' }}>
                      <td><strong>CCDS1 — SLC6A8</strong></td>
                      <td>80–90% (males)</td>
                      <td>50–60%</td>
                      <td>Brain creatine-phosphate energy buffer absent</td>
                    </tr>
                    <tr>
                      <td>CCDS2 — GAMT</td>
                      <td>80–95%</td>
                      <td>60–80%</td>
                      <td>Creatine absent + GAA GABA-A inhibition / NMDA activation</td>
                    </tr>
                    <tr>
                      <td>CCDS3 — AGAT</td>
                      <td>50–60%</td>
                      <td>25–35%</td>
                      <td>Creatine absent only (no GAA toxicity — milder)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: TREATMENTS ── */}
      {tab === 3 && (
        <div>
          <Alert
            text="⚠️ Creatine monohydrate is LARGELY INEFFECTIVE in hemizygous males — do not prescribe. GAA supplementation is the emerging strategy (enters via SLC6A6/BGT-1)."
            variant="danger"
          />

          <div className="card mb-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT }}>
              Treatments — Evidence Levels
            </div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-bordered">
                  <thead className="table-dark">
                    <tr><th>Treatment</th><th>Level</th><th>Mechanism</th><th>Note</th></tr>
                  </thead>
                  <tbody>
                    {(br?.treatments || []).map((t, i) => (
                      <tr key={i} style={{
                        background: t.level === 'NOT RECOMMENDED for hemizygous males' ? '#fff3f3' : 'inherit',
                      }}>
                        <td><strong>{t.treatment}</strong></td>
                        <td>
                          <span className={`badge bg-${
                            t.level.startsWith('Level A') ? 'success' :
                            t.level.startsWith('Level B') ? 'primary' :
                            t.level.includes('NOT') ? 'danger' :
                            'secondary'
                          }`}>{t.level}</span>
                        </td>
                        <td className="small">{t.mechanism}</td>
                        <td className="small text-muted">{t.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="card mb-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT7 }}>
              Drug Risks & Contraindications
            </div>
            <div className="card-body">
              {(br?.drug_risks || []).map((d, i) => (
                <div key={i} className={`alert alert-${
                  d.risk === 'NOT RECOMMENDED' ? 'danger' :
                  d.risk === 'MODERATE RISK' ? 'warning' : 'info'
                } py-2 mb-2 small`}>
                  <strong>{d.drug}</strong> — <span className="fw-bold">{d.risk}</span>
                  <div>{d.reason}</div>
                  <div className="text-muted">{d.action}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 4: DEFINITIONS ── */}
      {tab === 4 && (
        <div>
          {df && (
            <>
              <div className="card mb-3">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>
                  Gene Card — SLC6A8
                </div>
                <div className="card-body small">
                  <div className="row g-2">
                    {df.gene_definition && Object.entries(df.gene_definition).map(([k, v]) => (
                      <div className="col-md-6" key={k}>
                        <strong>{k.replace(/_/g, ' ')}:</strong> {v}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="card mb-3">
                <div className="card-header fw-bold small" style={{ color: ACCENT4 }}>
                  Key Concepts
                </div>
                <div className="card-body">
                  {(df.key_concepts || []).map((c, i) => (
                    <div key={i} className="mb-3 p-2 border rounded">
                      <div className="fw-bold small mb-1" style={{ color: ACCENT }}>{c.concept}</div>
                      <div className="small">{c.detail}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="card mb-3">
                <div className="card-header fw-bold small" style={{ color: ACCENT7 }}>
                  Differential Diagnosis
                </div>
                <div className="card-body">
                  <div className="table-responsive">
                    <table className="table table-sm table-bordered">
                      <thead className="table-dark">
                        <tr><th>Disease</th><th>Key Distinction</th><th>Plasma Creatine</th><th>Urine Creatine</th><th>Creatine Tx</th></tr>
                      </thead>
                      <tbody>
                        {(df.differential_diagnosis || []).map((d, i) => (
                          <tr key={i}>
                            <td><strong>{d.disease}</strong></td>
                            <td className="small">{d.key_distinction}</td>
                            <td className="small">{d.plasma_creatine || '—'}</td>
                            <td className="small">{d.urine_creatine || '—'}</td>
                            <td className="small">{d.creatine_tx || '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {df.treatment_summary && (
                <div className="row g-3 mb-3">
                  <div className="col-md-6">
                    <div className="card h-100 border-success">
                      <div className="card-header fw-bold small text-success">✅ Do</div>
                      <div className="card-body">
                        <ul className="mb-0 small">
                          {(df.treatment_summary.do || []).map((d, i) => <li key={i} className="mb-1">{d}</li>)}
                        </ul>
                      </div>
                    </div>
                  </div>
                  <div className="col-md-6">
                    <div className="card h-100 border-danger">
                      <div className="card-header fw-bold small text-danger">⛔ Avoid</div>
                      <div className="card-body">
                        <ul className="mb-0 small">
                          {(df.treatment_summary.avoid || []).map((d, i) => <li key={i} className="mb-1">{d}</li>)}
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="card mb-3">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>
                  Variant Catalogue (SLC6A8, Xq28)
                </div>
                <div className="card-body">
                  <div className="table-responsive">
                    <table className="table table-sm table-bordered">
                      <thead className="table-dark">
                        <tr><th>Variant</th><th>Domain</th><th>Severity</th><th>Frequency</th></tr>
                      </thead>
                      <tbody>
                        {(df.variant_summary || []).map((v, i) => (
                          <tr key={i}>
                            <td><code>{v.variant}</code></td>
                            <td className="small">{v.domain}</td>
                            <td>
                              <span className={`badge bg-${
                                v.severity === 'Severe' ? 'danger' :
                                v.severity.includes('null') || v.severity.includes('complex') ? 'dark' :
                                v.severity.includes('Moderate') ? 'warning' : 'success'
                              }`}>{v.severity}</span>
                            </td>
                            <td className="small">{v.frequency}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      <div className="mt-3 text-muted small">
        <Link href="/">← Portal Home</Link>
        {' · '}
        <Link href="/gamt">CCDS2 — GAMT</Link>
        {' · '}
        <Link href="/agat">CCDS3 — AGAT</Link>
        {' · '}
        SLC6A8 data: 40-patient synthetic cohort, seed 127
      </div>
    </div>
  );
}

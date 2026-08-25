'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// SLC25A13 / Citrin Deficiency color scheme — aspartate transport / CTLN2 / liver disease
const ACCENT  = '#0d47a1';   // deep blue — aspartate-glutamate carrier / mitochondrial
const ACCENT2 = '#b71c1c';   // deep crimson — hyperammonemia crisis / NH3 elevated
const ACCENT3 = '#2e7d32';   // deep green — carbohydrate aversion PATHOGNOMONIC / dietary
const ACCENT4 = '#e65100';   // deep orange — citrulline VERY HIGH
const ACCENT5 = '#6a1b9a';   // deep purple — threonine elevated / MAS defect
const ACCENT6 = '#1565c0';   // blue — dyslipidemia / CTLN2 hallmark
const ACCENT7 = '#c62828';   // red — VPA ABSOLUTE CI / danger
const ACCENT8 = '#4e342e';   // brown — liver disease / NICCD

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

export default function SLC25A13Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/slc25a13/overview`).then(r => r.json()),
      fetch(`${API}/api/slc25a13/breakdown`).then(r => r.json()),
      fetch(`${API}/api/slc25a13/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading SLC25A13 / Citrin Deficiency dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="card shadow mb-3" style={{ borderTop: `5px solid ${ACCENT}` }}>
        <div className="card-body pb-2">
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            &#x1f9ec; SLC25A13 Epilepsy — Citrin Deficiency (CTLN2 / NICCD / FTTDCD)
          </h4>
          <div className="text-muted small">
            SLC25A13 / Citrin / AGC2 · Aspartate Export BLOCKED · Citrulline{' '}
            <strong>VERY HIGH</strong> · Carbohydrate Aversion (PATHOGNOMONIC) ·
            Liver Transplant CURATIVE · AR · 7q21.3 · OMIM CTLN2 #603471 / NICCD #605814
          </div>
          <div className="mt-2 d-flex gap-2 flex-wrap">
            <span className="badge" style={{ background: ACCENT }}>Mitochondrial Aspartate-Glutamate Carrier</span>
            <span className="badge" style={{ background: ACCENT4 }}>Citrulline ↑↑↑ VERY HIGH</span>
            <span className="badge" style={{ background: ACCENT3 }}>Carb Aversion PATHOGNOMONIC</span>
            <span className="badge" style={{ background: ACCENT2 }}>NH3 Episodic HIGH</span>
            <span className="badge" style={{ background: ACCENT6 }}>Dyslipidemia CTLN2</span>
            <span className="badge bg-danger">VPA ABSOLUTE CI</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div className="row mb-3">
            {Object.values(kpi).map((k, i) => (
              <KPI key={i} label={k.label} value={k.value} color={k.color} />
            ))}
          </div>

          {/* Carbohydrate aversion callout — PATHOGNOMONIC */}
          <div className="alert mb-3" style={{ background: '#e8f5e9', border: `2px solid ${ACCENT3}` }}>
            <strong style={{ color: ACCENT3 }}>&#x1f331; CARBOHYDRATE AVERSION — PATHOGNOMONIC (CTLN2):</strong>{' '}
            {ov.carb_aversion_note}
          </div>

          {/* Hallmark biomarker alert */}
          <div className="alert mb-3" style={{ background: '#e3f2fd', border: `2px solid ${ACCENT}` }}>
            <strong style={{ color: ACCENT }}>&#x26a0; KEY BIOMARKERS:</strong>{' '}
            {ov.hallmark_biomarker}
          </div>

          {/* Carbohydrate risk alert */}
          <div className="alert alert-warning mb-3">
            <strong>&#x26a0; IV GLUCOSE / CARBOHYDRATES — HIGH RISK in CTLN2:</strong>{' '}
            {ov.carb_risk_note}
          </div>

          <div className="row">
            {/* Transport mechanism */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Transport Mechanism — SLC25A13 Block</h6>
                  {bd?.transport_mechanism && (
                    <div className="small">
                      <div className="mb-2 p-2 rounded" style={{ background: '#e3f2fd' }}>
                        <strong>Function:</strong> {bd.transport_mechanism.function}<br />
                        <strong>Reaction:</strong> {bd.transport_mechanism.reaction}
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#fce4ec' }}>
                        <strong style={{ color: ACCENT2 }}>Block:</strong> {bd.transport_mechanism.block}
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#fff8e1' }}>
                        <strong>Urea cycle impact:</strong> {bd.transport_mechanism.urea_cycle_impact}
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#f3e5f5' }}>
                        <strong style={{ color: ACCENT5 }}>MAS shuttle impact:</strong>{' '}
                        {bd.transport_mechanism.shuttle_impact}
                      </div>
                      <div className="p-2 rounded" style={{ background: '#e8f5e9' }}>
                        <strong style={{ color: ACCENT3 }}>PATHOGNOMONIC:</strong>{' '}
                        {bd.transport_mechanism.unique_feature}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Phenotype distribution */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT6}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT6 }}>Phenotype Distribution (n={ov.n_patients})</h6>
                  {ov.phenotype_dist?.map((pc, i) => (
                    <div key={i} className="mb-2">
                      <PctBar
                        label={`${pc.class} (n=${pc.n})`}
                        pct={pc.pct}
                        color={[ACCENT, ACCENT8, ACCENT3][i % 3]}
                      />
                    </div>
                  ))}

                  <hr className="my-2" />
                  <h6 className="fw-bold small mt-2" style={{ color: ACCENT }}>Urea Cycle Position</h6>
                  <div className="small text-muted" style={{ fontFamily: 'monospace', fontSize: '0.7rem' }}>
                    <div>CPS1 (Step 1) → Carbamoyl-P</div>
                    <div>OTC (Step 2) → Citrulline ↑↑↑ (produced normally)</div>
                    <div style={{ color: ACCENT7 }}>⚠ SLC25A13 — Aspartate export BLOCKED ↓</div>
                    <div style={{ color: '#999' }}>  ASS1 (Step 3) — ASP absent ⚠ STALLED</div>
                    <div style={{ color: ACCENT4 }}>  ↳ Citrulline ACCUMULATES (300–1000 µmol/L)</div>
                    <div style={{ color: '#bbb' }}>  ASL (Step 4) — cannot proceed</div>
                    <div style={{ color: '#bbb' }}>  ARG1 (Step 5) — cannot proceed</div>
                    <div style={{ color: ACCENT3 }}>Malate-Aspartate Shuttle: impaired → NADH↑ cytoplasm</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* vs ASS1 distinction */}
          <div className="row mb-3">
            <div className="col-12">
              <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT4}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT4 }}>SLC25A13 vs ASS1 — Both Cause High Citrulline (CRITICAL DISTINCTION)</h6>
                  <div className="small">{ov.vs_ass1_distinction}</div>
                  {bd?.differential_diagnosis?.vs_ass1_ctln1 && (
                    <div className="row mt-2">
                      <div className="col-md-6">
                        <div className="p-2 rounded" style={{ background: '#e3f2fd' }}>
                          <strong style={{ color: ACCENT }}>SLC25A13 (CTLN2) — Adult-onset:</strong><br />
                          <span className="text-muted small">{bd.differential_diagnosis.vs_ass1_ctln1.ctln2_features}</span>
                        </div>
                      </div>
                      <div className="col-md-6">
                        <div className="p-2 rounded" style={{ background: '#fff8e1' }}>
                          <strong style={{ color: ACCENT4 }}>ASS1 (CTLN1) — Neonatal-onset:</strong><br />
                          <span className="text-muted small">{bd.differential_diagnosis.vs_ass1_ctln1.ctln1_features}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Systemic features */}
          <div className="card shadow-sm mb-3" style={{ borderTop: `3px solid ${ACCENT2}` }}>
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT2 }}>Systemic Features (n={ov.n_patients})</h6>
              <div className="row">
                {bd?.systemic_features?.map((f, i) => (
                  <div key={i} className="col-md-6 mb-2">
                    <PctBar label={f.feature} pct={f.pct} color={[ACCENT8, ACCENT2, ACCENT3, ACCENT6, ACCENT5, ACCENT, ACCENT, ACCENT7][i % 8]} />
                    <div className="small text-muted ps-1">{f.note}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Liver transplant note */}
          <div className="alert mb-3" style={{ background: '#e8eaf6', border: `2px solid ${ACCENT}` }}>
            <strong style={{ color: ACCENT }}>&#x2764; LIVER TRANSPLANT — CURATIVE (CTLN2):</strong>{' '}
            {ov.liver_transplant_note}
          </div>
        </div>
      )}

      {/* ── TAB 1: Patients & Biomarkers ── */}
      {tab === 1 && (
        <div>
          <div className="row">
            {/* Biomarker table */}
            <div className="col-md-7 mb-3">
              <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Biomarker Profile — SLC25A13</h6>
                  <div className="table-responsive">
                    <table className="table table-sm table-hover small mb-0">
                      <thead>
                        <tr><th>Biomarker</th><th>Normal</th><th>Status in SLC25A13</th><th>Direction</th></tr>
                      </thead>
                      <tbody>
                        {bd?.biomarkers && Object.values(bd.biomarkers).map((bm, i) => (
                          <tr key={i}>
                            <td className="fw-bold">{bm.label}</td>
                            <td className="text-muted">{bm.normal}</td>
                            <td style={{ color: bm.color === 'danger' ? '#b71c1c' : bm.color === 'warning' ? '#e65100' : bm.color === 'success' ? '#2e7d32' : '#0d47a1' }}>
                              {bm.status}
                            </td>
                            <td><span className={`badge bg-${bm.color}`}>{bm.direction}</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

            {/* Variants */}
            <div className="col-md-5 mb-3">
              <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT5}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT5 }}>Top Pathogenic Variants (East Asian)</h6>
                  {bd?.variants?.map((v, i) => (
                    <div key={i} className="mb-2 p-2 rounded" style={{ background: i === 0 ? '#fce4ec' : i <= 2 ? '#f3e5f5' : '#f5f5f5' }}>
                      <div className="d-flex justify-content-between">
                        <span className="fw-bold small" style={{ color: ACCENT }}>{v.variant}</span>
                        <span className="badge" style={{ background: ACCENT5 }}>{v.freq}%</span>
                      </div>
                      <div className="small text-muted">{v.domain} · {v.phenotype}</div>
                      <div className="small text-muted fst-italic">{v.note}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Cohort preview */}
          <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT6}` }}>
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT6 }}>Cohort Preview (first 10 of {ov.n_patients})</h6>
              <div className="table-responsive">
                <table className="table table-sm table-hover small mb-0">
                  <thead>
                    <tr>
                      <th>ID</th><th>Phenotype</th><th>Age Onset (mo)</th>
                      <th>Citrulline (µmol/L)</th><th>NH3 Peak</th><th>Threonine</th>
                      <th>Carb Aversion</th><th>Seizures</th><th>Liver Dis.</th><th>Variant</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd?.cohort_preview?.map((p, i) => (
                      <tr key={i}>
                        <td>{p.id}</td>
                        <td>
                          <span className="badge" style={{
                            background: p.phenotype === 'CTLN2 (Adult)' ? ACCENT : p.phenotype === 'NICCD (Neonatal)' ? ACCENT8 : ACCENT3,
                            fontSize: '0.65rem'
                          }}>{p.phenotype}</span>
                        </td>
                        <td>{p.age_onset_months}</td>
                        <td className="fw-bold" style={{ color: ACCENT4 }}>{p.citrulline_umol_l}</td>
                        <td style={{ color: ACCENT2 }}>{p.nh3_peak_umol_l}</td>
                        <td style={{ color: ACCENT5 }}>{p.threonine_umol_l}</td>
                        <td>{p.carb_aversion ? <span className="text-success fw-bold">✓ PATHO</span> : '–'}</td>
                        <td>{p.seizures ? '✓' : '–'}</td>
                        <td>{p.liver_disease ? '✓' : '–'}</td>
                        <td className="text-muted" style={{ fontSize: '0.7rem' }}>{p.variant}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: Seizures & Treatments ── */}
      {tab === 2 && (
        <div>
          <div className="row">
            {/* Seizure types */}
            <div className="col-md-5 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT2}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT2 }}>Seizure Types (n={ov.n_patients})</h6>
                  {bd?.seizure_types?.map((s, i) => (
                    <div key={i} className="mb-2">
                      <PctBar label={s.type} pct={s.pct} color={[ACCENT2, ACCENT, ACCENT5, ACCENT6, ACCENT3, ACCENT8][i % 6]} />
                      <div className="small text-muted ps-1">{s.note}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Treatments */}
            <div className="col-md-7 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT3}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT3 }}>Treatment Table (Evidence-Graded)</h6>
                  <div className="table-responsive">
                    <table className="table table-sm table-hover small mb-0">
                      <thead>
                        <tr><th>Therapy</th><th>Level</th><th>Dose</th><th>Rationale</th></tr>
                      </thead>
                      <tbody>
                        {bd?.treatments?.map((t, i) => (
                          <tr key={i} style={{
                            background: t.level === 'ABSOLUTE CI' ? '#fce4ec' : t.level === 'HIGH RISK' ? '#fff8e1' : 'inherit'
                          }}>
                            <td className="fw-bold">{t.therapy}</td>
                            <td>
                              <span className={`badge ${
                                t.level === 'A' ? 'bg-success' :
                                t.level === 'B' ? 'bg-warning text-dark' :
                                t.level === 'ABSOLUTE CI' ? 'bg-danger' :
                                'bg-warning text-dark'
                              }`}>
                                {t.level}
                              </span>
                            </td>
                            <td className="text-muted">{t.dose}</td>
                            <td className="text-muted">{t.rationale}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="row">
            <div className="col-md-6 mb-3">
              <div className="alert alert-danger">
                <strong>&#x1f6ab; VPA — ABSOLUTE CI (Triple mechanism):</strong><br />
                (1) Inhibits NAGS → CPS1 off → NH₃ catastrophic.
                (2) Hepatotoxic in pre-existing liver disease (NICCD/CTLN2 cirrhosis).
                (3) Worsens mitochondrial dysfunction.
                NEVER use in any citrin deficiency phenotype.
              </div>
              <div className="alert alert-warning">
                <strong>&#x26a0; IV GLUCOSE / CARBOHYDRATE LOADS — HIGH RISK in CTLN2:</strong><br />
                Carbohydrates require intact MAS shuttle → BLOCKED in SLC25A13 → cytoplasmic
                NADH excess → lactate → paradoxically worsens NH₃. Prefer fat-based caloric
                support. CTLN2 patients instinctively avoid carbs for this reason.
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <InfoBox title="High-Fat Low-Carb Diet — Level A (Primary)" color={ACCENT3}>
                Fat bypasses the malate-aspartate shuttle — oxidised via beta-oxidation directly
                in mitochondria (no cytoplasmic NADH generated). Carbohydrate restriction reduces
                metabolic burden on impaired MAS shuttle. Patients self-select this diet (carb
                aversion = natural treatment adaptation).
              </InfoBox>
              <InfoBox title="Liver Transplant — CURATIVE (Level A for CTLN2)" color={ACCENT}>
                SLC25A13 predominantly expressed in liver. LT replaces deficient hepatocytes
                with normal donor liver → aspartate-glutamate transport restored → citrulline
                normalises → NH₃ normalises → MAS shuttle restored. Offer before encephalopathy.
              </InfoBox>
              <InfoBox title="East Asian Founder Variants" color={ACCENT5}>
                IVS16ins3kb (50-60% Japanese alleles); c.1638_1660dup23 (15% Japanese);
                p.Arg605Gln (Southern Chinese/Vietnamese). Targeted testing in East Asian ancestry
                before broad gene panel — cost-effective first step.
              </InfoBox>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && (
        <div>
          {def && Object.entries(def).map(([key, val], i) => (
            <div key={i} className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${[ACCENT, ACCENT2, ACCENT3, ACCENT4, ACCENT5, ACCENT6, ACCENT7, ACCENT8][i % 8]}` }}>
              <div className="card-body py-2">
                <div className="fw-bold small mb-1" style={{ color: [ACCENT, ACCENT2, ACCENT3, ACCENT4, ACCENT5, ACCENT6, ACCENT7, ACCENT8][i % 8] }}>
                  {key.replace(/_/g, ' ').toUpperCase()}
                </div>
                <div className="small text-muted" style={{ whiteSpace: 'pre-wrap' }}>{val}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Footer nav */}
      <div className="mt-4 d-flex gap-2 flex-wrap">
        <Link href="/slc25a15" className="btn btn-sm btn-outline-secondary">← SLC25A15 (HHH / Ornithine Transport)</Link>
        <Link href="/arg1" className="btn btn-sm btn-outline-secondary">ARG1 (Step 5)</Link>
        <Link href="/asl" className="btn btn-sm btn-outline-secondary">ASL (Step 4)</Link>
        <Link href="/ass1" className="btn btn-sm btn-outline-secondary">ASS1 (Step 3 enzyme)</Link>
        <Link href="/otc" className="btn btn-sm btn-outline-secondary">OTC (Step 2)</Link>
        <Link href="/cps1" className="btn btn-sm btn-outline-secondary">CPS1 (Step 1)</Link>
        <Link href="/agat" className="btn btn-sm btn-outline-secondary">AGAT →</Link>
      </div>
    </div>
  );
}

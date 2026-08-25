'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// SLC25A15 / HHH Syndrome color scheme — transport block / ornithine HIGH / homocitrullinuria UNIQUE
const ACCENT  = '#1a237e';   // deep navy — ornithine transport / mitochondrial
const ACCENT2 = '#b71c1c';   // deep crimson — hyperammonemia crisis / NH3 elevated
const ACCENT3 = '#880e4f';   // dark pink — homocitrullinuria PATHOGNOMONIC / unique
const ACCENT4 = '#e65100';   // deep orange — ornithine VERY HIGH
const ACCENT5 = '#4a148c';   // deep purple — HHH triad / triple pathognomonic
const ACCENT6 = '#006064';   // teal — coagulopathy UNIQUE
const ACCENT7 = '#c62828';   // red — VPA ABSOLUTE CI / danger
const ACCENT8 = '#1b5e20';   // deep green — citrulline bypass / treatment primary

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

export default function SLC25A15Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/slc25a15/overview`).then(r => r.json()),
      fetch(`${API}/api/slc25a15/breakdown`).then(r => r.json()),
      fetch(`${API}/api/slc25a15/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading SLC25A15 / HHH Syndrome dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="card shadow mb-3" style={{ borderTop: `5px solid ${ACCENT}` }}>
        <div className="card-body pb-2">
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            &#x26a1; SLC25A15 Epilepsy — HHH Syndrome (Ornithine Transporter 1 Deficiency)
          </h4>
          <div className="text-muted small">
            SLC25A15 / ORC1 / ORNT1 · Ornithine Transport BLOCKED · Ornithine{' '}
            <strong>VERY HIGH</strong> (PATHOGNOMONIC) · Homocitrullinuria (UNIQUE) ·
            Episodic Hyperammonemia · AR · 13q14.11 · OMIM #238970
          </div>
          <div className="mt-2 d-flex gap-2 flex-wrap">
            <span className="badge" style={{ background: ACCENT }}>Mitochondrial Ornithine Transport</span>
            <span className="badge" style={{ background: ACCENT4 }}>Ornithine ↑↑↑ PATHOGNOMONIC</span>
            <span className="badge" style={{ background: ACCENT3 }}>Homocitrullinuria UNIQUE</span>
            <span className="badge" style={{ background: ACCENT2 }}>NH3 Episodic HIGH</span>
            <span className="badge" style={{ background: ACCENT6 }}>Coagulopathy UNIQUE to HHH</span>
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

          {/* HHH Triad callout */}
          <div className="alert mb-3" style={{ background: '#e8eaf6', border: `2px solid ${ACCENT5}` }}>
            <strong style={{ color: ACCENT5 }}>&#x26a0; HHH TRIAD — ALL THREE PATHOGNOMONIC:</strong>{' '}
            {ov.hallmark_biomarker}
          </div>

          {/* Citrulline bypass alert */}
          <div className="alert mb-3" style={{ background: '#e8f5e9', border: `2px solid ${ACCENT8}` }}>
            <strong style={{ color: ACCENT8 }}>&#x2714; CITRULLINE BYPASS (Level A):</strong>{' '}
            {ov.citrulline_bypass} — bypasses the transport block; provides citrulline directly in cytoplasm for ASS1→ASL→ARG1.
          </div>

          {/* Ornithine supplementation CI */}
          <div className="alert alert-danger mb-3">
            <strong>&#x1f6ab; ORNITHINE SUPPLEMENTATION — ABSOLUTE CI (counterintuitive):</strong>{' '}
            {ov.ornithine_ci_note}
          </div>

          <div className="row">
            {/* Transport mechanism */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Transport Mechanism — SLC25A15 Block</h6>
                  {bd?.transport_mechanism && (
                    <div className="small">
                      <div className="mb-2 p-2 rounded" style={{ background: '#e8eaf6' }}>
                        <strong>Function:</strong> {bd.transport_mechanism.function}<br />
                        <strong>Reaction:</strong> {bd.transport_mechanism.reaction}
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#fce4ec' }}>
                        <strong style={{ color: ACCENT2 }}>Block:</strong> {bd.transport_mechanism.block}
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#fff8e1' }}>
                        <strong>Upstream intact:</strong> {bd.transport_mechanism.upstream_intact}
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#fce4ec' }}>
                        <strong style={{ color: ACCENT3 }}>Alternative path (UNIQUE):</strong>{' '}
                        {bd.transport_mechanism.alternative_path}
                      </div>
                      <div className="p-2 rounded" style={{ background: '#f3e5f5' }}>
                        <strong style={{ color: ACCENT5 }}>Unique consequence:</strong>{' '}
                        {bd.transport_mechanism.unique_consequence}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Phenotype distribution */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT5}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT5 }}>Phenotype Distribution (n={ov.n_patients})</h6>
                  {ov.phenotype_dist?.map((pc, i) => (
                    <div key={i} className="mb-2">
                      <PctBar
                        label={`${pc.class} (n=${pc.n})`}
                        pct={pc.pct}
                        color={[ACCENT, ACCENT2, ACCENT3][i % 3]}
                      />
                    </div>
                  ))}

                  <hr className="my-2" />
                  <h6 className="fw-bold small mt-2" style={{ color: ACCENT }}>Urea Cycle Position</h6>
                  <div className="small text-muted" style={{ fontFamily: 'monospace', fontSize: '0.7rem' }}>
                    <div>CPS1 (Step 1) → Carbamoyl-P</div>
                    <div style={{ color: ACCENT7 }}>⚠ SLC25A15 — Orn transport BLOCKED ↓</div>
                    <div style={{ color: '#999' }}>  OTC (Step 2) — ORN absent ⚠</div>
                    <div style={{ color: ACCENT3 }}>  ↳ Carbamoyl-P + Lys → Homocitrulline ↑↑ (UNIQUE)</div>
                    <div>ASS1 (Step 3) → Argininosuccinate</div>
                    <div>ASL (Step 4) → Arginine + Fumarate</div>
                    <div>ARG1 (Step 5) → Ornithine + Urea</div>
                    <div style={{ color: ACCENT7 }}>⚠ Ornithine → CANNOT re-enter mitochondria</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* vs OAT distinction */}
          <div className="row mb-3">
            <div className="col-12">
              <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT4}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT4 }}>SLC25A15 vs OAT — The Ornithine Twins (CRITICAL DISTINCTION)</h6>
                  <div className="small">{ov.key_distinction_oat}</div>
                  {bd?.differential_diagnosis?.vs_oat && (
                    <div className="row mt-2">
                      <div className="col-md-6">
                        <div className="p-2 rounded" style={{ background: '#e8eaf6' }}>
                          <strong style={{ color: ACCENT }}>SLC25A15 (HHH):</strong><br />
                          <span className="text-muted small">{bd.differential_diagnosis.vs_oat.hhh_features}</span>
                        </div>
                      </div>
                      <div className="col-md-6">
                        <div className="p-2 rounded" style={{ background: '#fff8e1' }}>
                          <strong style={{ color: ACCENT4 }}>OAT (Gyrate Atrophy):</strong><br />
                          <span className="text-muted small">{bd.differential_diagnosis.vs_oat.oat_features}</span>
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
                    <PctBar label={f.feature} pct={f.pct} color={[ACCENT2, ACCENT5, ACCENT6, ACCENT, ACCENT4, ACCENT8][i % 6]} />
                    <div className="small text-muted ps-1">{f.note}</div>
                  </div>
                ))}
              </div>
            </div>
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
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Biomarker Profile — SLC25A15</h6>
                  <div className="table-responsive">
                    <table className="table table-sm table-hover small mb-0">
                      <thead>
                        <tr><th>Biomarker</th><th>Normal</th><th>Status in SLC25A15</th><th>Direction</th></tr>
                      </thead>
                      <tbody>
                        {bd?.biomarkers && Object.values(bd.biomarkers).map((bm, i) => (
                          <tr key={i}>
                            <td className="fw-bold">{bm.label}</td>
                            <td className="text-muted">{bm.normal}</td>
                            <td style={{ color: bm.color === 'danger' ? '#b71c1c' : bm.color === 'warning' ? '#e65100' : bm.color === 'success' ? '#2e7d32' : '#1a237e' }}>
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
              <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT3}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT3 }}>Top Pathogenic Variants</h6>
                  {bd?.variants?.map((v, i) => (
                    <div key={i} className="mb-2 p-2 rounded" style={{ background: i <= 1 ? '#fce4ec' : i <= 3 ? '#f3e5f5' : '#f5f5f5' }}>
                      <div className="d-flex justify-content-between">
                        <span className="fw-bold small" style={{ color: ACCENT }}>{v.variant}</span>
                        <span className="badge" style={{ background: ACCENT3 }}>{v.freq}%</span>
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
          <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT5}` }}>
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT5 }}>Cohort Preview (first 10 of {ov.n_patients})</h6>
              <div className="table-responsive">
                <table className="table table-sm table-hover small mb-0">
                  <thead>
                    <tr>
                      <th>ID</th><th>Phenotype</th><th>Age Onset (mo)</th>
                      <th>Ornithine (µmol/L)</th><th>NH3 Peak</th><th>Homocit (urine)</th>
                      <th>Seizures</th><th>IDD</th><th>Spasticity</th><th>Variant</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd?.cohort_preview?.map((p, i) => (
                      <tr key={i}>
                        <td>{p.id}</td>
                        <td>
                          <span className="badge" style={{
                            background: p.phenotype === 'Classic Episodic' ? ACCENT : p.phenotype === 'Severe Neonatal' ? ACCENT7 : ACCENT8
                          }}>{p.phenotype}</span>
                        </td>
                        <td>{p.age_onset_months}</td>
                        <td className="fw-bold" style={{ color: ACCENT4 }}>{p.ornithine_plasma}</td>
                        <td style={{ color: ACCENT2 }}>{p.nh3_peak_umol_l}</td>
                        <td style={{ color: ACCENT3 }}>{p.homocitrulline_urine}</td>
                        <td>{p.seizures ? '✓' : '–'}</td>
                        <td>{p.idd ? '✓' : '–'}</td>
                        <td>{p.spastic_paraplegia ? '✓' : '–'}</td>
                        <td className="text-muted">{p.variant}</td>
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
                      <PctBar label={s.type} pct={s.pct} color={[ACCENT2, ACCENT3, ACCENT, ACCENT5, ACCENT7, ACCENT4][i % 6]} />
                      <div className="small text-muted ps-1">{s.note}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Treatments */}
            <div className="col-md-7 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT8}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT8 }}>Treatment Table (Evidence-Graded)</h6>
                  <div className="table-responsive">
                    <table className="table table-sm table-hover small mb-0">
                      <thead>
                        <tr><th>Therapy</th><th>Level</th><th>Dose</th><th>Rationale</th></tr>
                      </thead>
                      <tbody>
                        {bd?.treatments?.map((t, i) => (
                          <tr key={i} style={{ background: t.level === 'ABSOLUTE CI' ? '#fce4ec' : 'inherit' }}>
                            <td className="fw-bold">{t.therapy}</td>
                            <td>
                              <span className={`badge ${t.level === 'A' ? 'bg-success' : t.level === 'B' ? 'bg-warning text-dark' : 'bg-danger'}`}>
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
                <strong>&#x1f6ab; VPA — ABSOLUTE CI in ALL UCDs:</strong><br />
                Valproate inhibits NAGS → no NAG → CPS1 off → catastrophic hyperammonemia.
                In HHH syndrome, the already-impaired urea cycle collapses entirely. NEVER use.
              </div>
              <div className="alert alert-danger">
                <strong>&#x1f6ab; ORNITHINE SUPPLEMENTATION — ABSOLUTE CI in SLC25A15 (counterintuitive):</strong><br />
                Ornithine CANNOT cross the mitochondrial membrane (this IS the disease).
                Supplementing ornithine raises cytoplasmic ornithine further — worsening hyperornithinemia
                without helping the urea cycle.
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <InfoBox title="Citrulline Bypass — Level A Mechanism" color={ACCENT8}>
                Citrulline supplementation bypasses the SLC25A15 block: provides citrulline directly
                in cytoplasm → ASS1 (step 3) uses it → arginine produced → ornithine via ARG1 →
                partial urea cycle maintained WITHOUT ornithine needing to enter mitochondria.
              </InfoBox>
              <InfoBox title="Coagulopathy — UNIQUE to HHH (40%)" color={ACCENT6}>
                Elevated cytoplasmic ornithine (400–1500 µmol/L) inhibits thrombin formation and
                fibrinogen polymerisation. Monitor PT/INR/fibrinogen routinely.
                NO other UCD produces coagulopathy — unique pathophysiology of HHH.
              </InfoBox>
              <InfoBox title="French-Canadian Founder — p.Phe188del" color={ACCENT3}>
                ~50-60% of alleles in Quebec/French-Canadian patients carry p.Phe188del (c.562_564delTTC).
                HHH syndrome was first described in a French-Canadian family (Shih 1969).
                Targeted testing in French-Canadian ancestry; then pan-ethnic.
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
        <Link href="/arg1" className="btn btn-sm btn-outline-secondary">← ARG1 (Step 5 — Final)</Link>
        <Link href="/asl" className="btn btn-sm btn-outline-secondary">ASL (Step 4)</Link>
        <Link href="/ass1" className="btn btn-sm btn-outline-secondary">ASS1 (Step 3)</Link>
        <Link href="/otc" className="btn btn-sm btn-outline-secondary">OTC (Step 2)</Link>
        <Link href="/cps1" className="btn btn-sm btn-outline-secondary">CPS1 (Step 1)</Link>
        <Link href="/nags" className="btn btn-sm btn-outline-secondary">NAGS (Cofactor)</Link>
        <Link href="/oat" className="btn btn-sm btn-outline-secondary">OAT (Differential)</Link>
        <Link href="/agat" className="btn btn-sm btn-outline-secondary">AGAT →</Link>
      </div>
    </div>
  );
}

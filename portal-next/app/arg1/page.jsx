'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// ARG1 color scheme — arginine CRITICALLY HIGH / spastic paraplegia UNIQUE / final UCD step
const ACCENT  = '#4a148c';   // deep purple — arginine PATHOGNOMONIC / step 5 final
const ACCENT2 = '#b71c1c';   // deep crimson — spastic paraplegia UNIQUE / IDD severe
const ACCENT3 = '#880e4f';   // dark pink — guanidino neurotoxicity / spasticity
const ACCENT4 = '#e65100';   // deep orange — NH3 MILD (unique vs UCDs)
const ACCENT5 = '#1a237e';   // deep navy — IDD most severe / treatment
const ACCENT6 = '#4e342e';   // dark brown — arginine ABSOLUTELY CI / unique reversal
const ACCENT7 = '#c62828';   // red — VPA ABSOLUTE CI / danger
const ACCENT8 = '#2e7d32';   // deep green — low-arginine diet PRIMARY / treatment

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

export default function ARG1Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/arg1/overview`).then(r => r.json()),
      fetch(`${API}/api/arg1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/arg1/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading ARG1 dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="card shadow mb-3" style={{ borderTop: `5px solid ${ACCENT}` }}>
        <div className="card-body pb-2">
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            &#x1f9ec; ARG1 Epilepsy — Hyperargininemia (Arginase-1 Deficiency)
          </h4>
          <div className="text-muted small">
            Arginase-1 Deficiency · Urea Cycle Step 5 of 5 (FINAL) · Arginine → Ornithine + Urea ·
            Plasma Arginine <strong>VERY HIGH</strong> (PATHOGNOMONIC) · Spastic Paraplegia (UNIQUE) ·
            NH3 MILD (unique vs other UCDs) · AR · 6q23.2 · OMIM #207800
          </div>
          <div className="mt-2 d-flex gap-2 flex-wrap">
            <span className="badge" style={{ background: ACCENT }}>Step 5 / 5 FINAL — Cytoplasmic</span>
            <span className="badge" style={{ background: ACCENT2 }}>Plasma Arginine ↑↑↑ PATHOGNOMONIC</span>
            <span className="badge" style={{ background: ACCENT3 }}>Spastic Paraplegia UNIQUE</span>
            <span className="badge" style={{ background: ACCENT4 }}>NH3 MILD (NOT crisis-dominant)</span>
            <span className="badge" style={{ background: ACCENT6 }}>Arginine Supp ABSOLUTE CI (unique)</span>
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

          {/* Hallmark callout */}
          <div className="alert mb-3" style={{ background: '#fce4ec', border: `2px solid ${ACCENT2}` }}>
            <strong style={{ color: ACCENT2 }}>&#x26a0; PATHOGNOMONIC HALLMARK:</strong>{' '}
            {ov.hallmark_biomarker}
            <br />
            <strong style={{ color: ACCENT3 }}>&#x26a0; UNIQUE CLINICAL:</strong>{' '}
            {ov.hallmark_clinical}
          </div>

          {/* Arginine supplementation CI alert */}
          <div className="alert alert-danger mb-3">
            <strong>&#x1f6ab; ARGININE ABSOLUTELY CONTRAINDICATED IN ARG1:</strong>{' '}
            Arginine supplementation worsens ARG1 — it is the <em>toxic substrate</em>.
            In ALL other UCDs (OTC/CPS1/NAGS/ASS1/ASL), arginine is the <strong>PRIMARY therapy</strong>.
            In ARG1 it is <strong>ABSOLUTELY CONTRAINDICATED</strong> — a unique and critical reversal.
          </div>

          <div className="row">
            {/* Urea cycle context */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Urea Cycle — Step 5 of 5 (FINAL)</h6>
                  {bd?.urea_cycle_context && (
                    <div className="small">
                      <div className="mb-2 p-2 rounded" style={{ background: '#f3e5f5' }}>
                        <strong>Reaction:</strong> {bd.urea_cycle_context.reaction}<br />
                        <strong>Location:</strong> {bd.urea_cycle_context.location}<br />
                        <strong>Cofactor:</strong> {bd.urea_cycle_context.cofactor}
                      </div>
                      <div className="p-2 rounded mb-2" style={{ background: '#fce4ec' }}>
                        <strong>Upstream INTACT:</strong> {bd.urea_cycle_context.upstream_intact}
                      </div>
                      <div className="p-2 rounded mb-2" style={{ background: '#fff3e0' }}>
                        <strong>Downstream deficit:</strong> {bd.urea_cycle_context.downstream_deficit}
                      </div>
                      <div className="p-2 rounded" style={{ background: '#e8f5e9' }}>
                        <strong>Unique consequence:</strong> {bd.urea_cycle_context.unique_consequence}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Phenotype distribution */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT2}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT2 }}>Phenotypic Classes (n={ov.n_patients})</h6>
                  {ov.phenotype_dist?.map((p, i) => (
                    <div key={i} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{p.class}</span>
                        <span className="fw-bold">{p.n} ({p.pct}%)</span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div className="progress-bar" style={{ width: `${p.pct}%`, backgroundColor: [ACCENT2, ACCENT8, ACCENT4][i] }} />
                      </div>
                    </div>
                  ))}
                  <div className="mt-3 p-2 rounded small" style={{ background: '#e8eaf6' }}>
                    <strong>Key:</strong> Classic Spastic = progressive spastic diplegia + IDD + seizures (gradual onset)<br />
                    Mild Attenuated = NBS-detected; mild/no spasticity; better cognition<br />
                    Neonatal Acute = rare; severe NH3 crisis; null alleles
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Key distinctions vs other UCDs */}
          <div className="row">
            <div className="col-md-6 mb-3">
              <InfoBox title="KEY DISTINCTION vs ALL Proximal UCDs (CPS1/OTC/NAGS/ASS1/ASL)" color={ACCENT2}>
                {ov.key_distinction_all_ucds}
              </InfoBox>
              <InfoBox title="Liver Transplant Caveat (Level B only — unique)" color={ACCENT5}>
                {ov.liver_transplant_caveat}
              </InfoBox>
            </div>
            <div className="col-md-6 mb-3">
              <InfoBox title="VPA Contraindication (ALL UCDs)" color={ACCENT7}>
                {ov.vpa_contraindication}
              </InfoBox>
              {bd?.differential_diagnosis?.vs_proximal_ucds && (
                <div className="card shadow-sm" style={{ borderLeft: `4px solid ${ACCENT3}` }}>
                  <div className="card-body py-2">
                    <div className="fw-bold small mb-1" style={{ color: ACCENT3 }}>vs ALL Proximal UCDs — Summary</div>
                    <div className="small text-muted">
                      <strong>Arginine:</strong> {bd.differential_diagnosis.vs_proximal_ucds.key_diff}<br />
                      <strong>NH3:</strong> {bd.differential_diagnosis.vs_proximal_ucds.nh3}<br />
                      <strong>Clinical:</strong> {bd.differential_diagnosis.vs_proximal_ucds.clinical}<br />
                      <strong>Citrulline:</strong> {bd.differential_diagnosis.vs_proximal_ucds.citrulline}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Gene info */}
          <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT}` }}>
            <div className="card-body py-2">
              <div className="row small">
                {[
                  ['Gene', ov.gene],
                  ['OMIM Gene', `*${ov.omim_gene}`],
                  ['OMIM Disease', `#${ov.omim_disease}`],
                  ['Chromosome', ov.chromosome],
                  ['Inheritance', ov.inheritance],
                  ['Protein', ov.protein],
                  ['Prevalence', ov.prevalence],
                  ['Urea Cycle Step', ov.urea_cycle_step],
                ].map(([k, v], i) => (
                  <div key={i} className="col-md-6 mb-1">
                    <span className="fw-bold">{k}:</span> {v}
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
            {/* Biomarkers */}
            <div className="col-md-7 mb-3">
              <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Biomarker Profile</h6>
                  <div className="table-responsive">
                    <table className="table table-sm table-hover small mb-0">
                      <thead>
                        <tr>
                          <th>Biomarker</th>
                          <th>Normal</th>
                          <th>ARG1 Value</th>
                          <th>Direction</th>
                        </tr>
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
                    <div key={i} className="mb-2 p-2 rounded" style={{ background: i === 0 ? '#fce4ec' : '#f3e5f5' }}>
                      <div className="d-flex justify-content-between">
                        <span className="fw-bold small" style={{ color: ACCENT }}>{v.variant}</span>
                        <span className="badge" style={{ background: ACCENT2 }}>{v.freq}%</span>
                      </div>
                      <div className="small text-muted">{v.domain} · {v.phenotype}</div>
                      <div className="small text-muted fst-italic">{v.note}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Systemic features */}
          <div className="card shadow-sm mb-3" style={{ borderTop: `3px solid ${ACCENT2}` }}>
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT2 }}>Systemic Features (ARG1 — n={ov.n_patients})</h6>
              <div className="row">
                {bd?.systemic_features?.map((f, i) => (
                  <div key={i} className="col-md-6 mb-2">
                    <PctBar label={f.feature} pct={f.pct} color={[ACCENT2, ACCENT5, ACCENT3, ACCENT, ACCENT4, ACCENT8][i % 6]} />
                    <div className="small text-muted ps-1">{f.note}</div>
                  </div>
                ))}
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
                      <th>Arginine (µmol/L)</th><th>NH3 Peak</th><th>Citrulline</th>
                      <th>Spasticity</th><th>Seizures</th><th>IDD</th><th>Variant</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd?.cohort_preview?.map((p, i) => (
                      <tr key={i}>
                        <td>{p.id}</td>
                        <td><span className="badge" style={{ background: p.phenotype === 'Classic Spastic' ? ACCENT2 : p.phenotype === 'Neonatal Acute' ? ACCENT7 : ACCENT8 }}>{p.phenotype}</span></td>
                        <td>{p.age_onset_months}</td>
                        <td className="fw-bold" style={{ color: ACCENT2 }}>{p.arginine_plasma}</td>
                        <td style={{ color: ACCENT4 }}>{p.nh3_peak_umol_l}</td>
                        <td>{p.citrulline_umol_l}</td>
                        <td>{p.spastic_paraplegia ? '✓' : '–'}</td>
                        <td>{p.seizures ? '✓' : '–'}</td>
                        <td>{p.idd ? '✓' : '–'}</td>
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

          {/* Key treatment notes */}
          <div className="row">
            <div className="col-md-6 mb-3">
              <div className="alert alert-danger">
                <strong>&#x1f6ab; VPA — ABSOLUTE CI in ALL UCDs:</strong><br />
                Valproate inhibits NAGS → no NAG → CPS1 off → catastrophic hyperammonemia.
                Multiple deaths reported. NEVER use in any urea cycle disorder.
              </div>
              <div className="alert alert-danger">
                <strong>&#x1f6ab; ARGININE SUPPLEMENTATION — ABSOLUTE CI in ARG1 (UNIQUE):</strong><br />
                In ALL other UCDs (OTC, CPS1, NAGS, ASS1, ASL) — arginine is PRIMARY therapy.<br />
                In ARG1 — arginine IS THE TOXIC SUBSTRATE. Supplementation worsens accumulation,
                accelerates guanidino neurotoxicity, spasticity, and IDD.
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <InfoBox title="Spasticity Management (UNIQUE to ARG1)" color={ACCENT3}>
                Progressive spastic paraplegia is the hallmark — NOT seen in other UCDs.
                Baclofen (oral/intrathecal) + physiotherapy are Level B interventions unique to ARG1 management.
                Early low-arginine diet HALTS progression; established spasticity may not fully reverse.
              </InfoBox>
              <InfoBox title="Liver Transplant — Level B Only (unique caveat)" color={ACCENT5}>
                Corrects hepatic ARG1; reduces arginine. However, ARG1 is expressed extrahepnatically
                (erythrocytes, brain, kidney). Pre-existing neurological damage may NOT reverse.
                Compare: Level A CURATIVE in all proximal UCDs for NH3.
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
        <Link href="/asl" className="btn btn-sm btn-outline-secondary">← ASL (Step 4)</Link>
        <Link href="/otc" className="btn btn-sm btn-outline-secondary">OTC (Step 2)</Link>
        <Link href="/ass1" className="btn btn-sm btn-outline-secondary">ASS1 (Step 3)</Link>
        <Link href="/cps1" className="btn btn-sm btn-outline-secondary">CPS1 (Step 1)</Link>
        <Link href="/nags" className="btn btn-sm btn-outline-secondary">NAGS (Cofactor)</Link>
        <Link href="/oat" className="btn btn-sm btn-outline-secondary">OAT (Differential)</Link>
      </div>
    </div>
  );
}

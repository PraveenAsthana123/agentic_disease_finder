'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// IVD / Isovaleric Acidemia color scheme
const ACCENT  = '#1b5e20';   // deep green — IVD enzyme / leucine catabolism / glycine detox
const ACCENT2 = '#b71c1c';   // deep crimson — isovaleric acid / neurotoxicity / crisis
const ACCENT3 = '#e65100';   // deep orange — emergency protocol / fasting hazard / VPA CI
const ACCENT4 = '#0d47a1';   // deep blue — NBS C5 marker / biomarkers
const ACCENT5 = '#4a148c';   // deep purple — sweaty feet / odor / seizures
const ACCENT6 = '#2e7d32';   // green — glycine + carnitine treatment
const ACCENT7 = '#c62828';   // red — DRE / danger / ABSOLUTE CI
const ACCENT8 = '#37474f';   // dark slate — bone marrow suppression / systemic

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

export default function IVDPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/ivd/overview`).then(r => r.json()),
      fetch(`${API}/api/ivd/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ivd/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading IVD / Isovaleric Acidemia dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="card shadow mb-3" style={{ borderTop: `5px solid ${ACCENT}` }}>
        <div className="card-body pb-2">
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            &#x1f9ec; IVD Epilepsy — Isovaleric Acidemia (IVA)
          </h4>
          <div className="text-muted small">
            IVD / Isovaleryl-CoA Dehydrogenase · Leucine Catabolism Block ·{' '}
            <strong>"Sweaty Feet" Odor PATHOGNOMONIC</strong> · C5 NBS Marker ·
            Glycine + Carnitine CORNERSTONE · VPA ABSOLUTE CI · AR · 15q15.1 · OMIM #243500
          </div>
          <div className="mt-2 d-flex gap-2 flex-wrap">
            <span className="badge" style={{ background: ACCENT }}>Leucine Catabolism Block</span>
            <span className="badge" style={{ background: ACCENT2 }}>IVA ↑↑ Neurotoxic</span>
            <span className="badge" style={{ background: ACCENT4 }}>C5 NBS PRIMARY</span>
            <span className="badge" style={{ background: ACCENT5 }}>IVG PATHOGNOMONIC</span>
            <span className="badge" style={{ background: ACCENT6 }}>Glycine Level A — UNIQUE</span>
            <span className="badge" style={{ background: ACCENT7 }}>VPA ABSOLUTE CI</span>
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

          {/* Glycine unique treatment callout */}
          <div className="alert mb-3" style={{ background: '#e8f5e9', border: `2px solid ${ACCENT6}` }}>
            <strong style={{ color: ACCENT6 }}>&#x1f4a1; GLYCINE — CORNERSTONE UNIQUE TREATMENT:</strong>{' '}
            {ov.glycine_unique_note}
          </div>

          {/* VPA absolute CI warning */}
          <div className="alert mb-3" style={{ background: '#fce4ec', border: `2px solid ${ACCENT7}` }}>
            <strong style={{ color: ACCENT7 }}>&#x1f6ab; VPA ABSOLUTE CI (COMPETITIVE INHIBITOR OF IVD):</strong>{' '}
            {ov.vpa_absolute_ci_note}
          </div>

          <div className="row">
            {/* Enzyme mechanism */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Enzyme Mechanism — IVD Block</h6>
                  {bd?.enzyme_mechanism && (
                    <div className="small">
                      <div className="mb-2 p-2 rounded" style={{ background: '#e8f5e9' }}>
                        <strong>Function:</strong> {bd.enzyme_mechanism.function}<br />
                        <strong>Reaction:</strong> {bd.enzyme_mechanism.reaction}
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#fce4ec' }}>
                        <strong style={{ color: ACCENT2 }}>Block:</strong> {bd.enzyme_mechanism.block}
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#e8f5e9' }}>
                        <strong style={{ color: ACCENT6 }}>Glycine detox:</strong> {bd.enzyme_mechanism.detox_glycine}
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#e3f2fd' }}>
                        <strong style={{ color: ACCENT4 }}>Carnitine detox:</strong> {bd.enzyme_mechanism.detox_carnitine}
                      </div>
                      <div className="p-2 rounded" style={{ background: '#fff8e1' }}>
                        <strong style={{ color: ACCENT3 }}>Leucine pathway:</strong>{' '}
                        {bd.enzyme_mechanism.leucine_path}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Phenotype distribution */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT4}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT4 }}>Phenotype Distribution (n={ov.n_patients})</h6>
                  {ov.phenotype_dist?.map((pc, i) => (
                    <div key={i} className="mb-2">
                      <PctBar
                        label={`${pc.class} (n=${pc.n})`}
                        pct={pc.pct}
                        color={[ACCENT2, ACCENT5, ACCENT6][i % 3]}
                      />
                    </div>
                  ))}

                  <hr className="my-2" />
                  <h6 className="fw-bold small mt-2" style={{ color: ACCENT }}>Leucine Catabolism Pathway</h6>
                  <div className="small text-muted" style={{ fontFamily: 'monospace', fontSize: '0.7rem' }}>
                    <div>L-Leucine → BCAT (Step 1a) → KIC</div>
                    <div>  → BCKDH (Step 1b) → Isovaleryl-CoA</div>
                    <div style={{ color: ACCENT7 }}>  → [IVD BLOCKED] ⚠</div>
                    <div style={{ color: ACCENT2 }}>     ↳ Isovaleric acid ↑↑ ("sweaty feet")</div>
                    <div style={{ color: ACCENT5 }}>     ↳ + Glycine → IVG ↑↑ (PATHOGNOMONIC)</div>
                    <div style={{ color: ACCENT4 }}>     ↳ + Carnitine → C5 ↑ (NBS marker)</div>
                    <div style={{ color: ACCENT6 }}>  Normal: → 3-MC-CoA → HMG-CoA → AcCoA</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* NBS C5 differential */}
          <div className="row mb-3">
            <div className="col-12">
              <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT4}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT4 }}>Newborn Screening (NBS) — C5 Differential</h6>
                  <div className="small">{ov.nbs_note}</div>
                  <div className="row mt-2">
                    <div className="col-md-4">
                      <div className="p-2 rounded" style={{ background: '#e8f5e9' }}>
                        <strong style={{ color: ACCENT }}>IVD: C5 + IVG elevated</strong><br />
                        <span className="text-muted small">Isovalerylglycine PATHOGNOMONIC → confirms IVD over SBCAD. Gene panel: IVD.</span>
                      </div>
                    </div>
                    <div className="col-md-4">
                      <div className="p-2 rounded" style={{ background: '#fff8e1' }}>
                        <strong style={{ color: ACCENT3 }}>SBCAD: C5 + 2-MBG (not IVG)</strong><br />
                        <span className="text-muted small">2-methylbutyrylglycine in urine OA. Usually benign. Gene: ACADSB.</span>
                      </div>
                    </div>
                    <div className="col-md-4">
                      <div className="p-2 rounded" style={{ background: '#e3f2fd' }}>
                        <strong style={{ color: ACCENT4 }}>Pivalate: C5 alone</strong><br />
                        <span className="text-muted small">Iatrogenic (pivalate-conjugated antibiotics). No IVG or 2-MBG. Resolves on stopping drug.</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Systemic features */}
          <div className="card shadow-sm mb-3" style={{ borderTop: `3px solid ${ACCENT5}` }}>
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT5 }}>Systemic Features (n={ov.n_patients})</h6>
              <div className="row">
                {bd?.systemic_features?.map((f, i) => (
                  <div key={i} className="col-md-6 mb-2">
                    <PctBar label={f.feature} pct={f.pct} color={[ACCENT2, ACCENT5, ACCENT3, ACCENT4, ACCENT8, ACCENT6, ACCENT, ACCENT7][i % 8]} />
                    <div className="small text-muted ps-1">{f.note}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Hallmark alert */}
          <div className="alert mb-3" style={{ background: '#e8f5e9', border: `2px solid ${ACCENT}` }}>
            <strong style={{ color: ACCENT }}>&#x26a0; KEY BIOMARKERS:</strong>{' '}
            {ov.hallmark_biomarker}
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
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Biomarker Profile — IVD / Isovaleric Acidemia</h6>
                  <div className="table-responsive">
                    <table className="table table-sm table-hover small mb-0">
                      <thead>
                        <tr><th>Biomarker</th><th>Normal</th><th>Status in IVA</th><th>Direction</th></tr>
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
                  <h6 className="fw-bold" style={{ color: ACCENT5 }}>Top Pathogenic Variants</h6>
                  {bd?.variants?.map((v, i) => (
                    <div key={i} className="mb-2 p-2 rounded" style={{ background: i === 0 ? '#e8f5e9' : i <= 2 ? '#fce4ec' : '#f5f5f5' }}>
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
          <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT4}` }}>
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT4 }}>Cohort Preview (first 10 of {ov.n_patients})</h6>
              <div className="table-responsive">
                <table className="table table-sm table-hover small mb-0">
                  <thead>
                    <tr>
                      <th>ID</th><th>Phenotype</th><th>Age Onset (mo)</th>
                      <th>C5 (µmol/L)</th><th>IVG (urine)</th><th>Carnitine</th>
                      <th>NH3</th><th>Sweaty Feet</th><th>Crises</th><th>Seizures</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd?.cohort_preview?.map((p, i) => (
                      <tr key={i}>
                        <td>{p.id}</td>
                        <td>
                          <span className="badge" style={{
                            background: p.phenotype.includes('Classic') ? ACCENT2 : p.phenotype.includes('NBS') ? ACCENT6 : ACCENT5,
                            fontSize: '0.6rem'
                          }}>{p.phenotype.split(' ')[0]}</span>
                        </td>
                        <td>{p.age_onset_months}</td>
                        <td className="fw-bold" style={{ color: ACCENT2 }}>{p.c5_umol_l}</td>
                        <td style={{ color: ACCENT5 }}>{p.ivg_urine_mmol_mol_cr}</td>
                        <td style={{ color: p.free_carnitine_umol_l < 20 ? ACCENT7 : ACCENT6 }}>{p.free_carnitine_umol_l}</td>
                        <td style={{ color: p.nh3_umol_l > 80 ? ACCENT3 : '#2e7d32' }}>{p.nh3_umol_l}</td>
                        <td>{p.sweaty_feet_odor ? <span style={{ color: ACCENT5 }}>✓ ODOR</span> : '–'}</td>
                        <td>{p.crisis_count}</td>
                        <td>{p.seizures ? '✓' : '–'}</td>
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
                  <h6 className="fw-bold" style={{ color: ACCENT2 }}>Seizures / Neurological (n={ov.n_patients})</h6>
                  {bd?.seizure_types?.map((s, i) => (
                    <div key={i} className="mb-2">
                      <PctBar label={s.type} pct={s.pct} color={[ACCENT2, ACCENT5, ACCENT, ACCENT3, ACCENT7, ACCENT8][i % 6]} />
                      <div className="small text-muted ps-1">{s.note}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Treatments */}
            <div className="col-md-7 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT6}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT6 }}>Treatment Table (Evidence-Graded)</h6>
                  <div className="table-responsive">
                    <table className="table table-sm table-hover small mb-0">
                      <thead>
                        <tr><th>Therapy</th><th>Level</th><th>Dose</th><th>Rationale</th></tr>
                      </thead>
                      <tbody>
                        {bd?.treatments?.map((t, i) => (
                          <tr key={i} style={{
                            background: t.level === 'ABSOLUTE CI' ? '#fce4ec' : t.level === 'EXTREME HAZARD' ? '#fff3e0' : 'inherit'
                          }}>
                            <td className="fw-bold">{t.therapy}</td>
                            <td>
                              <span className={`badge ${
                                t.level === 'A' ? 'bg-success' :
                                t.level === 'B' ? 'bg-warning text-dark' :
                                t.level === 'C' ? 'bg-secondary' :
                                t.level === 'ABSOLUTE CI' ? 'bg-danger' :
                                t.level === 'EXTREME HAZARD' ? 'bg-danger' :
                                'bg-secondary'
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
              <div className="alert mb-2" style={{ background: '#fce4ec', border: `2px solid ${ACCENT7}` }}>
                <strong style={{ color: ACCENT7 }}>&#x1f6ab; VPA — ABSOLUTE CI (COMPETITIVE IVD INHIBITOR):</strong><br />
                Valproyl-CoA DIRECTLY competes with isovaleryl-CoA at the IVD active site →
                WORSENS the primary enzyme defect. Plus carnitine depletion → abolishes C5 detox.
                Plus hepatotoxicity. ALWAYS use LEV instead.
              </div>
              <div className="alert alert-warning mb-2">
                <strong>&#x26a0; FASTING / HIGH-PROTEIN — EXTREME HAZARD:</strong><br />
                Fasting triggers endogenous leucine release → isovaleryl-CoA surge → acute crisis.
                Emergency glucopolymer/glucose drinks at home. Written action plan for ALL families.
                Max fasting: 4–6 hours (infant), 8–10 hours (older child/adult).
              </div>
              <div className="alert mb-2" style={{ background: '#e8f5e9', border: `2px solid ${ACCENT6}` }}>
                <strong style={{ color: ACCENT6 }}>&#x1f4a1; GLYCINE SUPPLEMENT — DIAGNOSTIC + THERAPEUTIC:</strong><br />
                IVG on urine organic acids = both DIAGNOSIS confirmation AND treatment efficacy marker.
                Target IVG &gt; 100 mmol/mol Cr (confirms conjugation is occurring).
                Increase to 400 mg/kg/day in acute crisis.
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <InfoBox title="Glycine — Level A (Primary, UNIQUE to IVD)" color={ACCENT6}>
                150–300 mg/kg/day in 3 divided doses. Drives isovaleryl-CoA → IVG conjugation →
                renal excretion. Replenishes glycine depleted by ongoing IVG formation.
                Unique among organic acidemias — no other OA uses glycine as primary detox therapy.
                Start at diagnosis; increase during crisis (400 mg/kg/day).
              </InfoBox>
              <InfoBox title="Leucine Restriction — Level A (Primary)" color={ACCENT}>
                Natural protein 0.8–1.5 g/kg/day; leucine &lt; 100–150 mg/kg/day via IVA-free formula.
                Leucine is the sole substrate for IVD pathway. Reduce substrate flux → less IVA.
                Monitor plasma leucine (target 100–200 µmol/L). Avoid over-restriction → growth failure.
              </InfoBox>
              <InfoBox title="L-Carnitine — Level A (Secondary Detox)" color={ACCENT4}>
                100–200 mg/kg/day oral; IV during crises. Secondary carnitine deficiency from C5 conjugation.
                Carnitine supplementation replenishes pool + drives C5 excretion (synergistic with glycine).
                Protects cardiac and skeletal muscle during decompensation.
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
        <Link href="/gcdh" className="btn btn-sm btn-outline-secondary">← GCDH (Glutaric Aciduria Type 1)</Link>
        <Link href="/gamt" className="btn btn-sm btn-outline-secondary">GAMT (Creatine synthesis)</Link>
        <Link href="/pcca" className="btn btn-sm btn-outline-secondary">PCCA (Propionic Acidemia A)</Link>
        <Link href="/pccb" className="btn btn-sm btn-outline-secondary">PCCB (Propionic Acidemia B)</Link>
        <Link href="/mmut" className="btn btn-sm btn-outline-secondary">MMUT (MMA mut-type)</Link>
      </div>
    </div>
  );
}

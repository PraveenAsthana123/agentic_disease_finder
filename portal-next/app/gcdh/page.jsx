'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// GCDH / GA1 color scheme — striatal excitotoxicity / organic acidemia / emergency protocol
const ACCENT  = '#1a237e';   // deep indigo — GCDH enzyme / lysine catabolism / core
const ACCENT2 = '#b71c1c';   // deep crimson — 3-HGA neurotoxin / striatal injury
const ACCENT3 = '#e65100';   // deep orange — emergency protocol / crisis
const ACCENT4 = '#1565c0';   // blue — macrocephaly / MRI findings
const ACCENT5 = '#6a1b9a';   // deep purple — dystonia / striatal damage
const ACCENT6 = '#2e7d32';   // green — NBS / treatment / carnitine
const ACCENT7 = '#c62828';   // red — DRE / danger
const ACCENT8 = '#880e4f';   // dark pink — SDH / child abuse mimicry

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

export default function GCDHPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/gcdh/overview`).then(r => r.json()),
      fetch(`${API}/api/gcdh/breakdown`).then(r => r.json()),
      fetch(`${API}/api/gcdh/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading GCDH / Glutaric Aciduria Type 1 dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="card shadow mb-3" style={{ borderTop: `5px solid ${ACCENT}` }}>
        <div className="card-body pb-2">
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            &#x1f9ec; GCDH Epilepsy — Glutaric Aciduria Type 1 (GA1)
          </h4>
          <div className="text-muted small">
            GCDH / Glutaryl-CoA Dehydrogenase · 3-HGA Neurotoxin (NMDA Agonist) ·{' '}
            <strong>Macrocephaly + Striatal Crisis</strong> · Febrile Trigger ·
            Emergency Protocol CRITICAL · AR · 19p13.2 · OMIM #231670
          </div>
          <div className="mt-2 d-flex gap-2 flex-wrap">
            <span className="badge" style={{ background: ACCENT }}>Lysine/Trp Catabolism Block</span>
            <span className="badge" style={{ background: ACCENT2 }}>3-HGA Neurotoxin ↑↑</span>
            <span className="badge" style={{ background: ACCENT4 }}>Macrocephaly 90%</span>
            <span className="badge" style={{ background: ACCENT3 }}>Crisis = Striatal Injury</span>
            <span className="badge" style={{ background: ACCENT6 }}>Emergency Protocol CRITICAL</span>
            <span className="badge" style={{ background: ACCENT8 }}>SDH Mimics Child Abuse</span>
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

          {/* Emergency protocol callout */}
          <div className="alert mb-3" style={{ background: '#fff3e0', border: `2px solid ${ACCENT3}` }}>
            <strong style={{ color: ACCENT3 }}>&#x1f6a8; EMERGENCY PROTOCOL (TIME-CRITICAL):</strong>{' '}
            {ov.crisis_protocol_note}
          </div>

          {/* Low excretor warning */}
          <div className="alert alert-warning mb-3">
            <strong>&#x26a0; LOW EXCRETOR VARIANT (~15%):</strong>{' '}
            {ov.low_excretor_warning}
          </div>

          {/* SDH / NAI warning */}
          <div className="alert mb-3" style={{ background: '#fce4ec', border: `2px solid ${ACCENT8}` }}>
            <strong style={{ color: ACCENT8 }}>&#x1f6ab; SUBDURAL HAEMORRHAGE / CHILD ABUSE MIMICRY:</strong>{' '}
            {ov.sdh_note}
          </div>

          <div className="row">
            {/* Enzyme mechanism */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Enzyme Mechanism — GCDH Block</h6>
                  {bd?.enzyme_mechanism && (
                    <div className="small">
                      <div className="mb-2 p-2 rounded" style={{ background: '#e8eaf6' }}>
                        <strong>Function:</strong> {bd.enzyme_mechanism.function}<br />
                        <strong>Reaction:</strong> {bd.enzyme_mechanism.reaction}
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#fce4ec' }}>
                        <strong style={{ color: ACCENT2 }}>Block:</strong> {bd.enzyme_mechanism.block}
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#fff3e0' }}>
                        <strong style={{ color: ACCENT3 }}>Neurotoxin:</strong> {bd.enzyme_mechanism.neurotoxin}
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#e0e7ff' }}>
                        <strong style={{ color: ACCENT5 }}>Striatal vulnerability:</strong>{' '}
                        {bd.enzyme_mechanism.striatum_vuln}
                      </div>
                      <div className="p-2 rounded" style={{ background: '#e8f5e9' }}>
                        <strong style={{ color: ACCENT6 }}>Critical period:</strong>{' '}
                        {bd.enzyme_mechanism.critical_period}
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
                        color={[ACCENT2, ACCENT6, ACCENT5][i % 3]}
                      />
                    </div>
                  ))}

                  <hr className="my-2" />
                  <h6 className="fw-bold small mt-2" style={{ color: ACCENT }}>Lysine Catabolism Pathway</h6>
                  <div className="small text-muted" style={{ fontFamily: 'monospace', fontSize: '0.7rem' }}>
                    <div>L-Lysine → saccharopine pathway</div>
                    <div>  → α-aminoadipic semialdehyde</div>
                    <div>  → 2-oxoadipic acid</div>
                    <div style={{ color: ACCENT7 }}>  → Glutaryl-CoA → [GCDH BLOCKED] ⚠</div>
                    <div style={{ color: ACCENT2 }}>     ↳ Glutaric acid ↑↑↑ (VERY HIGH)</div>
                    <div style={{ color: ACCENT2 }}>     ↳ 3-HGA ↑↑ (NEUROTOXIN → NMDA)</div>
                    <div style={{ color: ACCENT6 }}>  Normal: → Crotonyl-CoA → Acetyl-CoA → TCA</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* NBS note */}
          <div className="row mb-3">
            <div className="col-12">
              <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT6}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT6 }}>Newborn Screening (NBS) — GA1</h6>
                  <div className="small">{ov.nbs_note}</div>
                  <div className="row mt-2">
                    <div className="col-md-6">
                      <div className="p-2 rounded" style={{ background: '#e8f5e9' }}>
                        <strong style={{ color: ACCENT6 }}>NBS marker: C5DC (glutarylcarnitine)</strong><br />
                        <span className="text-muted small">Detected on tandem mass spectrometry (TMS/MS-MS); expanded NBS programs. NBS outcomes excellent — pre-crisis treatment possible.</span>
                      </div>
                    </div>
                    <div className="col-md-6">
                      <div className="p-2 rounded" style={{ background: '#fff8e1' }}>
                        <strong style={{ color: ACCENT3 }}>Low-excretor: C5DC may be borderline/normal</strong><br />
                        <span className="text-muted small">p.Arg402Trp (Amish): low excretor → NBS may MISS. Macrocephaly + clinical suspicion → urine OA + gene panel regardless of NBS result.</span>
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
                    <PctBar label={f.feature} pct={f.pct} color={[ACCENT4, ACCENT5, ACCENT3, ACCENT2, ACCENT8, ACCENT6, ACCENT, ACCENT7][i % 8]} />
                    <div className="small text-muted ps-1">{f.note}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Hallmark alert */}
          <div className="alert mb-3" style={{ background: '#e8eaf6', border: `2px solid ${ACCENT}` }}>
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
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Biomarker Profile — GCDH / GA1</h6>
                  <div className="table-responsive">
                    <table className="table table-sm table-hover small mb-0">
                      <thead>
                        <tr><th>Biomarker</th><th>Normal</th><th>Status in GA1</th><th>Direction</th></tr>
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
                    <div key={i} className="mb-2 p-2 rounded" style={{ background: i === 0 ? '#fff3e0' : i <= 2 ? '#fce4ec' : '#f5f5f5' }}>
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
                      <th>GA Urine</th><th>3-HGA</th><th>C5DC</th>
                      <th>Carnitine</th><th>Macroceph.</th><th>Crises</th><th>Dystonia</th><th>Seizures</th>
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
                        <td className="fw-bold" style={{ color: ACCENT2 }}>{p.ga_urine_mmol_mol_cr}</td>
                        <td style={{ color: ACCENT7 }}>{p.hga_urine_mmol_mol_cr}</td>
                        <td style={{ color: ACCENT3 }}>{p.c5dc_umol_l}</td>
                        <td style={{ color: p.free_carnitine < 20 ? ACCENT7 : ACCENT6 }}>{p.free_carnitine}</td>
                        <td>{p.macrocephaly ? <span className="text-primary fw-bold">✓ MACRO</span> : '–'}</td>
                        <td>{p.crisis_count}</td>
                        <td>{p.dystonia ? <span style={{ color: ACCENT5 }}>✓ DYS</span> : '–'}</td>
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
                  <h6 className="fw-bold" style={{ color: ACCENT2 }}>Seizures / Movement Disorder (n={ov.n_patients})</h6>
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
                            background: t.level === 'MODERATE RISK' ? '#fff8e1' : t.level === 'AVOID' ? '#fce4ec' : 'inherit'
                          }}>
                            <td className="fw-bold">{t.therapy}</td>
                            <td>
                              <span className={`badge ${
                                t.level === 'A' ? 'bg-success' :
                                t.level === 'B' ? 'bg-warning text-dark' :
                                t.level === 'MODERATE RISK' ? 'bg-warning text-dark' :
                                t.level === 'AVOID' ? 'bg-danger' :
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
              <div className="alert mb-2" style={{ background: '#fff3e0', border: `2px solid ${ACCENT3}` }}>
                <strong style={{ color: ACCENT3 }}>&#x1f6a8; EMERGENCY PROTOCOL — MANDATORY for ALL fever:</strong><br />
                High-dose glucose (anti-catabolic) + IV/oral L-carnitine + protein restriction 24–48h.
                Start within HOURS of fever &gt;38°C. Written emergency plan for every family.
                Emergency letter to A&amp;E departments. Delay → permanent striatal damage.
              </div>
              <div className="alert alert-warning mb-2">
                <strong>&#x26a0; VPA — MODERATE RISK (not absolute CI):</strong><br />
                Worsens already-depleted carnitine (GA1 has secondary deficiency).
                Hepatotoxic potential. Prefer LEV. If VPA essential, supplement carnitine + monitor liver.
              </div>
              <div className="alert mb-2" style={{ background: '#fce4ec', border: `2px solid ${ACCENT8}` }}>
                <strong style={{ color: ACCENT8 }}>&#x1f6ab; DYSTONIC CRISIS ≠ SEIZURE:</strong><br />
                EEG during episodes: acute crisis movements often NON-epileptic (striatal origin).
                Treat with benzodiazepines + emergency metabolic protocol.
                Do NOT give AEDs alone for dystonic crisis — underlying metabolic emergency continues.
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <InfoBox title="Lysine Restriction — Level A (Primary)" color={ACCENT}>
                Lys is the principal substrate for GCDH pathway. Natural protein 0.8–1.5 g/kg/day
                + Lys-free amino acid supplement. Reduces 3-HGA production. Start at NBS diagnosis
                — before first crisis. Long-term Lys restriction reduces cumulative neurotoxin exposure.
              </InfoBox>
              <InfoBox title="L-Carnitine — Level A (Primary)" color={ACCENT6}>
                Secondary carnitine depletion in GA1 (glutaryl-CoA + carnitine → C5DC → carnitine lost).
                100–200 mg/kg/day oral; IV in crises. Replenishes free carnitine + enhances
                glutaric acid excretion as C5DC (detoxification). Protects cardiac muscle.
              </InfoBox>
              <InfoBox title="Riboflavin — Level B (FAD-Responsive Subset)" color={ACCENT5}>
                FAD-binding domain variants (e.g. p.Arg402Trp) may respond to pharmacological
                riboflavin (100–300 mg/day). Assess by &gt;50% urinary GA reduction at 3 months.
                Not all respond — still add Lys restriction regardless.
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
        <Link href="/slc25a13" className="btn btn-sm btn-outline-secondary">← SLC25A13 (Citrin/CTLN2)</Link>
        <Link href="/gamt" className="btn btn-sm btn-outline-secondary">GAMT (Creatine)</Link>
        <Link href="/agat" className="btn btn-sm btn-outline-secondary">AGAT (Creatine)</Link>
        <Link href="/slc6a8" className="btn btn-sm btn-outline-secondary">SLC6A8 (Creatine Transport)</Link>
        <Link href="/abat" className="btn btn-sm btn-outline-secondary">ABAT (GABA catabolism)</Link>
        <Link href="/aldh5a1" className="btn btn-sm btn-outline-secondary">ALDH5A1 (SSADH/GABA)</Link>
      </div>
    </div>
  );
}

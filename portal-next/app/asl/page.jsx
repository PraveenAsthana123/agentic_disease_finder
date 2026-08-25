'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// ASL color scheme — argininosuccinate CRITICALLY HIGH / trichorrhexis UNIQUE / NO deficiency
const ACCENT  = '#7b1fa2';   // deep purple — argininosuccinate PATHOGNOMONIC / step 4 cytoplasmic
const ACCENT2 = '#b71c1c';   // deep crimson — ammonia ELEVATED / seizures
const ACCENT3 = '#e65100';   // deep orange — trichorrhexis nodosa UNIQUE / hair hallmark
const ACCENT4 = '#880e4f';   // dark pink — hypertension UNIQUE / NO deficiency
const ACCENT5 = '#1a237e';   // deep navy — liver disease / treatment
const ACCENT6 = '#2e7d32';   // deep green — arginine PRIMARY therapy / conditionally essential
const ACCENT7 = '#c62828';   // red — VPA ABSOLUTE CI / danger
const ACCENT8 = '#006064';   // teal — key negatives / differentials

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

export default function ASLPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/asl/overview`).then(r => r.json()),
      fetch(`${API}/api/asl/breakdown`).then(r => r.json()),
      fetch(`${API}/api/asl/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading ASL dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="card shadow mb-3" style={{ borderTop: `5px solid ${ACCENT}` }}>
        <div className="card-body pb-2">
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            &#x1f9ec; ASL Epilepsy — Argininosuccinic Aciduria
          </h4>
          <div className="text-muted small">
            Argininosuccinate Lyase Deficiency · Urea Cycle Step 4 of 5 · Argininosuccinate → Arginine + Fumarate ·
            Argininosuccinate <strong>VERY HIGH</strong> (PATHOGNOMONIC) · Trichorrhexis Nodosa (UNIQUE) ·
            Hypertension (UNIQUE) · AR · 7q11.21 · OMIM #207900
          </div>
          <div className="mt-2 d-flex gap-2 flex-wrap">
            <span className="badge" style={{ background: ACCENT }}>Step 4 / 5 Cytoplasmic</span>
            <span className="badge" style={{ background: ACCENT2 }}>Argininosuccinate ↑↑↑ PATHOGNOMONIC</span>
            <span className="badge" style={{ background: ACCENT3 }}>Trichorrhexis Nodosa UNIQUE</span>
            <span className="badge" style={{ background: ACCENT4 }}>Hypertension UNIQUE (NO deficit)</span>
            <span className="badge" style={{ background: ACCENT6 }}>Arginine PRIMARY Therapy</span>
            <span className="badge bg-danger">VPA ABSOLUTE CI</span>
          </div>
        </div>
      </div>

      {/* Urea Cycle Position */}
      <div className="card shadow mb-3" style={{ borderLeft: `4px solid ${ACCENT}` }}>
        <div className="card-body py-2">
          <div className="fw-bold small mb-1" style={{ color: ACCENT }}>Urea Cycle — Step 4 Block (Cytoplasmic)</div>
          <div className="small font-monospace text-muted">
            NAGS→NAG · CPS1[Step1] · OTC[Step2] · ASS1[Step3] · <strong style={{ color: ACCENT }}>ASL[Step4 ✗ BLOCKED]</strong> · ARG1[Step5]<br/>
            Argininosuccinate (Step3→Step4 substrate) <strong style={{ color: ACCENT2 }}>ACCUMULATES MASSIVELY</strong> — pathognomonic in ALL body fluids<br/>
            Arginine NOT produced → conditionally essential · Fumarate NOT produced → TCA link broken<br/>
            NO synthesis impaired (ASL also in NO cycle) → <span style={{ color: ACCENT4 }}>systemic hypertension UNIQUE</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* TAB 0 — Overview */}
      {tab === 0 && (
        <>
          <div className="row g-2 mb-3">
            {Object.entries(kpi).map(([k, v]) => (
              <KPI key={k} label={v.label} value={v.value} color={v.color} />
            ))}
          </div>

          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Phenotypic Classes</h6>
                  {(ov?.phenotype_dist || []).map((c, i) => (
                    <PctBar key={i} label={c.class} pct={c.pct}
                      color={i === 0 ? ACCENT2 : i === 1 ? ACCENT : ACCENT6} />
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Unique Systemic Features (not seen in OTC/CPS1/NAGS/ASS1)</h6>
                  <PctBar label="Trichorrhexis nodosa (brittle hair)" pct={60} color={ACCENT3} />
                  <PctBar label="Systemic hypertension (NO deficit)" pct={45} color={ACCENT4} />
                  <PctBar label="Liver disease / hepatomegaly" pct={50} color={ACCENT5} />
                  <PctBar label="IDD (even with NH3 control)" pct={57} color={ACCENT} />
                  <PctBar label="Periventricular leukoencephalopathy" pct={35} color={ACCENT2} />
                </div>
              </div>
            </div>
          </div>

          <div className="row g-3">
            <div className="col-md-4">
              <InfoBox title="HALLMARK — Argininosuccinate VERY HIGH" color={ACCENT}>
                Plasma AND urine argininosuccinate VERY HIGH (>100 µmol/L; undetectable in health and ALL other UCDs).
                THE pathognomonic biomarker — present in plasma, urine, CSF. Massive accumulation at step 4 block.
              </InfoBox>
            </div>
            <div className="col-md-4">
              <InfoBox title="KEY DISTINCTION vs ASS1" color={ACCENT4}>
                ASL: argininosuccinate <strong>VERY HIGH</strong> (substrate accumulates at step 4 block).
                ASS1: argininosuccinate <strong>ABSENT</strong> (cannot be MADE at step 3 block).
                Single metabolite decisively separates these two cytoplasmic UCDs.
              </InfoBox>
            </div>
            <div className="col-md-4">
              <InfoBox title="Liver Transplant Caveat — UNIQUE to ASL" color={ACCENT5}>
                Liver Tx curative for hyperammonemia + normalises argininosuccinate/citrulline.
                BUT trichorrhexis, hypertension, neurocognition may NOT fully resolve (ASL functions in
                endothelium, renal, CNS beyond liver). UNIQUE among UCDs.
              </InfoBox>
            </div>
          </div>
        </>
      )}

      {/* TAB 1 — Patients & Biomarkers */}
      {tab === 1 && (
        <>
          <div className="row g-3 mb-3">
            <div className="col-lg-7">
              <div className="card shadow-sm">
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Biomarker Profile — ASL Deficiency</h6>
                  <div className="table-responsive">
                    <table className="table table-sm table-hover small">
                      <thead><tr>
                        <th>Biomarker</th><th>Normal</th><th>ASL Disease Value</th><th>Direction</th>
                      </tr></thead>
                      <tbody>
                        {Object.values(bd?.biomarkers || {}).map((b, i) => (
                          <tr key={i}>
                            <td className="fw-bold">{b.label}</td>
                            <td className="text-muted">{b.normal}</td>
                            <td>
                              <span className={`badge bg-${b.color}`}>{b.status}</span>
                              <div className="text-muted" style={{ fontSize: '0.72em' }}>{b.disease}</div>
                            </td>
                            <td className="fw-bold" style={{ color: b.color === 'danger' ? ACCENT2 : b.color === 'warning' ? ACCENT3 : b.color === 'success' ? '#2e7d32' : ACCENT }}>
                              {b.direction}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

            <div className="col-lg-5">
              <div className="card shadow-sm mb-3">
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Top Pathogenic Variants</h6>
                  <div className="table-responsive">
                    <table className="table table-sm small">
                      <thead><tr><th>Variant</th><th>Domain</th><th>Freq%</th><th>Phenotype</th></tr></thead>
                      <tbody>
                        {(bd?.variants || []).map((v, i) => (
                          <tr key={i}>
                            <td><code>{v.variant}</code></td>
                            <td className="text-muted">{v.domain}</td>
                            <td><span className="badge" style={{ background: ACCENT }}>{v.freq}%</span></td>
                            <td className="text-muted">{v.phenotype}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              <div className="card shadow-sm">
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Sample Cohort (first 10 / 40 patients)</h6>
                  <div className="table-responsive" style={{ maxHeight: 260, overflowY: 'auto' }}>
                    <table className="table table-sm small">
                      <thead><tr>
                        <th>ID</th><th>Phenotype</th><th>AS (µM)</th><th>NH3 (µM)</th><th>Citr (µM)</th><th>Hair</th>
                      </tr></thead>
                      <tbody>
                        {(bd?.cohort_preview || []).map((p, i) => (
                          <tr key={i}>
                            <td><code>{p.id}</code></td>
                            <td><span className="badge" style={{ background: p.phenotype === 'Classic Neonatal' ? ACCENT2 : p.phenotype === 'Late-Onset/Episodic' ? ACCENT : ACCENT6, fontSize: '0.7em' }}>{p.phenotype.split('/')[0]}</span></td>
                            <td className="fw-bold" style={{ color: ACCENT }}>{p.argininosuccinate_plasma}</td>
                            <td style={{ color: ACCENT2 }}>{p.nh3_peak_umol_l}</td>
                            <td>{p.citrulline_umol_l}</td>
                            <td>{p.trichorrhexis_nodosa ? <span style={{ color: ACCENT3 }}>✓</span> : <span className="text-muted">—</span>}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Differential diagnosis */}
          <div className="card shadow-sm">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT8 }}>Differential Diagnosis — ASL vs Other UCDs</h6>
              <div className="row g-2">
                {Object.entries(bd?.differential_diagnosis || {}).map(([k, v]) => (
                  <div key={k} className="col-md-4">
                    <div className="card" style={{ borderLeft: `3px solid ${ACCENT8}` }}>
                      <div className="card-body py-2">
                        <div className="fw-bold small" style={{ color: ACCENT8 }}>{k.replace('_', ' vs ').toUpperCase()}</div>
                        <div className="small text-muted">{v.key_diff}</div>
                        {v.citrulline && <div className="small text-muted mt-1"><em>Citrulline:</em> {v.citrulline}</div>}
                        {v.ncg_trial && <div className="small text-muted mt-1"><em>NCG trial:</em> {v.ncg_trial}</div>}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      {/* TAB 2 — Seizures & Treatments */}
      {tab === 2 && (
        <div className="row g-3">
          <div className="col-md-5">
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT2 }}>Seizure Types in ASL</h6>
                {(bd?.seizure_types || []).map((s, i) => (
                  <div key={i} className="mb-2">
                    <PctBar label={s.type} pct={s.pct} color={i === 0 ? ACCENT2 : i === 1 ? ACCENT : i === 4 ? ACCENT3 : ACCENT5} />
                    <div className="text-muted" style={{ fontSize: '0.72em', marginTop: -4 }}>{s.note}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card shadow-sm">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT3 }}>Unique Systemic Features</h6>
                {(bd?.systemic_features || []).map((sf, i) => (
                  <div key={i} className="mb-2">
                    <PctBar label={sf.feature} pct={sf.pct} color={i === 0 ? ACCENT3 : i === 1 ? ACCENT4 : i === 2 ? ACCENT5 : ACCENT} />
                    <div className="text-muted" style={{ fontSize: '0.72em', marginTop: -4 }}>{sf.note}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-7">
            <div className="card shadow-sm">
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: ACCENT6 }}>Treatment Table (Evidence-Graded)</h6>
                <div className="table-responsive">
                  <table className="table table-sm small">
                    <thead><tr>
                      <th>Therapy</th><th>Level</th><th>Dose / Note</th><th>Rationale</th>
                    </tr></thead>
                    <tbody>
                      {(bd?.treatments || []).map((t, i) => (
                        <tr key={i} style={{ background: t.level === 'ABSOLUTE CI' ? '#fff5f5' : '' }}>
                          <td className="fw-bold" style={{ color: t.level === 'ABSOLUTE CI' ? ACCENT7 : t.level === 'A' ? ACCENT6 : ACCENT }}>
                            {t.therapy}
                          </td>
                          <td>
                            <span className={`badge ${t.level === 'A' ? 'bg-success' : t.level === 'B' ? 'bg-info' : 'bg-danger'}`}>
                              {t.level}
                            </span>
                          </td>
                          <td className="text-muted">{t.dose}</td>
                          <td className="text-muted" style={{ fontSize: '0.78em' }}>{t.rationale}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3 — Definitions */}
      {tab === 3 && def && (
        <div className="row g-3">
          {Object.entries(def).map(([k, v]) => {
            const titles = {
              gene_function:                 { title: 'Gene & Enzyme Function (ASL, OMIM *608310)', color: ACCENT },
              pathomechanism:                { title: 'Pathomechanism — Step 4 Block → Dual Consequences', color: ACCENT2 },
              biomarker_pattern:             { title: 'Biomarker Pattern — Argininosuccinate PATHOGNOMONIC', color: ACCENT3 },
              key_distinction_from_ass1:     { title: 'Key Distinction: ASL vs ASS1 (single metabolite)', color: ACCENT4 },
              unique_systemic_features:      { title: 'Unique Systemic Features — Trichorrhexis + Hypertension (UNIQUE to ASL)', color: ACCENT3 },
              liver_transplant_note:         { title: 'Liver Transplant Caveat — Systemic Features May Persist (UNIQUE)', color: ACCENT5 },
              seizure_management:            { title: 'Seizure Management — Dual Mechanism Requires Dual Strategy', color: ACCENT2 },
              ar_inheritance_note:           { title: 'AR Inheritance, NBS, Epidemiology', color: ACCENT6 },
              unique_features_vs_other_ucd:  { title: 'What Makes ASL Unique vs All Other UCDs', color: ACCENT8 },
            };
            const meta = titles[k] || { title: k, color: ACCENT };
            return (
              <div key={k} className="col-md-6">
                <InfoBox title={meta.title} color={meta.color}>{v}</InfoBox>
              </div>
            );
          })}
        </div>
      )}

      {/* Footer nav */}
      <div className="mt-4 pt-3 border-top small text-muted d-flex gap-3 flex-wrap">
        <Link href="/ass1" style={{ color: ACCENT }}>← ASS1 (Step 3)</Link>
        <span>ASL (Step 4) — Argininosuccinic Aciduria · OMIM #207900 · 7q11.21 · Seed-217 · 40 patients</span>
        <Link href="/" style={{ color: ACCENT }}>← Home</Link>
      </div>
    </div>
  );
}

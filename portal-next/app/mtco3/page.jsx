'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
// Deep green-teal — MT-CO3 structural scaffold / proton exit pathway (no metal centres)
const COLOR  = '#1b5e20';   // deep forest green (structural / proton-exit theme)
const LIGHT  = '#e8f5e9';
const COLOR2 = '#2e7d32';
const COLOR3 = '#b71c1c';   // danger / absolute CI
const COLOR4 = '#e65100';   // warning / contraindication
const COLOR5 = '#1b5e20';   // success / treatments / normal

function KPI({ label, value, color = COLOR }) {
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

function Bar({ label, value, color = COLOR }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ variant, text }) {
  const bg     = variant === 'danger'  ? '#ffebee' : variant === 'warning' ? '#fff8e1'
               : variant === 'success' ? '#e8f5e9' : LIGHT;
  const border = variant === 'danger'  ? '#c62828' : variant === 'warning' ? '#f57f17'
               : variant === 'success' ? '#2e7d32' : COLOR;
  return (
    <div className="mb-2 p-2 rounded small" style={{ background: bg, borderLeft: `4px solid ${border}` }}>
      {text}
    </div>
  );
}

export default function MTCO3Page() {
  const [tab,  setTab]  = useState(0);
  const [ov,   setOv]   = useState(null);
  const [bk,   setBk]   = useState(null);
  const [df,   setDf]   = useState(null);
  const [err,  setErr]  = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mtco3/overview`).then(r => r.json()),
      fetch(`${API}/api/mtco3/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mtco3/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">{err}</div>;
  if (!ov)  return <div className="text-center p-5 text-muted">Loading MT-CO3…</div>;

  const k = ov.kpis;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="rounded p-3 mb-3 text-white" style={{ background: `linear-gradient(135deg,${COLOR},${COLOR2})` }}>
        <h4 className="mb-1 fw-bold">
          &#x1f9ec; MT-CO3 — {ov.protein_size}
        </h4>
        <div className="small opacity-75">{ov.disease}</div>
        <div className="small opacity-75 mt-1">
          <span className="me-3">&#x1f9ec; {ov.chromosome}</span>
          <span className="me-3">OMIM *{ov.omim_gene}</span>
          <span className="me-3">{ov.inheritance}</span>
          <span>n={ov.cohort_n} (seed {ov.seed})</span>
        </div>
      </div>

      {/* Structural-function banner */}
      <div className="alert alert-info small mb-3 py-2">
        <strong>MT-CO3 IS THE STRUCTURAL SCAFFOLD + PROTON EXIT PATHWAY:</strong>{' '}
        7 TM helices (most of 3 mtDNA CIV subunits) — <strong>NO metal centres</strong> (unlike MT-CO1 Heme-a/a3/CuB, MT-CO2 CuA) —
        wraps MT-CO1 back face — required for CIV trimer (CO1+CO2+CO3) stability —
        K-channel/D-channel proton exit from IMM matrix to IMS —
        joins CIV assembly LAST (after CO1-module + CO2-module).
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button className={`nav-link ${tab === i ? 'active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <>
          {/* KPI row */}
          <div className="row g-2 mb-4">
            <KPI label="CIV Mean Residual"  value={`${k.mean_civ_pct}%`}   color={COLOR3} />
            <KPI label="Seizures"           value={`${k.seizures_pct}%`}   color={COLOR3} />
            <KPI label="Hypotonia"          value={`${k.hypotonia_pct}%`}  color={COLOR4} />
            <KPI label="Lactic Acidosis"    value={`${k.lactic_acidosis_pct}%`} color={COLOR4} />
            <KPI label="Exercise Intol."    value={`${k.exercise_intol_pct}%`}  color={COLOR2} />
            <KPI label="Myopathy"           value={`${k.myopathy_pct}%`}   color={COLOR2} />
            <KPI label="Leigh MRI"          value={`${k.leigh_mri_pct}%`}  color={COLOR3} />
            <KPI label="SLE (MELAS-CIV)"    value={`${k.sle_mri_pct}%`}   color={COLOR4} />
            <KPI label="Median Onset"       value={`${k.median_onset_mo}mo`} />
            <KPI label="HCM"                value={`${k.hcm_pct}%`}        color={COLOR5} />
            <KPI label="Hepatopathy"        value={`${k.hepatopathy_pct}%`} color={COLOR5} />
            <KPI label="Cohort n"           value={ov.cohort_n} />
          </div>

          {/* Critical alerts */}
          <Alert variant="danger"
            text="PROPOFOL ABSOLUTE CI: directly inhibits heme a3-CuB in MT-CO1; MT-CO3 scaffold destabilisation + PRIS double CIV hit → fatal lactic acidosis. Use SEVOFLURANE." />
          <Alert variant="danger"
            text="VPA ABSOLUTE CI: CoA sequestration + POLG inhibition + CI inhibition. Use LEV." />
          <Alert variant="danger"
            text="LINEZOLID ABSOLUTE CI: blocks mt 23S rRNA → prevents MT-CO1, MT-CO2, MT-CO3 synthesis → CIV assembly collapse." />
          <Alert variant="warning"
            text="FASTING: NEVER beyond 4h without GIR 6–8 mg/kg/min IV dextrose. Fasting → β-oxidation → CIV bottleneck → crisis." />
          <Alert variant="warning"
            text="KD CONTRAINDICATED in CIV deficiency: β-oxidation → excess FADH2 → electron pressure at CIV scaffold." />
          <Alert variant="success"
            text="NO HCM (KEY DDx SCO2 — 100% HCM). NO Hepatopathy (KEY DDx SCO1). Isolated CIV (point mutations); Combined CI+CIV → suspect large deletion or LRPPRC." />

          {/* Two columns */}
          <div className="row">
            <div className="col-md-6">
              <div className="card mb-3">
                <div className="card-header fw-semibold" style={{ background: LIGHT }}>
                  Phenotype Distribution (n={ov.cohort_n})
                </div>
                <div className="card-body">
                  {ov.phenotype_distribution.map(p => (
                    <Bar key={p.class} label={p.class.split('(')[0].trim()} value={p.pct} />
                  ))}
                </div>
              </div>

              <div className="card mb-3">
                <div className="card-header fw-semibold" style={{ background: LIGHT }}>
                  Seizure Types
                </div>
                <div className="card-body">
                  {ov.seizure_types.map(s => (
                    <Bar key={s.type} label={s.type} value={s.pct} color={COLOR3} />
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-6">
              <div className="card mb-3">
                <div className="card-header fw-semibold" style={{ background: LIGHT }}>
                  Clinical Triggers
                </div>
                <div className="card-body">
                  {ov.triggers.map(t => (
                    <Bar key={t.trigger} label={t.trigger} value={t.pct} color={COLOR4} />
                  ))}
                </div>
              </div>

              <div className="card mb-3">
                <div className="card-header fw-semibold" style={{ background: LIGHT }}>
                  Key Concepts
                </div>
                <div className="card-body p-0">
                  {ov.key_concepts.map(c => (
                    <div key={c.concept} className="p-2 border-bottom">
                      <div className="fw-semibold small" style={{ color: COLOR }}>{c.concept}</div>
                      <div className="text-muted small mt-1">{c.detail.slice(0, 220)}{c.detail.length > 220 ? '…' : ''}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Module info */}
          <div className="card mb-3">
            <div className="card-header fw-semibold" style={{ background: LIGHT }}>Gene / Protein Module</div>
            <div className="card-body small">
              <strong>Module:</strong> {ov.module}<br />
              <strong>Size:</strong> {ov.protein_size}<br />
              <strong>Locus:</strong> {ov.chromosome}<br />
              <strong>Inheritance:</strong> {ov.inheritance}<br />
              <strong>OMIM Gene:</strong> *{ov.omim_gene} &nbsp; <strong>OMIM Disease:</strong> #{ov.omim_disease}
            </div>
          </div>
        </>
      )}

      {/* ── TAB 1: PATIENTS & FEATURES ── */}
      {tab === 1 && bk && (
        <>
          <h6 className="fw-semibold mb-3">Variant Summary (n={bk.cohort_n})</h6>
          {bk.variants.map(v => (
            <div key={v.variant} className="card mb-3">
              <div className="card-header d-flex justify-content-between align-items-start"
                style={{ background: LIGHT }}>
                <div>
                  <span className="fw-bold" style={{ color: COLOR }}>{v.variant}</span>
                  <span className="text-muted small ms-2">{v.amino_acid}</span>
                </div>
                <span className="badge rounded-pill" style={{ background: COLOR, color: '#fff' }}>
                  {v.freq_pct}% · n={v.n_in_cohort}
                </span>
              </div>
              <div className="card-body small">
                <div className="mb-1"><strong>Structural impact:</strong> {v.structural_impact}</div>
                <div className="mb-1"><strong>Modal phenotype:</strong> {v.modal_phenotype}</div>
                <div className="text-muted">{v.detail.slice(0, 400)}{v.detail.length > 400 ? '…' : ''}</div>
              </div>
            </div>
          ))}

          <h6 className="fw-semibold mb-2 mt-4">Patient Table (n={bk.cohort_n})</h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover small">
              <thead className="table-dark">
                <tr>
                  <th>ID</th><th>Phenotype</th><th>Variant</th>
                  <th>CIV%</th><th>Onset(mo)</th>
                  <th>Sz</th><th>Hypo</th><th>LA</th>
                  <th>LeighMRI</th><th>SLE</th><th>Myo</th><th>Exer</th>
                  <th>Ophthalmo</th><th>HB</th><th>HCM</th>
                </tr>
              </thead>
              <tbody>
                {bk.patients.map(p => (
                  <tr key={p.id}>
                    <td>{p.id}</td>
                    <td style={{ maxWidth: 180, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {p.phenotype.split('(')[0].trim()}
                    </td>
                    <td>{p.variant}</td>
                    <td><span className={`badge ${p.civ_pct < 10 ? 'bg-danger' : p.civ_pct < 30 ? 'bg-warning text-dark' : 'bg-success'}`}>{p.civ_pct}%</span></td>
                    <td>{p.onset_mo}</td>
                    <td>{p.seizure ? '✓' : ''}</td>
                    <td>{p.hypotonia ? '✓' : ''}</td>
                    <td>{p.lactic_ac ? '✓' : ''}</td>
                    <td>{p.leigh_mri ? '✓' : ''}</td>
                    <td>{p.sle_mri ? '✓' : ''}</td>
                    <td>{p.myopathy ? '✓' : ''}</td>
                    <td>{p.exercise_intol ? '✓' : ''}</td>
                    <td>{p.ophthalmo ? '✓' : ''}</td>
                    <td>{p.heart_block ? '✓' : ''}</td>
                    <td className="text-success">{p.hcm ? '✓' : '✗'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── TAB 2: TREATMENTS & DDx ── */}
      {tab === 2 && bk && (
        <>
          <div className="row">
            <div className="col-md-6">
              <h6 className="fw-semibold mb-2">Treatments</h6>
              {bk.treatments.map(t => (
                <div key={t.name} className="card mb-2">
                  <div className="card-body py-2 small">
                    <div className="fw-semibold" style={{ color: COLOR5 }}>{t.name}</div>
                    <div className="text-muted">{t.evidence}</div>
                    <div className="mt-1">{t.notes.slice(0, 280)}{t.notes.length > 280 ? '…' : ''}</div>
                  </div>
                </div>
              ))}
            </div>

            <div className="col-md-6">
              <h6 className="fw-semibold mb-2">Contraindications</h6>
              {bk.contraindications.map(c => (
                <div key={c.drug} className="card mb-2 border-danger">
                  <div className="card-body py-2 small">
                    <div className="fw-bold text-danger">{c.drug}</div>
                    <div className="text-warning fw-semibold">{c.class}</div>
                    <div className="text-muted mt-1">{c.reason.slice(0, 260)}{c.reason.length > 260 ? '…' : ''}</div>
                  </div>
                </div>
              ))}

              <h6 className="fw-semibold mb-2 mt-3">Monitoring</h6>
              {bk.monitoring.map(m => (
                <div key={m.item} className="d-flex border-bottom py-1 small">
                  <div className="fw-semibold me-2" style={{ minWidth: 160, color: COLOR }}>{m.item}</div>
                  <div className="text-muted">{m.protocol}</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* ── TAB 3: DEFINITIONS ── */}
      {tab === 3 && df && (
        <>
          <h6 className="fw-semibold mb-3">Glossary — {df.gene}</h6>
          {df.glossary.map(g => (
            <div key={g.term} className="mb-3 p-2 rounded border-start border-3"
              style={{ borderColor: COLOR, background: LIGHT }}>
              <div className="fw-semibold small" style={{ color: COLOR }}>{g.term}</div>
              <div className="small text-muted mt-1">{g.definition}</div>
            </div>
          ))}

          <h6 className="fw-semibold mb-3 mt-4">References</h6>
          <ol className="small text-muted">
            {df.references.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
          </ol>
        </>
      )}
    </div>
  );
}

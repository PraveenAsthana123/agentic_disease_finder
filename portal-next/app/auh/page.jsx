'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// AUH / 3-methylglutaconyl-CoA hydratase — step 4 leucine catabolism colour scheme
const ACCENT  = '#1b5e20';   // deep green — AUH enzyme / step 4 leucine catabolism
const ACCENT2 = '#bf360c';   // deep orange-red — 3-MGC primary marker / accumulation
const ACCENT3 = '#006064';   // deep teal — 3-MG secondary / leucine pathway
const ACCENT4 = '#1565c0';   // deep blue — KEY NEGATIVES / 3-HMG normal
const ACCENT5 = '#2e7d32';   // mid-green — ketones PRESENT (ketogenesis INTACT)
const ACCENT6 = '#e65100';   // orange — moderate risk / VPA caution
const ACCENT7 = '#4a148c';   // deep purple — intellectual disability / severe
const ACCENT8 = '#37474f';   // dark slate — MGA Type I context / systemic

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

export default function AUHPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/auh/overview`).then(r => r.json()),
      fetch(`${API}/api/auh/breakdown`).then(r => r.json()),
      fetch(`${API}/api/auh/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading AUH / 3-Methylglutaconyl-CoA Hydratase Deficiency dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="card shadow mb-3" style={{ borderTop: `5px solid ${ACCENT}` }}>
        <div className="card-body pb-2">
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            &#x1f9ec; AUH Epilepsy — 3-Methylglutaconyl-CoA Hydratase Deficiency (MGA Type I)
          </h4>
          <div className="text-muted small">
            AUH · Leucine Catabolism STEP 4 (between MCC Step 3 and HMGCL Step 5) ·{' '}
            <strong>3-MGC ELEVATED · 3-HMG NORMAL · Ketones PRESENT · VPA = Moderate Risk Only</strong> ·
            AR · 9q22.31 · OMIM Gene *600529 · Disease #250950
          </div>
          <div className="mt-2 d-flex gap-2 flex-wrap">
            <span className="badge" style={{ background: ACCENT }}>Leucine Catabolism Step 4</span>
            <span className="badge" style={{ background: ACCENT2 }}>3-MGC ELEVATED (Primary)</span>
            <span className="badge" style={{ background: ACCENT4 }}>3-HMG NORMAL (Key Negative vs HMGCL)</span>
            <span className="badge" style={{ background: ACCENT5 }}>Ketogenesis INTACT</span>
            <span className="badge" style={{ background: ACCENT8 }}>MGA Type I (AUH)</span>
            <span className="badge" style={{ background: ACCENT6 }}>VPA Moderate Risk (Not Absolute CI)</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* Tab 0: Overview */}
      {tab === 0 && ov && (
        <div>
          <Alert
            text="AUH deficiency (MGA Type I): 3-MGC ELEVATED · 3-HMG NORMAL (KEY NEGATIVE vs HMGCL) · Ketones PRESENT (ketogenesis INTACT) · 3-MCG ABSENT (KEY NEGATIVE vs 3-MCC) · VPA = Moderate Risk only (NOT absolute CI) · Leucine restriction + L-Carnitine = Level A"
            variant="success"
          />

          {/* KPI row */}
          <div className="row mb-3">
            {Object.values(kpi).map((k, i) => (
              <KPI key={i} label={k.label} value={k.value} color={k.color} />
            ))}
          </div>

          {/* Phenotype distribution */}
          <div className="row mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-body">
                  <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Phenotype Distribution (n=40)</h6>
                  {(ov.phenotype_dist || []).map((ph, i) => (
                    <PctBar key={i} label={`${ph.class} (n=${ph.n})`} pct={ph.pct}
                      color={i === 0 ? ACCENT : i === 1 ? ACCENT5 : ACCENT7} />
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <InfoBox title="Ketogenesis INTACT — Key Distinction from HMGCL" color={ACCENT5}>
                {ov.ketogenesis_intact_note}
              </InfoBox>
              <InfoBox title="MGA Type I Context" color={ACCENT8}>
                {ov.mga_type_note}
              </InfoBox>
            </div>
          </div>

          {/* Hallmark biomarker box */}
          <InfoBox title="Biomarker Pattern Summary" color={ACCENT2}>
            {ov.hallmark_biomarker}
          </InfoBox>

          {/* VPA risk note */}
          <InfoBox title="VPA — Moderate Risk (NOT Absolute CI unlike HMGCL)" color={ACCENT6}>
            {ov.vpa_risk_note}
          </InfoBox>

          {/* Disease overview table */}
          <div className="table-responsive">
            <table className="table table-sm table-bordered">
              <tbody>
                <tr><th>Gene</th><td>{ov.gene}</td><th>OMIM Gene</th><td>*{ov.omim_gene}</td></tr>
                <tr><th>Disease</th><td colSpan={3}>{ov.disease}</td></tr>
                <tr><th>OMIM Disease</th><td>#{ov.omim_disease}</td><th>Locus</th><td>{ov.locus}</td></tr>
                <tr><th>Inheritance</th><td>{ov.inheritance}</td><th>Prevalence</th><td>{ov.prevalence}</td></tr>
                <tr><th>Pathway Step</th><td colSpan={3}>{ov.pathway_step}</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 1: Patients & Biomarkers */}
      {tab === 1 && bd && (
        <div>
          {/* Biomarkers */}
          <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Biomarker Pattern — AUH Deficiency</h6>
          <div className="row">
            {Object.entries(bd.biomarkers || {}).map(([key, bm]) => (
              <div key={key} className="col-md-6 mb-3">
                <div className={`card border-${bm.color} shadow-sm h-100`}>
                  <div className="card-body py-2">
                    <div className="d-flex justify-content-between align-items-start mb-1">
                      <strong className="small">{bm.label}</strong>
                      <span className={`badge bg-${bm.color} ms-1`}>{bm.direction}</span>
                    </div>
                    <div className="small text-muted mb-1"><em>Normal: {bm.normal}</em></div>
                    <div className="small fw-bold mb-1">{bm.status}</div>
                    <div className="small text-muted">{bm.rationale}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Enzyme mechanism */}
          {bd.enzyme_mechanism && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>AUH Enzyme Mechanism</h6>
                <div className="small text-muted mb-2">{bd.enzyme_mechanism.function}</div>
                <div className="alert alert-success py-2 small">
                  <strong>Reaction:</strong> {bd.enzyme_mechanism.reaction}
                </div>
                <div className="alert alert-warning py-2 small">
                  <strong>Block:</strong> <pre className="mb-0 small bg-transparent border-0 p-0">{bd.enzyme_mechanism.block}</pre>
                </div>
                <div className="alert alert-info py-2 small">
                  <strong>Leucine Catabolism Step 4:</strong> {bd.enzyme_mechanism.leucine_path}
                </div>
                <div className="alert alert-success py-2 small">
                  <strong>Ketogenesis (INTACT):</strong> {bd.enzyme_mechanism.ketogenesis_note}
                </div>
              </div>
            </div>
          )}

          {/* Variants */}
          <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Common Pathogenic Variants</h6>
          <div className="table-responsive mb-3">
            <table className="table table-sm table-bordered table-hover">
              <thead className="table-dark">
                <tr><th>Variant</th><th>Freq %</th><th>Domain</th><th>Phenotype</th><th>Note</th></tr>
              </thead>
              <tbody>
                {(bd.variants || []).map((v, i) => (
                  <tr key={i}>
                    <td><code>{v.variant}</code></td>
                    <td>{v.freq}%</td>
                    <td>{v.domain}</td>
                    <td>{v.phenotype}</td>
                    <td className="text-muted small">{v.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Cohort preview */}
          <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Patient Cohort Preview (first 10 / 40)</h6>
          <div className="table-responsive">
            <table className="table table-sm table-striped table-bordered small">
              <thead className="table-dark">
                <tr>
                  <th>ID</th><th>Phenotype</th><th>Variant</th><th>Onset (mo)</th>
                  <th>3-MGC (mmol/mol Cr)</th><th>3-MG</th><th>3-HMG (NORMAL)</th>
                  <th>C5-OH</th><th>Ketones</th><th>Seizures</th><th>IDD</th>
                </tr>
              </thead>
              <tbody>
                {(bd.cohort_preview || []).map((p, i) => (
                  <tr key={i}>
                    <td><code>{p.id}</code></td>
                    <td>{p.phenotype.split(' (')[0]}</td>
                    <td><code>{p.variant}</code></td>
                    <td>{p.age_onset_months}</td>
                    <td className="fw-bold text-danger">{p.mgc_urine_mmol_mol_cr}</td>
                    <td>{p.mg_urine_mmol_mol_cr}</td>
                    <td className="text-success fw-bold">{p.hmg_3_urine} (Normal)</td>
                    <td>{p.c5oh_umol_l}</td>
                    <td><span className={`badge ${p.ketones_present ? 'bg-success' : 'bg-danger'}`}>{p.ketones_present ? 'Present ✓' : 'Absent'}</span></td>
                    <td>{p.seizures ? '✓' : '—'}</td>
                    <td>{p.intellectual_disability ? '✓' : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 2: Seizures & Treatments */}
      {tab === 2 && bd && (
        <div>
          <div className="row">
            {/* Seizure types */}
            <div className="col-md-5">
              <div className="card shadow-sm mb-3">
                <div className="card-body">
                  <h6 className="fw-bold mb-3" style={{ color: ACCENT2 }}>Seizure Types in AUH Deficiency</h6>
                  {(bd.seizure_types || []).map((s, i) => (
                    <div key={i} className="mb-3">
                      <div className="d-flex justify-content-between align-items-center mb-1">
                        <span className="small fw-bold">{s.type}</span>
                        <span className="badge" style={{ background: ACCENT2 }}>{s.pct}%</span>
                      </div>
                      <div className="progress mb-1" style={{ height: 8 }}>
                        <div className="progress-bar" style={{ width: `${s.pct}%`, backgroundColor: ACCENT2 }} />
                      </div>
                      <div className="small text-muted">{s.note}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Systemic features */}
            <div className="col-md-7">
              <div className="card shadow-sm mb-3">
                <div className="card-body">
                  <h6 className="fw-bold mb-3" style={{ color: ACCENT8 }}>Systemic Features</h6>
                  {(bd.systemic_features || []).map((f, i) => (
                    <div key={i} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className={f.pct === 0 ? 'text-success fw-bold' : ''}>{f.feature}</span>
                        <span className={`fw-bold ${f.pct === 0 ? 'text-success' : 'text-muted'}`}>{f.pct}%</span>
                      </div>
                      {f.pct > 0 && (
                        <div className="progress mb-1" style={{ height: 6 }}>
                          <div className="progress-bar" style={{ width: `${f.pct}%`, backgroundColor: f.pct === 0 ? '#2e7d32' : ACCENT8 }} />
                        </div>
                      )}
                      <div className="small text-muted">{f.note}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Treatments */}
          <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Treatment Ladder — AUH Deficiency</h6>
          <div className="row">
            {(bd.treatments || []).map((t, i) => {
              const lvl = t.level;
              const borderColor =
                lvl === 'A' ? '#1b5e20' :
                lvl === 'B' ? '#1565c0' :
                lvl === 'MODERATE RISK' ? '#e65100' :
                lvl === 'NOT EFFECTIVE' ? '#9e9e9e' :
                lvl === 'NOT ABSOLUTE CI (caution)' ? '#f9a825' : '#9e9e9e';
              const badgeBg =
                lvl === 'A' ? '#1b5e20' :
                lvl === 'B' ? '#1565c0' :
                lvl === 'MODERATE RISK' ? '#e65100' :
                '#757575';
              return (
                <div key={i} className="col-md-6 mb-3">
                  <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${borderColor}` }}>
                    <div className="card-body py-2">
                      <div className="d-flex justify-content-between align-items-start mb-1">
                        <strong className="small">{t.therapy}</strong>
                        <span className="badge ms-1 text-nowrap" style={{ background: badgeBg, fontSize: 10 }}>Level {lvl}</span>
                      </div>
                      <div className="small text-muted mb-1"><em>Dose: {t.dose}</em></div>
                      <div className="small text-muted">{t.rationale}</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Tab 3: Definitions */}
      {tab === 3 && def && (
        <div>
          <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Definitions & Key Concepts — AUH / MGA-I</h6>
          {Object.entries(def).map(([term, defn]) => (
            <div key={term} className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT}` }}>
              <div className="card-body py-2">
                <div className="fw-bold small mb-1" style={{ color: ACCENT }}>{term}</div>
                <div className="small text-muted" style={{ whiteSpace: 'pre-line' }}>{defn}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Footer navigation */}
      <div className="mt-4 d-flex gap-2">
        <Link href="/mcc" className="btn btn-outline-secondary btn-sm">← MCC (Step 3)</Link>
        <Link href="/hmgcl" className="btn btn-outline-secondary btn-sm">HMGCL (Step 5) →</Link>
        <Link href="/expert-dashboards-catalog" className="btn btn-outline-primary btn-sm ms-auto">Dashboard Catalog</Link>
      </div>
    </div>
  );
}

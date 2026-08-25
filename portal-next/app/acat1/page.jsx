'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// ACAT1 / Beta-Ketothiolase Deficiency / T2 Deficiency colour scheme
const ACCENT  = '#b71c1c';   // deep red — T2 deficiency / ketoacidotic crisis
const ACCENT2 = '#4e342e';   // dark brown — 2M3HBA primary marker / isoleucine catabolism
const ACCENT3 = '#e65100';   // deep orange — tiglylglycine pathognomonic
const ACCENT4 = '#1b5e20';   // deep green — KEY NEGATIVES / C3 NORMAL
const ACCENT5 = '#c62828';   // mid-red — HYPERKETOSIS / crisis
const ACCENT6 = '#bf360c';   // orange-red — HIGH RISK warnings (VPA)
const ACCENT7 = '#1a237e';   // deep navy — absolute CI (KD)
const ACCENT8 = '#37474f';   // dark slate — systemic / dual block

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

export default function ACAT1Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/acat1/overview`).then(r => r.json()),
      fetch(`${API}/api/acat1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/acat1/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading ACAT1 / Beta-Ketothiolase Deficiency dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="card shadow mb-3" style={{ borderTop: `5px solid ${ACCENT}` }}>
        <div className="card-body pb-2">
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            &#x1f9ec; ACAT1 Epilepsy — Beta-Ketothiolase Deficiency (T2 Deficiency)
          </h4>
          <div className="text-muted small">
            ACAT1 · Isoleucine Catabolism STEP 4 (FINAL) + Ketone Utilisation ·{' '}
            <strong>2M3HBA ELEVATED · Tiglylglycine PATHOGNOMONIC · C3 NORMAL (not PA) · HYPERKETOSIS · KD = ABSOLUTE CI</strong> ·
            AR · 11q22.3 · OMIM Gene *607809 · Disease #203750
          </div>
          <div className="mt-2 d-flex gap-2 flex-wrap">
            <span className="badge" style={{ background: ACCENT }}>Isoleucine Catabolism Step 4 (Final)</span>
            <span className="badge" style={{ background: ACCENT2 }}>2M3HBA ELEVATED (Primary)</span>
            <span className="badge" style={{ background: ACCENT3 }}>Tiglylglycine PATHOGNOMONIC</span>
            <span className="badge" style={{ background: ACCENT4 }}>C3 NORMAL (Key Negative vs PA)</span>
            <span className="badge" style={{ background: ACCENT5 }}>HYPERKETOSIS (not HYPOketosis)</span>
            <span className="badge" style={{ background: ACCENT7 }}>KD = ABSOLUTE CI</span>
            <span className="badge" style={{ background: ACCENT6 }}>VPA HIGH RISK</span>
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
            text="ACAT1 deficiency (T2 deficiency): DUAL BLOCK — isoleucine catabolism STEP 4 (2M3HBA + tiglylglycine ↑) AND ketone body utilisation (HYPERKETOSIS). C3 NORMAL (KEY NEGATIVE vs PA). KD = ABSOLUTE CI (floods acetoacetyl-CoA that cannot be utilised). EPISODIC disease — biomarkers can be NORMAL between crises."
            variant="danger"
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
                      color={i === 0 ? ACCENT : i === 1 ? '#0d47a1' : i === 2 ? ACCENT3 : ACCENT7} />
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <InfoBox title="DUAL BLOCK — Isoleucine + Ketolysis" color={ACCENT8}>
                {ov.dual_block_note}
              </InfoBox>
              <InfoBox title="Episodic Disease — When to Test" color={ACCENT3}>
                {ov.episodic_note}
              </InfoBox>
            </div>
          </div>

          {/* Key biomarker box */}
          <InfoBox title="Biomarker Pattern Summary" color={ACCENT2}>
            {ov.hallmark_biomarker}
          </InfoBox>

          {/* C3 normal exam pearl */}
          <InfoBox title="C3 NORMAL — KEY NEGATIVE vs PA (Critical Exam Pearl)" color={ACCENT4}>
            {ov.c3_normal_note}
          </InfoBox>

          {/* KD CI */}
          <InfoBox title="KD ABSOLUTE CI — Critical Treatment Distinction (KD Helps PDH, KD Kills ACAT1)" color={ACCENT7}>
            {ov.kd_ci_note}
          </InfoBox>

          {/* VPA risk */}
          <InfoBox title="VPA — HIGH RISK (not absolute CI but avoid if possible)" color={ACCENT6}>
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
          <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Biomarker Pattern — ACAT1 Deficiency</h6>
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
                <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>ACAT1 Enzyme Mechanism — Dual Reactions</h6>
                <div className="small text-muted mb-2">{bd.enzyme_mechanism.function}</div>
                <div className="alert alert-danger py-2 small">
                  <strong>Reaction A (Isoleucine — BLOCKED):</strong>
                  <pre className="mb-0 small bg-transparent border-0 p-0">{bd.enzyme_mechanism.reaction_a}</pre>
                </div>
                <div className="alert alert-danger py-2 small">
                  <strong>Reaction B (Ketolysis — BLOCKED):</strong>
                  <pre className="mb-0 small bg-transparent border-0 p-0">{bd.enzyme_mechanism.reaction_b}</pre>
                </div>
                <div className="alert alert-info py-2 small">
                  <strong>Isoleucine Catabolism Context (ACAT1 = Last Step):</strong>
                  <pre className="mb-0 small bg-transparent border-0 p-0">{bd.enzyme_mechanism.isoleucine_path}</pre>
                </div>
                <div className="alert alert-success py-2 small">
                  <strong>Why C3 is NORMAL (Exam Pearl):</strong>
                  <pre className="mb-0 small bg-transparent border-0 p-0">{bd.enzyme_mechanism.why_c3_normal}</pre>
                </div>
                <div className="alert alert-warning py-2 small">
                  <strong>HYPERKETOSIS Mechanism:</strong>
                  <pre className="mb-0 small bg-transparent border-0 p-0">{bd.enzyme_mechanism.ketone_hyperketosis}</pre>
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
                  <th>2M3HBA (mmol/mol Cr)</th><th>Tiglylglycine</th><th>C5:1 (µmol/L)</th>
                  <th>β-OHB (mmol/L)</th><th>C3 (NORMAL)</th><th>Seizures</th><th>ID</th>
                </tr>
              </thead>
              <tbody>
                {(bd.cohort_preview || []).map((p, i) => (
                  <tr key={i}>
                    <td><code>{p.id}</code></td>
                    <td>{p.phenotype.split(' (')[0]}</td>
                    <td><code>{p.variant}</code></td>
                    <td>{p.age_onset_months}</td>
                    <td className="fw-bold text-danger">{p.m3hba_mmol_mol_cr}</td>
                    <td>{p.tiglylglycine_mmol_mol_cr}</td>
                    <td>{p.c5_1_umol_l}</td>
                    <td className={p.ketoacidotic_crisis ? 'fw-bold text-danger' : ''}>{p.bohb_mmol_l}</td>
                    <td className="text-success fw-bold">{p.c3_umol_l} ✓ Normal</td>
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
                  <h6 className="fw-bold mb-3" style={{ color: ACCENT5 }}>Seizure Types in ACAT1 Deficiency</h6>
                  {(bd.seizure_types || []).map((s, i) => (
                    <div key={i} className="mb-3">
                      <div className="d-flex justify-content-between align-items-center mb-1">
                        <span className="small fw-bold">{s.type}</span>
                        <span className="badge" style={{ background: ACCENT5 }}>{s.pct}%</span>
                      </div>
                      <div className="progress mb-1" style={{ height: 8 }}>
                        <div className="progress-bar" style={{ width: `${s.pct}%`, backgroundColor: ACCENT5 }} />
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
                          <div className="progress-bar" style={{ width: `${f.pct}%`, backgroundColor: ACCENT8 }} />
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
          <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Treatment Ladder — ACAT1 Deficiency</h6>
          <div className="row">
            {(bd.treatments || []).map((t, i) => {
              const lvl = t.level;
              const borderColor =
                lvl === 'A' ? '#1b5e20' :
                lvl === 'B' ? '#1565c0' :
                lvl === 'HIGH RISK' ? '#e65100' :
                lvl === 'ABSOLUTE CI' ? '#b71c1c' :
                lvl === 'EXTREME HAZARD' ? '#4a148c' : '#9e9e9e';
              const badgeBg =
                lvl === 'A' ? '#1b5e20' :
                lvl === 'B' ? '#1565c0' :
                lvl === 'HIGH RISK' ? '#e65100' :
                lvl === 'ABSOLUTE CI' ? '#b71c1c' :
                lvl === 'EXTREME HAZARD' ? '#4a148c' : '#757575';
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
          <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Definitions & Key Concepts — ACAT1 / T2 Deficiency</h6>
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
        <Link href="/ivd" className="btn btn-outline-secondary btn-sm">← IVD (Isovaleric Acidemia)</Link>
        <Link href="/pcca" className="btn btn-outline-secondary btn-sm">PCCA (Propionic Acidemia) →</Link>
        <Link href="/expert-dashboards-catalog" className="btn btn-outline-primary btn-sm ms-auto">Dashboard Catalog</Link>
      </div>
    </div>
  );
}

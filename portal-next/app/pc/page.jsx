'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// PC / Pyruvate Carboxylase Deficiency — colour scheme
const ACCENT  = '#b71c1c';   // deep red — lactic acidosis / primary marker
const ACCENT2 = '#c62828';   // crimson — pyruvate accumulation / L:P ratio
const ACCENT3 = '#e64a19';   // deep orange — alanine elevation / OAA depletion
const ACCENT4 = '#1565c0';   // deep blue — KEY NEGATIVES / C5-OH NORMAL
const ACCENT5 = '#2e7d32';   // deep green — biotin-dependent (partial; PC is ONE of 4 carboxylases)
const ACCENT6 = '#4a148c';   // deep purple — intellectual disability / type B severe
const ACCENT7 = '#bf360c';   // dark orange-red — KD ABSOLUTE CI warning
const ACCENT8 = '#37474f';   // dark slate — type context / systemic

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

export default function PCPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/pc/overview`).then(r => r.json()),
      fetch(`${API}/api/pc/breakdown`).then(r => r.json()),
      fetch(`${API}/api/pc/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading PC / Pyruvate Carboxylase Deficiency dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="card shadow mb-3" style={{ borderTop: `5px solid ${ACCENT}` }}>
        <div className="card-body pb-2">
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            &#x1f9ec; PC Epilepsy — Pyruvate Carboxylase Deficiency (PCD)
          </h4>
          <div className="text-muted small">
            PC · Anaplerosis BLOCKED (Pyruvate &#x2192; OAA) ·{' '}
            <strong>Lactate ↑↑↑ · L:P &gt;20:1 · Alanine ↑↑ · KD = ABSOLUTE CI · Glucose = TREATMENT</strong> ·
            AR · 11q13.2 · OMIM Gene *608786 · Disease #266150
          </div>
          <div className="mt-1">
            <span className="badge me-1" style={{ background: ACCENT }}>Lactic Acidosis PRIMARY</span>
            <span className="badge me-1" style={{ background: ACCENT3 }}>Alanine ↑↑</span>
            <span className="badge me-1" style={{ background: ACCENT4 }}>C5-OH NORMAL (KEY NEG)</span>
            <span className="badge me-1" style={{ background: ACCENT7 }}>KD ABSOLUTE CI</span>
            <span className="badge me-1" style={{ background: ACCENT5 }}>Biotin Level B (partial)</span>
            <span className="badge" style={{ background: ACCENT8 }}>1:250,000 (1:3,000 Cree)</span>
          </div>
        </div>

        {/* Tabs */}
        <div className="card-footer p-0">
          <ul className="nav nav-tabs border-0">
            {TABS.map((t, i) => (
              <li key={i} className="nav-item">
                <button
                  className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
                  style={tab === i ? { borderBottom: `3px solid ${ACCENT}`, color: ACCENT } : {}}
                  onClick={() => setTab(i)}
                >{t}</button>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* ── TAB 0: OVERVIEW ─────────────────────────────────── */}
      {tab === 0 && (
        <div>
          <Alert variant="danger" text="⚠ CRITICAL: Ketogenic Diet is ABSOLUTE CONTRAINDICATION in PC deficiency — KD worsens lactic acidosis catastrophically (OAA absent → Acetyl-CoA CANNOT enter TCA). IV Glucose is TREATMENT, not KD." />
          <Alert variant="warning" text="PC vs PDH Differential: PC deficiency L:P ratio >20:1 (NADH excess); PDH deficiency L:P <10:1 (pyruvate cannot be oxidised, NADH not excess). KD helps PDH, kills PC." />

          {/* KPI row */}
          <div className="row mb-3">
            {Object.values(kpi).map((k, i) => (
              <KPI key={i} label={k.label} value={k.value} color={k.color} />
            ))}
          </div>

          {/* Phenotype distribution */}
          <div className="row g-3 mb-3">
            <div className="col-md-5">
              <div className="card shadow-sm h-100">
                <div className="card-header py-2 fw-bold small" style={{ color: ACCENT }}>Phenotype Distribution (n=40)</div>
                <div className="card-body">
                  {(ov?.phenotype_dist || []).map((p, i) => (
                    <PctBar key={i} label={`${p.class} (n=${p.n})`} pct={p.pct}
                      color={i === 0 ? ACCENT : i === 1 ? ACCENT6 : ACCENT4} />
                  ))}
                  <div className="text-muted small mt-2">Type A (North American): 50% modal · Type B (French/severe): 35% · Type C (benign): 15%</div>
                </div>
              </div>
            </div>

            <div className="col-md-7">
              <InfoBox title="🔑 PC Anaplerosis — Why Glucose is Treatment (not KD)" color={ACCENT}>
                {ov?.kd_absolute_ci_note}
              </InfoBox>
              <InfoBox title="🔬 PC vs PDH Deficiency — Critical L:P Ratio Differential" color={ACCENT2}>
                {ov?.pc_vs_pdh_note}
              </InfoBox>
            </div>
          </div>

          {/* Hallmark + notes */}
          <div className="row g-3">
            <div className="col-md-6">
              <InfoBox title="🧬 Hallmark Biomarkers (PC Deficiency)" color={ACCENT}>
                {ov?.hallmark_biomarker}
              </InfoBox>
            </div>
            <div className="col-md-6">
              <InfoBox title="💊 Biotin in PC Deficiency (Level B — Partial)" color={ACCENT5}>
                {ov?.biotin_note}
              </InfoBox>
              <InfoBox title="⚠ VPA Risk in PC Deficiency" color={ACCENT7}>
                {ov?.vpa_risk_note}
              </InfoBox>
            </div>
          </div>

          {/* Disease context */}
          <InfoBox title="📍 Pathway Context — Where PC Fits" color={ACCENT8}>
            {ov?.pathway_context}
          </InfoBox>
        </div>
      )}

      {/* ── TAB 1: PATIENTS & BIOMARKERS ────────────────────── */}
      {tab === 1 && (
        <div>
          {/* Biomarkers */}
          <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Biomarker Panel — PC Deficiency</h6>
          <div className="row g-3 mb-4">
            {Object.entries(bd?.biomarkers || {}).map(([key, bm]) => (
              <div key={key} className="col-md-6">
                <div className={`card shadow-sm border-${bm.color} h-100`} style={{ borderLeft: `4px solid` }}>
                  <div className="card-body py-2">
                    <div className="d-flex justify-content-between align-items-start">
                      <span className="fw-bold small">{bm.label}</span>
                      <span className={`badge bg-${bm.color}`}>{bm.direction}</span>
                    </div>
                    <div className="text-muted small mt-1"><strong>Normal:</strong> {bm.normal}</div>
                    <div className="small mt-1" style={{ color: bm.color === 'success' ? '#2e7d32' : bm.color === 'danger' ? '#b71c1c' : '#e65100' }}>
                      {bm.status}
                    </div>
                    <div className="text-muted small mt-1" style={{ fontSize: 11 }}>{bm.rationale}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Enzyme mechanism */}
          <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>PC Enzyme — Mechanism & Pathway Block</h6>
          <div className="row g-3 mb-4">
            {Object.entries(bd?.enzyme_mechanism || {}).map(([key, val]) => (
              <div key={key} className="col-md-6">
                <InfoBox title={key.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())} color={ACCENT}>
                  <pre className="mb-0" style={{ whiteSpace:'pre-wrap', fontSize:11 }}>{val}</pre>
                </InfoBox>
              </div>
            ))}
          </div>

          {/* Cohort table */}
          <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Cohort Preview (first 10 of 40 patients)</h6>
          <div className="table-responsive">
            <table className="table table-sm table-bordered table-hover" style={{ fontSize: 11 }}>
              <thead className="table-dark">
                <tr>
                  <th>ID</th><th>Phenotype</th><th>Variant</th><th>Onset (mo)</th>
                  <th>Lactate (mmol/L)</th><th>L:P Ratio</th><th>Alanine (µmol/L)</th>
                  <th>NH3 (µmol/L)</th><th>C5-OH</th><th>C3</th>
                  <th>Seizures</th><th>IDD</th><th>Hypoglycemia</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.cohort_preview || []).map((p, i) => (
                  <tr key={i}>
                    <td className="fw-bold">{p.id}</td>
                    <td>{p.phenotype.split(' (')[0]}</td>
                    <td><code>{p.variant}</code></td>
                    <td>{p.age_onset_months}</td>
                    <td style={{ color: ACCENT, fontWeight:'bold' }}>{p.lactate_mmol_l}</td>
                    <td style={{ color: p.lp_ratio > 30 ? '#b71c1c' : '#e65100', fontWeight:'bold' }}>{p.lp_ratio}</td>
                    <td style={{ color: p.alanine_umol_l > 600 ? ACCENT3 : 'inherit' }}>{p.alanine_umol_l}</td>
                    <td style={{ color: p.nh3_umol_l > 100 ? '#7b1fa2' : 'inherit' }}>{p.nh3_umol_l}</td>
                    <td style={{ color: ACCENT4 }}>{p.c5oh_umol_l} ✓</td>
                    <td style={{ color: ACCENT4 }}>{p.c3_umol_l} ✓</td>
                    <td>{p.seizures ? '✅' : '—'}</td>
                    <td>{p.intellectual_disability ? '✅' : '—'}</td>
                    <td>{p.hypoglycemia ? '⚠' : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Variants */}
          <h6 className="fw-bold mb-2 mt-3" style={{ color: ACCENT }}>Common Pathogenic Variants</h6>
          <div className="table-responsive">
            <table className="table table-sm table-bordered" style={{ fontSize: 11 }}>
              <thead className="table-secondary">
                <tr><th>Variant</th><th>Freq (%)</th><th>Domain</th><th>Phenotype</th><th>Note</th></tr>
              </thead>
              <tbody>
                {(bd?.variants || []).map((v, i) => (
                  <tr key={i}>
                    <td><code className="fw-bold">{v.variant}</code></td>
                    <td>{v.freq}%</td>
                    <td>{v.domain}</td>
                    <td>{v.phenotype}</td>
                    <td className="text-muted">{v.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── TAB 2: SEIZURES & TREATMENTS ────────────────────── */}
      {tab === 2 && (
        <div>
          <Alert variant="danger" text="⚠ ABSOLUTE CI: Ketogenic Diet is NEVER used in PC deficiency. KD worsens lactic acidosis (no OAA → Acetyl-CoA cannot enter TCA → NADH excess worsens). IV Glucose is treatment." />

          <div className="row g-3 mb-4">
            <div className="col-md-6">
              <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Seizure Types in PC Deficiency</h6>
              {(bd?.seizure_types || []).map((s, i) => (
                <div key={i} className="card shadow-sm mb-2">
                  <div className="card-body py-2">
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <span className="fw-bold small">{s.type}</span>
                      <span className="badge" style={{ background: ACCENT }}>{s.pct}%</span>
                    </div>
                    <PctBar label="" pct={s.pct} color={ACCENT} />
                    <div className="text-muted" style={{ fontSize: 11 }}>{s.note}</div>
                  </div>
                </div>
              ))}
            </div>

            <div className="col-md-6">
              <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Systemic Features</h6>
              {(bd?.systemic_features || []).map((f, i) => (
                <div key={i} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className={f.pct === 0 ? 'text-success' : ''}>{f.feature}{f.pct === 0 ? ' ✓ ABSENT' : ''}</span>
                    <span className="text-muted">{f.pct}%</span>
                  </div>
                  {f.pct > 0 && <div className="progress mb-1" style={{ height: 8 }}>
                    <div className="progress-bar" style={{ width: `${f.pct}%`, backgroundColor: f.pct === 100 ? ACCENT : i % 2 === 0 ? ACCENT6 : ACCENT3 }} />
                  </div>}
                  <div className="text-muted" style={{ fontSize: 10 }}>{f.note}</div>
                </div>
              ))}
            </div>
          </div>

          <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Treatment Ladder</h6>
          {(bd?.treatments || []).map((t, i) => {
            const lvlColor = t.level === 'A' || t.level === 'A (EXTREME HAZARD)' ? '#2e7d32'
              : t.level === 'B' ? '#1565c0'
              : t.level === 'AVOID' ? '#e65100'
              : t.level === 'ABSOLUTE CONTRAINDICATION' ? '#b71c1c'
              : '#757575';
            return (
              <div key={i} className="card shadow-sm mb-2" style={{ borderLeft: `4px solid ${lvlColor}` }}>
                <div className="card-body py-2">
                  <div className="d-flex justify-content-between align-items-start">
                    <span className="fw-bold small">{t.therapy}</span>
                    <span className="badge" style={{ background: lvlColor }}>Level {t.level}</span>
                  </div>
                  <div className="text-muted small"><strong>Dose:</strong> {t.dose}</div>
                  <div className="text-muted small mt-1" style={{ fontSize: 11 }}>{t.rationale}</div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── TAB 3: DEFINITIONS ──────────────────────────────── */}
      {tab === 3 && (
        <div>
          <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>PC Deficiency — Expert Definitions</h6>
          {Object.entries(def || {}).map(([term, defn], i) => (
            <div key={i} className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${i % 3 === 0 ? ACCENT : i % 3 === 1 ? ACCENT4 : ACCENT5}` }}>
              <div className="card-body py-2">
                <div className="fw-bold small mb-1" style={{ color: i % 3 === 0 ? ACCENT : i % 3 === 1 ? ACCENT4 : ACCENT5 }}>{term}</div>
                <div className="text-muted small" style={{ whiteSpace: 'pre-wrap' }}>{defn}</div>
              </div>
            </div>
          ))}

          {/* Back nav */}
          <div className="mt-4 d-flex gap-2">
            <Link href="/btd" className="btn btn-sm btn-outline-secondary">← BTD (Biotinidase Deficiency)</Link>
            <Link href="/pcca" className="btn btn-sm btn-outline-secondary">PCCA (Propionic Acidemia A) →</Link>
          </div>
        </div>
      )}
    </div>
  );
}

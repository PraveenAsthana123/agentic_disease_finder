'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — isolated HHcy / MTR-MTRR axis
const ACCENT2 = '#880e4f';   // deep pink — HHcy elevation / methylation block
const ACCENT3 = '#4a148c';   // deep purple — megaloblastic anemia / folate trap
const ACCENT4 = '#1565c0';   // blue — treatment / Level A
const ACCENT5 = '#37474f';   // slate — KEY NEGATIVES / MMA normal
const ACCENT6 = '#1b5e20';   // dark green — MeCbl pathway / folate cycle
const ACCENT7 = '#e65100';   // deep orange — neonatal severe
const ACCENT8 = '#006064';   // teal — AdoCbl normal / intact MMUT arm

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

// ── Overview Tab ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  const k = data.kpis || {};
  const pd = data.phenotype_distribution || {};

  return (
    <div>
      {/* Gene banner */}
      <div className="alert alert-primary mb-3" style={{ background: ACCENT, color: '#fff', border: 'none' }}>
        <strong>{data.gene}</strong> — {data.full_name}<br />
        <small>{data.chromosome} · {data.inheritance} · {data.omim_gene} / {data.omim_disease}</small>
      </div>

      {/* Critical alerts */}
      <Alert variant="danger"
        text="⚠ N2O ABSOLUTE CI — MTR already compromised in cblE; N2O permanently inactivates residual MTR → LIFE-THREATENING acute homocysteine crisis." />
      <Alert variant="success"
        text="✅ MMA NORMAL in cblE — KEY NEGATIVE vs cblC/cblD/cblX. C3 propionylcarnitine NORMAL → NBS does NOT detect cblE. Folinic acid Level A (methylfolate trap)." />
      <Alert variant="info"
        text="🔬 AdoCbl NORMAL in fibroblasts — KEY POSITIVE: MMUT/MMAB arm intact. MeCbl ABSENT (MTR inactive without MTRR reactivation). Isolated HHcy = cblE or cblG." />

      {/* KPIs */}
      <div className="row g-2 mb-3">
        <KPI label="Avg tHcy (µmol/L)" value={k.avg_homocysteine_umol_l} color={ACCENT2} />
        <KPI label="Avg Methionine (µmol/L)" value={k.avg_methionine_umol_l} color={ACCENT} />
        <KPI label="Serum Folate (nmol/L)" value={k.avg_serum_folate_nmol_l} color={ACCENT3} />
        <KPI label="% Seizures" value={`${k.pct_seizures}%`} color={ACCENT7} />
        <KPI label="% Megaloblastic Anemia" value={`${k.pct_megaloblastic_anemia}%`} color={ACCENT3} />
        <KPI label="% NBS Detected" value={`${k.pct_nbs_detected}%`} color={ACCENT5} />
        <KPI label="Cohort N" value={data.cohort_n} color={ACCENT4} />
        <KPI label="OHCbl Response" value={`${k.pct_ohcbl_response}%`} color={ACCENT4} />
        <KPI label="Avg Hgb (g/dL)" value={k.avg_hgb_gdl} color={ACCENT2} />
      </div>

      {/* Gene function */}
      <div className="card mb-3 shadow-sm">
        <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>
          MTRR Function — The Methionine Synthase Reductase (cblE)
        </div>
        <div className="card-body">
          <p className="mb-1 small">{data.function}</p>
          <hr />
          <p className="mb-0 small text-muted">{data.mechanism}</p>
        </div>
      </div>

      {/* Key negative */}
      <div className="card mb-3 shadow-sm border-success">
        <div className="card-header fw-bold text-success">KEY NEGATIVES / POSITIVES — cblE vs Combined Disorders</div>
        <div className="card-body small">{data.key_negative}</div>
      </div>

      {/* Phenotype distribution */}
      <div className="card mb-3 shadow-sm">
        <div className="card-header fw-bold">Phenotype Distribution (N={data.cohort_n})</div>
        <div className="card-body">
          {pd.neonatal_severe && (
            <PctBar label={`Neonatal Severe (n=${pd.neonatal_severe.n})`}
              pct={pd.neonatal_severe.pct} color={ACCENT7} />
          )}
          {pd.infantile_classic && (
            <PctBar label={`Infantile Classic — MODAL (n=${pd.infantile_classic.n})`}
              pct={pd.infantile_classic.pct} color={ACCENT2} />
          )}
          {pd.late_onset_attenuated && (
            <PctBar label={`Late-Onset Attenuated (n=${pd.late_onset_attenuated.n})`}
              pct={pd.late_onset_attenuated.pct} color={ACCENT4} />
          )}
        </div>
      </div>

      {/* NBS note */}
      <div className="card mb-3 shadow-sm border-warning">
        <div className="card-header fw-bold text-warning">NBS Status — cblE is INVISIBLE to Standard NBS</div>
        <div className="card-body small">
          <strong>Primary NBS marker (C3):</strong> {data.nbs_primary}<br />
          <strong>Secondary confirmatory:</strong> {data.nbs_secondary}
        </div>
      </div>

      {/* Prevalence */}
      <div className="alert alert-secondary small">
        <strong>Prevalence:</strong> {data.prevalence}
      </div>
    </div>
  );
}

// ── Patients & Biomarkers Tab ─────────────────────────────────────────────────
function BiomarkersTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;

  const cblE_vs_cblC = [
    { feature: 'Urine MMA', cblE: 'NORMAL (< 5 mmol/mol Cr)', cblC: 'ELEVATED 200-2000 mmol/mol Cr', verdict: 'KEY NEG vs cblC' },
    { feature: 'tHcy', cblE: 'ELEVATED 40-200 µmol/L', cblC: 'ELEVATED 30-300 µmol/L', verdict: 'Both elevated' },
    { feature: 'Methionine', cblE: 'LOW 5-20 µmol/L', cblC: 'LOW 8-18 µmol/L', verdict: 'Both low' },
    { feature: 'MeCbl fibroblasts', cblE: 'ABSENT', cblC: 'ABSENT', verdict: 'Same' },
    { feature: 'AdoCbl fibroblasts', cblE: 'NORMAL (KEY POSITIVE)', cblC: 'ABSENT', verdict: 'KEY DIFF vs cblC' },
    { feature: 'C3 (NBS)', cblE: 'NORMAL — NBS INVISIBLE', cblC: 'ELEVATED — NBS detects', verdict: 'KEY NEG vs cblC' },
    { feature: 'Megaloblastic anemia', cblE: '~90% (methylfolate trap)', cblC: '~60-70% (combined block)', verdict: 'MORE in cblE' },
    { feature: 'Serum folate', cblE: 'HIGH (methylfolate trap)', cblC: 'Normal-high (less prominent)', verdict: 'cblE more pronounced' },
    { feature: 'Maculopathy', cblE: 'ABSENT', cblC: '~80% (pathognomonic)', verdict: 'KEY NEG vs cblC' },
    { feature: 'Protein restriction', cblE: 'NOT NEEDED', cblC: 'Level A', verdict: 'KEY DIFF: MMA normal' },
    { feature: 'Folinic acid', cblE: 'Level A (trap is primary)', cblC: 'Level B (secondary)', verdict: 'Higher priority in cblE' },
    { feature: 'Inheritance', cblE: 'AR (5p15.31)', cblC: 'AR (1p34.1)', verdict: 'Both AR' },
  ];

  return (
    <div>
      <Alert variant="info"
        text="cblE (MTRR) has ISOLATED HHcy — MMA is NORMAL. This distinguishes cblE/cblG from ALL combined MMA+HHcy disorders (cblC, cblD-combined, cblF, cblJ, cblX)." />

      {/* cblE vs cblC comparison */}
      <div className="card mb-3 shadow-sm">
        <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>
          cblE (MTRR) vs cblC (MMACHC) — 12 Distinguishing Features
        </div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-striped mb-0" style={{ fontSize: 12 }}>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th style={{ color: ACCENT }}>cblE / MTRR</th>
                  <th style={{ color: ACCENT3 }}>cblC / MMACHC</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {cblE_vs_cblC.map((r, i) => (
                  <tr key={i}>
                    <td className="fw-bold">{r.feature}</td>
                    <td>{r.cblE}</td>
                    <td>{r.cblC}</td>
                    <td><span className="badge bg-secondary">{r.verdict}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Patient sample */}
      {data.patient_sample && (
        <div className="card mb-3 shadow-sm">
          <div className="card-header fw-bold">Patient Sample (6 representative cases)</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-striped mb-0" style={{ fontSize: 11 }}>
                <thead>
                  <tr>
                    <th>ID</th><th>Sex</th><th>Phenotype</th><th>Genotype</th>
                    <th>Onset (mo)</th><th>tHcy</th><th>Met</th><th>Folate</th>
                    <th>Hgb</th><th>Sz</th><th>NBS</th><th>OHCbl</th>
                  </tr>
                </thead>
                <tbody>
                  {data.patient_sample.map((p, i) => (
                    <tr key={i}>
                      <td>{p.id}</td>
                      <td>{p.sex}</td>
                      <td>{p.phenotype}</td>
                      <td style={{ fontSize: 10, maxWidth: 180 }}>{p.genotype}</td>
                      <td>{p.onset_mo}</td>
                      <td className="fw-bold" style={{ color: ACCENT2 }}>{p.hcy}</td>
                      <td className="fw-bold" style={{ color: ACCENT }}>{p.met}</td>
                      <td>{p.folate}</td>
                      <td style={{ color: p.hgb < 9 ? ACCENT7 : 'inherit' }}>{p.hgb}</td>
                      <td>{p.sz ? '✅' : '—'}</td>
                      <td>{p.nbs ? '✅' : '❌'}</td>
                      <td>{p.ohcbl_resp ? '✅' : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Pathway diagram */}
      <div className="card mb-3 shadow-sm border-info">
        <div className="card-header fw-bold text-info">MTRR Position in Cobalamin Pathway — Step 4b Reactivation</div>
        <div className="card-body small">
          <div className="d-flex flex-wrap gap-2 align-items-center">
            {[
              { label: 'TC2-CD320 Endocytosis', note: 'Step 0', color: '#78909c' },
              { label: 'LMBRD1+ABCD4 Lysosomal Export', note: 'Step 1', color: '#5c6bc0' },
              { label: 'MMACHC Processing → cob(I)alamin', note: 'Step 2', color: '#7b1fa2' },
              { label: 'MMADHC Distribution', note: 'Step 3', color: '#1565c0' },
              { label: '→ MMAB → AdoCbl → MMUT (MMA)', note: 'Arm A NORMAL ✅', color: ACCENT8 },
              { label: '→ MTR → MeCbl → Hcy→Met', note: 'Arm B BLOCKED ❌', color: ACCENT2 },
              { label: 'MTRR reactivates MTR (cob(II)→cob(I))', note: 'MTRR MISSING ❌', color: ACCENT7 },
            ].map((s, i) => (
              <div key={i} className="badge px-2 py-2" style={{ background: s.color, fontSize: 11, whiteSpace: 'normal', maxWidth: 200 }}>
                {s.label}<br /><em>{s.note}</em>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Seizures & Triggers Tab ───────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;

  return (
    <div>
      <Alert variant="danger"
        text="⚠ N2O ABSOLUTE CI — permanently inactivates MTR by oxidizing cob(I)alamin to cob(III)alamin. MTR is already compromised in cblE → complete MTR failure → acute severe HHcy crisis." />

      {/* Seizure types */}
      <div className="card mb-3 shadow-sm">
        <div className="card-header fw-bold">Seizure Type Distribution</div>
        <div className="card-body">
          {(data.seizure_type_distribution || []).map((s, i) => (
            <PctBar key={i} label={s.type} pct={s.pct} color={i === 0 ? ACCENT2 : i === 1 ? ACCENT7 : ACCENT} />
          ))}
        </div>
      </div>

      {/* Metabolic triggers */}
      <div className="card mb-3 shadow-sm">
        <div className="card-header fw-bold">Metabolic Triggers</div>
        <div className="card-body">
          {(data.metabolic_triggers || []).map((t, i) => (
            <div key={i} className="mb-2 pb-2 border-bottom">
              <div className="d-flex justify-content-between mb-1">
                <span className="fw-bold small">{t.trigger}</span>
                <span className="badge" style={{ background: t.pct === 100 ? '#b71c1c' : ACCENT }}>{t.pct}%</span>
              </div>
              <div className="text-muted" style={{ fontSize: 12 }}>{t.mechanism}</div>
            </div>
          ))}
        </div>
      </div>

      {/* High-risk drugs */}
      <div className="card mb-3 shadow-sm">
        <div className="card-header fw-bold text-danger">High-Risk Drugs in cblE</div>
        <div className="card-body">
          {(data.high_risk_drugs || []).map((d, i) => (
            <div key={i} className={`alert ${d.risk === 'ABSOLUTE CI' ? 'alert-danger' : 'alert-warning'} py-2 mb-2`}>
              <strong>{d.drug}</strong> — <span className="badge bg-danger">{d.risk}</span><br />
              <small>{d.mechanism}</small>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Treatments Tab ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;

  const evidenceColor = (ev) => {
    if (ev === 'Level A') return '#1565c0';
    if (ev === 'Level B') return '#6a1b9a';
    if (ev === 'AVOID') return '#b71c1c';
    return ACCENT5;
  };

  return (
    <div>
      <Alert variant="success"
        text="✅ Protein restriction NOT needed in cblE — MMA arm is intact (MMUT/AdoCbl unaffected). This is a key treatment difference vs combined MMA+HHcy disorders (cblC, cblX, cblF, cblJ)." />
      <Alert variant="info"
        text="🔬 Folinic acid is Level A in cblE (not just B) — methylfolate trap is the PRIMARY mechanism of megaloblastic anemia in cblE; folinic acid directly bypasses the 5-methylTHF block." />

      <div className="card shadow-sm">
        <div className="card-header fw-bold" style={{ background: ACCENT4, color: '#fff' }}>
          Treatment Ladder — MTRR (cblE) Deficiency
        </div>
        <div className="card-body">
          {(data.treatments || []).map((t, i) => (
            <div key={i} className="mb-3 pb-3 border-bottom">
              <div className="d-flex justify-content-between align-items-center mb-1">
                <span className="fw-bold">{t.treatment}</span>
                <div>
                  <span className="badge me-1" style={{ background: evidenceColor(t.evidence) }}>{t.evidence}</span>
                  {t.response_pct > 0 && (
                    <span className="badge bg-secondary">{t.response_pct}% response</span>
                  )}
                </div>
              </div>
              {t.response_pct > 0 && (
                <div className="progress mb-1" style={{ height: 6 }}>
                  <div className="progress-bar" style={{ width: `${t.response_pct}%`, backgroundColor: evidenceColor(t.evidence) }} />
                </div>
              )}
              <div className="text-muted" style={{ fontSize: 12 }}>{t.note}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  const gc = data.gene_card || {};

  return (
    <div>
      {/* Gene card */}
      <div className="card mb-3 shadow-sm">
        <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>Gene Card — MTRR (cblE)</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0" style={{ fontSize: 12 }}>
            <tbody>
              {Object.entries(gc).map(([k, v]) => (
                <tr key={k}>
                  <td className="fw-bold text-muted" style={{ width: '30%' }}>{k}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Key concepts */}
      <div className="card mb-3 shadow-sm">
        <div className="card-header fw-bold">Key Concepts ({(data.key_concepts || []).length})</div>
        <div className="card-body">
          {(data.key_concepts || []).map((c, i) => (
            <div key={i} className="mb-3 pb-3 border-bottom">
              <div className="fw-bold mb-1" style={{ color: ACCENT, fontSize: 13 }}>{c.concept}</div>
              <div className="text-muted" style={{ fontSize: 12 }}>{c.explanation}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Diagnostic thresholds */}
      <div className="card mb-3 shadow-sm">
        <div className="card-header fw-bold">Diagnostic Thresholds & Action Points</div>
        <div className="card-body p-0">
          <table className="table table-sm table-striped mb-0" style={{ fontSize: 12 }}>
            <thead>
              <tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr>
            </thead>
            <tbody>
              {(data.diagnostic_thresholds || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.parameter}</td>
                  <td><span className="badge" style={{ background: ACCENT2 }}>{t.threshold}</span></td>
                  <td style={{ fontSize: 11 }}>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Differential diagnosis */}
      <div className="card mb-3 shadow-sm">
        <div className="card-header fw-bold">Differential Diagnosis</div>
        <div className="card-body">
          {(data.differential_diagnosis || []).map((d, i) => (
            <div key={i} className="mb-2 pb-2 border-bottom">
              <span className="fw-bold" style={{ color: ACCENT3 }}>{d.disease}</span><br />
              <span className="text-muted" style={{ fontSize: 12 }}>{d.distinguishing}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function MTRRPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mtrr/overview`).then(r => r.json()),
      fetch(`${API}/api/mtrr/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mtrr/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
        🧬 MTRR Epilepsy (Homocystinuria-Megaloblastic Anemia cblE type)
      </h4>
      <p className="text-muted small mb-3">
        Methionine Synthase Reductase Deficiency · 5p15.31 · AR · OMIM *602568 / #236270 ·
        Isolated HHcy (MMA NORMAL) · NBS INVISIBLE · &lt;200 cases worldwide 2026
      </p>

      {err && <div className="alert alert-danger">{err}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active' : ''}`}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <BiomarkersTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}

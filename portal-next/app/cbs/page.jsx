'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — CBS / transsulfuration
const ACCENT2 = '#b71c1c';   // deep red — very high HHcy / thrombosis
const ACCENT3 = '#4a148c';   // deep purple — B6 responsiveness
const ACCENT4 = '#1565c0';   // blue — Level A treatment
const ACCENT5 = '#37474f';   // slate — KEY NEGATIVES / normal values
const ACCENT6 = '#1b5e20';   // dark green — MeCbl NORMAL / cobalamin intact
const ACCENT7 = '#e65100';   // deep orange — ectopia lentis / thromboembolism
const ACCENT8 = '#006064';   // teal — methionine restriction / cysteine

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

function SectionCard({ title, children, color = ACCENT }) {
  return (
    <div className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-header fw-bold" style={{ background: '#eef2ff', color }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
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
      <div className="alert mb-3" style={{ background: ACCENT, color: '#fff', border: 'none' }}>
        <strong>{data.gene}</strong> — {data.full_name}<br />
        <small>{data.chromosome} · {data.inheritance} · {data.omim_gene} / {data.omim_disease}</small>
      </div>

      <Alert variant="danger"
        text="⚠ Methionine ELEVATED (HIGH) — KEY MAJOR DIFFERENCE from ALL remethylation defects (cblE/cblG/MTHFR where methionine is LOW). HIGH methionine + VERY HIGH Hcy = CBS until proven otherwise. This single lab finding narrows the differential immediately." />
      <Alert variant="warning"
        text="🔴 Ectopia lentis (lens dislocation) — PATHOGNOMONIC for CBS. Present in 90% untreated. INFERIOR dislocation (vs superior in Marfan). ABSENT in cblE/cblG/MTHFR. Ophthalmology slit-lamp exam is mandatory in all HHcy patients." />
      <Alert variant="info"
        text="💊 B6 (pyridoxine) responsiveness in 50% — UNIQUE to CBS among all HHcy disorders. Test all CBS patients with 100–500 mg B6/day × 3–4 weeks before finalizing treatment. B6-responsive patients achieve near-normal tHcy on B6 + betaine." />
      <Alert variant="success"
        text="✅ MeCbl NORMAL, MMA NORMAL, Serum Folate NORMAL — cobalamin system (MTR/MTRR/LMBRD1/ABCD4/MMACHC) fully intact. No methylfolate trap. Distinguished from cblE/cblG (MeCbl absent, folate high) and MTHFR (folate low/normal, no ectopia)." />

      <div className="row g-2 mb-3">
        <KPI label="Avg tHcy (µmol/L)" value={k.avg_homocysteine_umol_l} color={ACCENT2} />
        <KPI label="Avg Met (µmol/L) ↑HIGH" value={k.avg_methionine_umol_l} color={ACCENT7} />
        <KPI label="Avg Cysteine (µmol/L) ↓LOW" value={k.avg_cysteine_umol_l} color={ACCENT8} />
        <KPI label="% Ectopia Lentis" value={`${k.pct_ectopia_lentis}%`} color={ACCENT7} />
        <KPI label="% Thromboembolism" value={`${k.pct_thromboembolism}%`} color={ACCENT2} />
        <KPI label="% B6-Responsive" value={`${k.pct_b6_responsive}%`} color={ACCENT3} />
        <KPI label="% NBS Detected" value={`${k.pct_nbs_detected}%`} color={ACCENT4} />
        <KPI label="% MeCbl Normal" value={`${k.pct_mecbl_normal}%`} color={ACCENT6} />
        <KPI label="% IDD" value={`${k.pct_idd}%`} color={ACCENT5} />
      </div>

      <SectionCard title="CBS Function — Cystathionine Beta-Synthase (21q22.3)">
        <p className="mb-1 small">{data.function}</p>
        <hr />
        <p className="mb-0 small text-muted">{data.mechanism}</p>
      </SectionCard>

      <div className="card mb-3 shadow-sm border-danger">
        <div className="card-header fw-bold text-danger">KEY POSITIVES / KEY NEGATIVES — CBS vs cblE/cblG/MTHFR</div>
        <div className="card-body small">{data.key_negative}</div>
      </div>

      {/* 10-feature comparison table: CBS vs cblE vs cblG vs MTHFR */}
      <SectionCard title="Transsulfuration vs Remethylation: CBS vs cblE (MTRR) vs cblG (MTR) vs MTHFR — 10-Feature Comparison" color={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-bordered table-sm small mb-0">
            <thead style={{ background: ACCENT2, color: '#fff' }}>
              <tr>
                <th>Feature</th>
                <th>CBS (Classical HHcy)</th>
                <th>cblE (MTRR)</th>
                <th>cblG (MTR)</th>
                <th>MTHFR</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>Gene / Protein</td><td>CBS — Hcy+Ser→Cystathionine (PLP)</td><td>MTRR — MTR reductase (FMN/FAD)</td><td>MTR — Methionine synthase (MeCbl)</td><td>MTHFR — 5,10-methyleneTHF reductase (FAD)</td></tr>
              <tr><td>tHcy (µmol/L)</td><td className="text-danger fw-bold">↑↑↑ 100–500 (HIGHEST)</td><td className="text-danger">↑ 40–200</td><td className="text-danger">↑ 40–200</td><td className="text-danger">↑↑ 50–300</td></tr>
              <tr><td>Methionine</td><td className="text-danger fw-bold">HIGH ↑ 60–500 ← KEY +VE</td><td className="text-success">LOW &lt;15</td><td className="text-success">LOW &lt;15</td><td className="text-success">LOW 5–15</td></tr>
              <tr><td>MMA</td><td className="text-success fw-bold">NORMAL (KEY NEG)</td><td className="text-success">NORMAL</td><td className="text-success">NORMAL</td><td className="text-success">NORMAL</td></tr>
              <tr><td>MeCbl (fibroblasts)</td><td className="text-success fw-bold">NORMAL ← KEY</td><td className="text-danger">ABSENT</td><td className="text-danger">ABSENT</td><td className="text-success">NORMAL</td></tr>
              <tr><td>Serum Folate</td><td className="fw-bold">NORMAL ← KEY DISTINCTION</td><td className="text-warning">HIGH (trap)</td><td className="text-warning">HIGH (trap)</td><td>LOW/NORMAL</td></tr>
              <tr><td>Ectopia Lentis</td><td className="text-danger fw-bold">~90% ← PATHOGNOMONIC</td><td className="text-success">ABSENT</td><td className="text-success">ABSENT</td><td className="text-success">ABSENT</td></tr>
              <tr><td>NBS Detection</td><td className="fw-bold">~60% (Met↑)</td><td>INVISIBLE</td><td>INVISIBLE</td><td>INVISIBLE</td></tr>
              <tr><td>B6 Response</td><td className="text-primary fw-bold">50% RESPONSIVE ← UNIQUE</td><td>None</td><td>None</td><td>None</td></tr>
              <tr><td>Methionine Rx</td><td className="fw-bold">RESTRICT (Level A) ← KEY</td><td>Not needed</td><td>Not needed</td><td>SUPPLEMENT (Level A)</td></tr>
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title={`Phenotype Distribution (N=${data.cohort_n})`} color={ACCENT7}>
        {pd.b6_responsive_classical && (
          <PctBar label={`B6-Responsive Classical (n=${pd.b6_responsive_classical.n}) — p.Ile278Thr / p.Glu302Lys; HHcy normalizes on B6`}
            pct={pd.b6_responsive_classical.pct} color={ACCENT3} />
        )}
        {pd.b6_nr_severe && (
          <PctBar label={`B6-Non-Responsive Severe (n=${pd.b6_nr_severe.n}) — p.Gly307Ser / p.Arg369Cys; betaine + Met-restriction required`}
            pct={pd.b6_nr_severe.pct} color={ACCENT2} />
        )}
        {pd.mild_attenuated && (
          <PctBar label={`Mild / Attenuated (n=${pd.mild_attenuated.n}) — p.Asp444Asn; adult-onset mild HHcy; sometimes incidental`}
            pct={pd.mild_attenuated.pct} color={ACCENT4} />
        )}
      </SectionCard>

      <div className="row g-3">
        <div className="col-md-6">
          <SectionCard title="NBS / Diagnosis" color={ACCENT5}>
            <p className="small mb-1"><strong>Primary NBS:</strong> {data.nbs_primary}</p>
            <p className="small mb-0"><strong>Secondary:</strong> {data.nbs_secondary}</p>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Protein Size / Locus" color={ACCENT8}>
            <p className="small mb-1"><strong>Size:</strong> {data.protein_size}</p>
            <p className="small mb-0"><strong>Prevalence:</strong> {data.prevalence}</p>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}

// ── Patients & Biomarkers Tab ─────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  const pts = data.patient_sample || [];
  const bm = data.biomarker_ranges || {};
  const vv = data.variant_breakdown || [];

  const phenoLabel = {
    b6_responsive_classical: 'B6-Responsive',
    b6_nr_severe: 'B6-NR Severe',
    mild_attenuated: 'Mild/Attenuated',
  };
  const phenoColor = {
    b6_responsive_classical: ACCENT3,
    b6_nr_severe: ACCENT2,
    mild_attenuated: ACCENT4,
  };

  return (
    <div>
      <SectionCard title="Biomarker Ranges by Phenotype" color={ACCENT2}>
        {Object.entries(bm).map(([key, val]) => (
          <div key={key} className="mb-2">
            <strong className="small text-uppercase">{key.replace(/_/g, ' ')}: </strong>
            {typeof val === 'object'
              ? <ul className="mb-1 small">
                  {Object.entries(val).map(([k2, v2]) => (
                    <li key={k2}><em>{k2.replace(/_/g, ' ')}:</em> {v2}</li>
                  ))}
                </ul>
              : <span className="small">{val}</span>}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Variant Breakdown" color={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-bordered table-sm small mb-0">
            <thead style={{ background: ACCENT3, color: '#fff' }}>
              <tr>
                <th>Variant</th>
                <th>B6 Status</th>
                <th>Prevalence</th>
                <th>Severity</th>
              </tr>
            </thead>
            <tbody>
              {vv.map((v, i) => (
                <tr key={i}>
                  <td className="fw-bold">{v.variant}</td>
                  <td style={{ color: v.b6.includes('RESPONSIVE') && !v.b6.includes('NON') ? ACCENT3 : ACCENT2 }}>
                    {v.b6}
                  </td>
                  <td>{v.prevalence}</td>
                  <td>{v.severity}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Patient Sample (first 12)" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-bordered table-sm small mb-0">
            <thead style={{ background: ACCENT, color: '#fff' }}>
              <tr>
                <th>ID</th><th>Sex</th><th>Phenotype</th><th>Genotype</th>
                <th>tHcy (µmol/L)</th><th>Met (µmol/L)</th><th>Cys (µmol/L)</th>
                <th>MMA</th><th>MeCbl</th><th>NBS</th><th>Ectopia</th><th>Thrombo</th><th>B6-Resp</th>
              </tr>
            </thead>
            <tbody>
              {pts.map(p => (
                <tr key={p.patient_id}>
                  <td>{p.patient_id}</td>
                  <td>{p.sex}</td>
                  <td style={{ color: phenoColor[p.phenotype] || ACCENT, fontWeight: 600 }}>
                    {phenoLabel[p.phenotype] || p.phenotype}
                  </td>
                  <td className="small text-muted">{p.genotype}</td>
                  <td className="text-danger fw-bold">{p.homocysteine_umol_l}</td>
                  <td className="text-danger fw-bold">{p.methionine_umol_l}</td>
                  <td style={{ color: ACCENT8 }}>{p.cysteine_umol_l}</td>
                  <td className="text-success">NORMAL</td>
                  <td className="text-success">NORMAL</td>
                  <td>{p.nbs_detected ? '✅' : '❌'}</td>
                  <td>{p.ectopia_lentis ? '👁 Yes' : 'No'}</td>
                  <td>{p.thromboembolism ? '🔴 Yes' : 'No'}</td>
                  <td style={{ color: p.b6_responsive ? ACCENT3 : ACCENT5 }}>
                    {p.b6_responsive ? 'Yes' : 'No'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Seizures & Triggers Tab ───────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  const st = data.seizure_types || [];
  const mt = data.metabolic_triggers || [];
  const kp = data.kpi_pcts || {};

  return (
    <div>
      <Alert variant="info"
        text="ℹ CBS epilepsy is SECONDARY — seizures occur in 20–30% of patients, typically from cerebrovascular events (HHcy-induced thrombosis) or direct HHcy neuronal excitotoxicity. Less seizure-prominent than cblE/cblG/MTHFR where metabolic brain injury is the primary mechanism." />

      <div className="row g-2 mb-3">
        <KPI label="% Seizures" value={`${kp.seizures || '--'}%`} color={ACCENT2} />
        <KPI label="% Ectopia Lentis" value={`${kp.ectopia_lentis || '--'}%`} color={ACCENT7} />
        <KPI label="% Thromboembolism" value={`${kp.thromboembolism || '--'}%`} color={ACCENT2} />
        <KPI label="% IDD" value={`${kp.idd || '--'}%`} color={ACCENT5} />
        <KPI label="% Psychiatric" value={`${kp.psychiatric || '--'}%`} color={ACCENT3} />
        <KPI label="% Osteoporosis" value={`${kp.osteoporosis || '--'}%`} color={ACCENT5} />
      </div>

      <SectionCard title="Seizure Types (when present)" color={ACCENT2}>
        {st.map((s, i) => (
          <PctBar key={i} label={s.type} pct={s.pct} color={ACCENT2} />
        ))}
      </SectionCard>

      <SectionCard title="Metabolic Triggers & High-Risk Exposures" color={ACCENT7}>
        {mt.map((t, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <strong className="small">{t.trigger}</strong>
              <span className="badge" style={{ background: ACCENT7, fontSize: 11 }}>{t.pct}%</span>
            </div>
            <p className="small text-muted mb-0">{t.mechanism}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Features Beyond Seizures" color={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-bordered table-sm small mb-0">
            <thead style={{ background: ACCENT3, color: '#fff' }}>
              <tr><th>Feature</th><th>Prevalence</th><th>Notes</th></tr>
            </thead>
            <tbody>
              <tr><td>Ectopia lentis (lens dislocation)</td><td className="text-danger fw-bold">~90% untreated</td><td>PATHOGNOMONIC; inferior in CBS (vs superior in Marfan); onset earlier in B6-NR severe</td></tr>
              <tr><td>Marfanoid habitus</td><td className="text-danger fw-bold">~80%</td><td>Tall, thin, long limbs, arachnodactyly, pectus; HHcy disrupts fibrillin-1</td></tr>
              <tr><td>Thromboembolism</td><td className="text-danger fw-bold">50–60% by age 30 untreated</td><td>DVT, PE, cerebral venous thrombosis, arterial stroke; HIGHEST of all HHcy disorders</td></tr>
              <tr><td>Intellectual disability</td><td>60–65%</td><td>More severe in B6-NR; milder in B6-responsive with early treatment</td></tr>
              <tr><td>Psychiatric (schizophrenia-like)</td><td>50–60%</td><td>HHcy → NMDA receptor antagonism; adult onset; often misdiagnosed</td></tr>
              <tr><td>Osteoporosis</td><td>~50%</td><td>HHcy inhibits lysyl oxidase → impaired collagen cross-linking; glutathione deficiency</td></tr>
              <tr><td>Scoliosis / pectus excavatum</td><td>30–40%</td><td>Connective tissue; fibrillin + collagen cross-linking impaired</td></tr>
              <tr><td>Epilepsy / seizures</td><td>20–30%</td><td>Secondary (thrombotic stroke + HHcy excitotoxicity); LESS prominent than in cobalamin disorders</td></tr>
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Treatments Tab ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  const tx = data.treatments || [];
  const dr = data.drug_risks || [];

  const levelColor = (level) => {
    if (level.includes('Level A')) return ACCENT4;
    if (level.includes('Level B')) return ACCENT3;
    return ACCENT5;
  };

  return (
    <div>
      <Alert variant="danger"
        text="🚨 ALWAYS test B6 (pyridoxine) responsiveness FIRST in every CBS patient — 100–500 mg B6/day × 3–4 weeks. 50% of patients normalize tHcy on B6 alone (+ betaine). B6-non-responsive patients proceed to betaine + strict methionine restriction. Never assume non-responsiveness without a proper trial." />

      <SectionCard title="Treatments — CBS Classical Homocystinuria" color={ACCENT4}>
        {tx.map((t, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="d-flex align-items-start justify-content-between mb-1">
              <strong className="small">{t.treatment}</strong>
              <span className="badge ms-2" style={{ background: levelColor(t.level), fontSize: 10, whiteSpace: 'nowrap' }}>
                {t.level}
              </span>
            </div>
            <p className="small text-muted mb-0">{t.mechanism}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Drug / Exposure Risks" color={ACCENT2}>
        {dr.map((d, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="d-flex align-items-start justify-content-between mb-1">
              <strong className="small">{d.agent}</strong>
              <span className="badge ms-2" style={{
                background: d.risk.includes('ABSOLUTE') ? '#b71c1c' : d.risk.includes('HIGH') ? ACCENT7 : ACCENT3,
                fontSize: 10, whiteSpace: 'normal', maxWidth: 180, textAlign: 'right'
              }}>
                {d.risk}
              </span>
            </div>
            <p className="small text-muted mb-0">{d.mechanism}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="N2O Risk — CBS vs cblE/cblG vs MTHFR (Critical Comparison)" color={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-bordered table-sm small mb-0">
            <thead style={{ background: ACCENT3, color: '#fff' }}>
              <tr>
                <th>Feature</th>
                <th>CBS</th>
                <th>cblE / cblG</th>
                <th>MTHFR</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>N2O Risk</td>
                <td><strong>MODERATE</strong> — caution, not absolute CI</td>
                <td className="text-danger fw-bold">ABSOLUTE CI — MTR already compromised</td>
                <td><strong>HIGH RISK</strong> — not absolute CI</td>
              </tr>
              <tr>
                <td>Reason</td>
                <td>MTR/cobalamin system intact; but very high HHcy sensitizes</td>
                <td>N2O inactivates MTR cobalamin → complete MTR failure → HHcy surge</td>
                <td>MTR intact but methionine/HHcy imbalance; more vulnerable than CBS</td>
              </tr>
              <tr>
                <td>Periop cover</td>
                <td>Supplement B12 + betaine; use total IV anesthesia if possible</td>
                <td>AVOID N2O absolutely — use propofol-based TIVA</td>
                <td>Use TIVA; supplement B2 + betaine perioperatively</td>
              </tr>
              <tr>
                <td>Estrogen / OCP</td>
                <td className="text-danger fw-bold">ABSOLUTE CI — thrombosis</td>
                <td>High Risk (25% thrombosis)</td>
                <td>Moderate Risk</td>
              </tr>
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  const gc = data.gene_card || {};
  const kc = data.key_concepts || [];
  const dd = data.differential_diagnosis || [];

  return (
    <div>
      <SectionCard title="CBS Gene Card" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-bordered table-sm small mb-0">
            <tbody>
              {Object.entries(gc).map(([k, v]) => (
                <tr key={k}>
                  <td className="fw-bold text-muted" style={{ width: '30%' }}>{k.replace(/_/g, ' ').toUpperCase()}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Key Concepts — CBS / Classical Homocystinuria" color={ACCENT7}>
        {kc.map((c, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <strong className="small d-block mb-1" style={{ color: ACCENT7 }}>{c.concept}</strong>
            <p className="small text-muted mb-0">{c.explanation}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Differential Diagnosis" color={ACCENT3}>
        {dd.map((d, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <strong className="small d-block mb-1" style={{ color: ACCENT3 }}>vs {d.disease}</strong>
            <p className="small text-muted mb-0">{d.distinguishing}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function CBSPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/cbs/overview`).then(r => r.json()),
      fetch(`${API}/api/cbs/breakdown`).then(r => r.json()),
      fetch(`${API}/api/cbs/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefinitions(df);
    }).catch(e => setErr(e.message));
  }, []);

  if (err) return (
    <div className="container py-4">
      <div className="alert alert-danger">API error: {err}</div>
    </div>
  );

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2">
        <Link href="/" className="btn btn-sm btn-outline-secondary">← Home</Link>
        <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
          🧬 CBS Epilepsy — Classical Homocystinuria (Cystathionine Beta-Synthase Deficiency)
        </h4>
      </div>

      <div className="alert alert-secondary py-2 small mb-3">
        <strong>Pathway:</strong> 21q22.3 · AR · PLP/B6-dependent · Transsulfuration gateway ·
        Hcy+Ser→Cystathionine → Cysteine · <span className="text-danger fw-bold">Methionine HIGH</span> ·
        tHcy 100–500 µmol/L (HIGHEST) · Ectopia lentis PATHOGNOMONIC · B6-responsive 50% ·
        <span className="text-success"> MMA NORMAL · MeCbl NORMAL</span> ·
        OMIM *613381 / #236200
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}

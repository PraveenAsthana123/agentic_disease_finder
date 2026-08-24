'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — MTHFR / remethylation
const ACCENT2 = '#880e4f';   // deep pink — HHcy elevation
const ACCENT3 = '#4a148c';   // deep purple — white matter disease
const ACCENT4 = '#1565c0';   // blue — Level A treatment
const ACCENT5 = '#37474f';   // slate — KEY NEGATIVES
const ACCENT6 = '#1b5e20';   // dark green — MeCbl NORMAL / cobalamin intact
const ACCENT7 = '#e65100';   // deep orange — neonatal severe
const ACCENT8 = '#006064';   // teal — riboflavin / B2 cofactor

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
        text="⚠ N2O HIGH RISK — MTR cobalamin starts intact in MTHFR; however severely reduced methionine + HHcy makes N2O-induced MTR inactivation DANGEROUS. Not absolute CI (unlike cblE/cblG where MTR is already compromised), but HIGH RISK — avoid in surgery." />
      <Alert variant="success"
        text="✅ MeCbl NORMAL in fibroblasts — KEY POSITIVE DISTINCTION from cblE/cblG. Cobalamin system (MTR, MTRR, LMBRD1, ABCD4, MMACHC) is INTACT. The problem is upstream: MTHFR cannot produce 5-methylTHF for MTR to use." />
      <Alert variant="info"
        text="🔬 Serum folate LOW/NORMAL — KEY DISTINCTION from cblE/cblG where serum folate is HIGH (methylfolate trap). In MTHFR: 5-methylTHF is not being MADE (not accumulating). No megaloblastic anemia in most patients (another KEY NEGATIVE vs cblE/cblG 80–90%)." />
      <Alert variant="warning"
        text="💊 Riboflavin B2 Level A — MTHFR is a FAD-dependent flavoprotein. B2 supplementation stabilizes MTHFR enzyme (especially thermolabile mutants) and is mandatory in ALL severe MTHFR deficiency, unlike other HHcy disorders." />

      <div className="row g-2 mb-3">
        <KPI label="Avg tHcy (µmol/L)" value={k.avg_homocysteine_umol_l} color={ACCENT2} />
        <KPI label="Avg Methionine (µmol/L)" value={k.avg_methionine_umol_l} color={ACCENT} />
        <KPI label="Serum Folate (nmol/L)" value={k.avg_serum_folate_nmol_l} color={ACCENT5} />
        <KPI label="% Seizures" value={`${k.pct_seizures}%`} color={ACCENT7} />
        <KPI label="% White Matter Disease" value={`${k.pct_white_matter_disease}%`} color={ACCENT3} />
        <KPI label="% NBS Detected" value={`${k.pct_nbs_detected}%`} color={ACCENT5} />
        <KPI label="Cohort N" value={data.cohort_n} color={ACCENT4} />
        <KPI label="% Megaloblastic Anemia" value={`${k.pct_megaloblastic_anemia}%`} color={ACCENT5} />
        <KPI label="% MeCbl Normal" value={`${k.pct_mecbl_normal}%`} color={ACCENT6} />
      </div>

      <SectionCard title="MTHFR Function — Methylenetetrahydrofolate Reductase (1p36.3)">
        <p className="mb-1 small">{data.function}</p>
        <hr />
        <p className="mb-0 small text-muted">{data.mechanism}</p>
      </SectionCard>

      <div className="card mb-3 shadow-sm border-success">
        <div className="card-header fw-bold text-success">KEY NEGATIVES / KEY POSITIVES — MTHFR vs cblE/cblG</div>
        <div className="card-body small">{data.key_negative}</div>
      </div>

      {/* 10-feature comparison table: MTHFR vs cblE vs cblG */}
      <SectionCard title="Remethylation Pathway: MTHFR vs cblE (MTRR) vs cblG (MTR) — 10-Feature Comparison" color={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-bordered table-sm small mb-0">
            <thead style={{ background: ACCENT3, color: '#fff' }}>
              <tr>
                <th>Feature</th>
                <th>MTHFR (Severe Deficiency)</th>
                <th>cblE (MTRR)</th>
                <th>cblG (MTR)</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>Gene / Protein</td><td>MTHFR — 5,10-methyleneTHF reductase (FAD)</td><td>MTRR — MTR reductase (diflavin FMN/FAD)</td><td>MTR — Methionine synthase (MeCbl)</td></tr>
              <tr><td>tHcy</td><td className="text-danger fw-bold">↑↑ 50–300 µmol/L</td><td className="text-danger">↑ 40–200 µmol/L</td><td className="text-danger">↑ 40–200 µmol/L</td></tr>
              <tr><td>MMA</td><td className="text-success fw-bold">NORMAL (KEY NEG)</td><td className="text-success">NORMAL (KEY NEG)</td><td className="text-success">NORMAL (KEY NEG)</td></tr>
              <tr><td>Serum Folate</td><td className="fw-bold">LOW/NORMAL ← KEY DISTINCTION</td><td className="text-warning">HIGH (methylfolate trap)</td><td className="text-warning">HIGH (methylfolate trap)</td></tr>
              <tr><td>MeCbl (fibroblasts)</td><td className="text-success fw-bold">NORMAL ← KEY DISTINCTION</td><td className="text-danger">ABSENT</td><td className="text-danger">ABSENT</td></tr>
              <tr><td>Megaloblastic Anemia</td><td className="text-success fw-bold">RARE ~10% ← KEY NEG</td><td className="text-danger">90%</td><td className="text-danger">80%</td></tr>
              <tr><td>White Matter Disease</td><td className="text-danger fw-bold">82% (PROMINENT)</td><td>20%</td><td>25%</td></tr>
              <tr><td>NBS Visibility</td><td>INVISIBLE (C3 normal)</td><td>INVISIBLE (C3 normal)</td><td>INVISIBLE (C3 normal)</td></tr>
              <tr><td>Folinic Acid</td><td>Level B (not Level A)</td><td className="fw-bold">Level A (methylfolate trap)</td><td className="fw-bold">Level A (methylfolate trap)</td></tr>
              <tr><td>Riboflavin B2</td><td className="text-primary fw-bold">Level A (FAD cofactor MTHFR)</td><td>Not indicated</td><td>Not indicated</td></tr>
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Phenotype Distribution (N={data.cohort_n})" color={ACCENT7}>
        {pd.neonatal_severe && (
          <PctBar label={`Neonatal Severe (n=${pd.neonatal_severe.n}) — null/null <1% MTHFR activity`}
            pct={pd.neonatal_severe.pct} color={ACCENT7} />
        )}
        {pd.infantile_classic && (
          <PctBar label={`Infantile Classic — MODAL (n=${pd.infantile_classic.n}) — 1–5% residual activity`}
            pct={pd.infantile_classic.pct} color={ACCENT2} />
        )}
        {pd.late_onset_attenuated && (
          <PctBar label={`Late-Onset Attenuated (n=${pd.late_onset_attenuated.n}) — 5–20% residual activity`}
            pct={pd.late_onset_attenuated.pct} color={ACCENT4} />
        )}
        {pd.adult_onset_psychiatric && (
          <PctBar label={`Adult-Onset Psychiatric (n=${pd.adult_onset_psychiatric.n}) — >20% residual`}
            pct={pd.adult_onset_psychiatric.pct} color={ACCENT5} />
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

  return (
    <div>
      <Alert variant="info"
        text="🔬 MeCbl NORMAL in ALL patients — cobalamin system intact. Serum folate LOW/NORMAL — no methylfolate trap (5-methylTHF not accumulating, not being made). These two tests immediately distinguish MTHFR from cblE/cblG." />

      <SectionCard title="Patient Sample (6 of 40)" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped small mb-0">
            <thead>
              <tr>
                <th>ID</th><th>Age (yr)</th><th>Phenotype</th>
                <th>tHcy (µmol/L)</th><th>Met (µmol/L)</th><th>Folate</th>
                <th>MeCbl</th><th>WMD</th><th>Seizures</th>
              </tr>
            </thead>
            <tbody>
              {pts.map(p => (
                <tr key={p.patient_id}>
                  <td>{p.patient_id}</td>
                  <td>{p.age_at_diagnosis_years}</td>
                  <td>{p.phenotype}</td>
                  <td className="text-danger fw-bold">{p.total_homocysteine_umol_l}</td>
                  <td className="text-primary">{p.methionine_umol_l}</td>
                  <td className={p.serum_folate_nmol_l < 15 ? 'text-muted' : 'text-warning'}>{p.serum_folate_nmol_l}</td>
                  <td className="text-success fw-bold">{p.mecbl_fibroblasts}</td>
                  <td>{p.white_matter_disease ? '✅' : '—'}</td>
                  <td>{p.seizures ? '✅' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Biomarker Ranges (Cohort N=40)" color={ACCENT2}>
        <div className="row g-2">
          {Object.entries(bm).map(([key, val]) => (
            <div className="col-md-4 col-lg-3" key={key}>
              <div className="card card-body py-1 px-2 small shadow-sm">
                <strong className="d-block text-truncate">{key.replace(/_/g,' ')}</strong>
                <span className="text-muted">min {val.min} · avg {val.avg} · max {val.max}</span>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Published Pathogenic Variants (Severe MTHFR Deficiency)" color={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: ACCENT3, color: '#fff' }}>
              <tr>
                <th>Variant</th><th>Domain</th><th>Frequency</th><th>Severity</th><th>Note</th>
              </tr>
            </thead>
            <tbody>
              {vv.map((v, i) => (
                <tr key={i}>
                  <td className="fw-bold">{v.variant}</td>
                  <td>{v.domain}</td>
                  <td>{v.frequency_pct}%</td>
                  <td className={v.severity_class === 'Severe' ? 'text-danger' : v.severity_class === 'Moderate' ? 'text-warning' : 'text-success'}>
                    {v.severity_class}
                  </td>
                  <td className="text-muted small">{v.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <small className="text-muted mt-2 d-block">
          ⚠ C677T and A1298C are COMMON POLYMORPHISMS — NOT pathogenic alone. Severe deficiency requires homozygous or compound heterozygous TRULY pathogenic alleles (both copies non-functional, residual activity &lt;30%).
        </small>
      </SectionCard>
    </div>
  );
}

// ── Seizures & Triggers Tab ───────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  const sz = data.seizure_types || [];
  const tr = data.metabolic_triggers || [];

  return (
    <div>
      <Alert variant="danger"
        text="🚨 WHITE MATTER DISEASE 82% — PROMINENT in MTHFR deficiency. Diffuse periventricular leukoencephalopathy is the most distinctive MRI finding. Contributes to drug-resistant epilepsy. Betaine + methionine supplementation may stabilize progression." />

      <SectionCard title="Seizure Types" color={ACCENT7}>
        <div className="row g-2">
          {sz.map((s, i) => (
            <div className="col-md-4" key={i}>
              <div className="card card-body py-2 px-3 small shadow-sm">
                <strong>{s.type}</strong>
                <div className="text-muted">{s.frequency_pct}% of cases</div>
                <div className="text-muted small mt-1">{s.notes}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Metabolic Triggers & Risk Factors" color={ACCENT2}>
        {tr.map((t, i) => (
          <div key={i} className={`alert ${t.risk_level.includes('ABSOLUTE') ? 'alert-danger' : t.risk_level.includes('HIGH') ? 'alert-warning' : 'alert-secondary'} py-2 mb-2`}>
            <strong>{t.trigger}</strong> — <span className="fw-bold">{t.risk_level}</span>
            <div className="small mt-1">{t.mechanism}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Treatments Tab ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  const tx = data.treatments || [];
  const stats = data.treatment_response_stats || {};

  return (
    <div>
      <Alert variant="success"
        text="💊 MTHFR treatment differs from cblE/cblG: (1) Riboflavin B2 is Level A (FAD cofactor — MTHFR-unique); (2) Methionine supplementation Level A (product deficiency — MTHFR-unique); (3) Folinic acid is Level B (not Level A as in cblE/cblG — no methylfolate trap to bypass)." />
      <Alert variant="warning"
        text="⚠ VPA HIGH RISK — riboflavin depletion worsens FAD-dependent MTHFR function. Also: antifolates (methotrexate, trimethoprim) are HIGH RISK — block DHFR upstream of MTHFR." />

      <SectionCard title="Treatment Protocols" color={ACCENT4}>
        {tx.map((t, i) => (
          <div key={i} className="card mb-2 shadow-sm">
            <div className="card-header d-flex justify-content-between py-1 px-3" style={{ background: '#e8eaf6' }}>
              <strong className="small">{t.treatment}</strong>
              <span className={`badge ${t.evidence_level === 'Level A' ? 'bg-success' : t.evidence_level === 'Level B' ? 'bg-primary' : t.evidence_level === 'Level C' ? 'bg-secondary' : 'bg-danger'}`}>
                {t.evidence_level}
              </span>
            </div>
            <div className="card-body py-2 px-3 small">
              <div><strong>Dose:</strong> {t.dose}</div>
              <div><strong>Mechanism:</strong> {t.mechanism}</div>
              {t.monitoring && <div className="text-muted mt-1">{t.monitoring}</div>}
            </div>
          </div>
        ))}
      </SectionCard>

      {Object.keys(stats).length > 0 && (
        <SectionCard title="Treatment Response Statistics" color={ACCENT5}>
          <div className="row g-2">
            {Object.entries(stats).map(([k, v]) => (
              <div className="col-md-4" key={k}>
                <div className="card card-body py-1 px-2 small shadow-sm">
                  <strong className="d-block">{k.replace(/_/g,' ')}</strong>
                  <span className="text-muted">{JSON.stringify(v)}</span>
                </div>
              </div>
            ))}
          </div>
        </SectionCard>
      )}
    </div>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  const gc = data.gene_card || {};
  const kc = data.key_concepts || [];
  const dt = data.diagnostic_thresholds || [];
  const dd = data.differential_diagnosis || [];

  return (
    <div>
      <SectionCard title="Gene Card — MTHFR" color={ACCENT}>
        <div className="row g-2">
          {Object.entries(gc).map(([k, v]) => (
            <div className="col-md-6" key={k}>
              <div className="small"><strong>{k.replace(/_/g,' ')}:</strong> {String(v)}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Key Clinical Concepts" color={ACCENT3}>
        {kc.map((c, i) => (
          <div key={i} className="mb-3">
            <strong className="d-block text-primary">{c.concept}</strong>
            <div className="small text-muted mt-1">{c.explanation}</div>
            {i < kc.length - 1 && <hr className="my-2" />}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Diagnostic Thresholds & Action Points" color={ACCENT2}>
        {dt.map((d, i) => (
          <div key={i} className="mb-3 border-start border-danger ps-3">
            <strong className="d-block">{d.parameter}</strong>
            <div className="small text-danger">{d.threshold}</div>
            <div className="small text-muted mt-1">{d.action}</div>
            {i < dt.length - 1 && <hr className="my-2" />}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Differential Diagnosis" color={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: ACCENT5, color: '#fff' }}>
              <tr>
                <th>Disease</th><th>Distinguishing Feature</th>
              </tr>
            </thead>
            <tbody>
              {dd.map((d, i) => (
                <tr key={i}>
                  <td className="fw-bold">{d.disease}</td>
                  <td>{d.distinguishing}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function MthfrPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mthfr/overview`).then(r => r.json()),
      fetch(`${API}/api/mthfr/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mthfr/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
    }).catch(e => setError(String(e)));
  }, []);

  if (error) return <div className="alert alert-danger m-3">{error}</div>;

  const tabContent = [
    <OverviewTab data={overview} />,
    <PatientsTab data={breakdown} />,
    <SeizuresTab data={breakdown} />,
    <TreatmentsTab data={breakdown} />,
    <DefinitionsTab data={definitions} />,
  ];

  return (
    <div className="container-fluid py-3">
      <div className="d-flex justify-content-between align-items-center mb-2">
        <h4 style={{ color: ACCENT, margin: 0 }}>
          🧬 MTHFR Epilepsy — Severe MTHFR Deficiency / Homocystinuria-Methylenetetrahydrofolate Reductase Deficiency
        </h4>
        <Link href="/eeg-viz" className="btn btn-outline-primary btn-sm">← EEG Viz</Link>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tabContent[tab]}
    </div>
  );
}

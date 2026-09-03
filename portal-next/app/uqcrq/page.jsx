'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Biochemistry', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#00695c';   // deep teal — CIII structural subunit / Qo-site
const LIGHT  = '#e0f2f1';
const COLOR2 = '#00838f';
const COLOR3 = '#b71c1c';   // red — severe / mortality highlight
const COLOR4 = '#e65100';   // deep orange — dystonia / optic atrophy highlight

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

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${borderColor}` }}>
      <div className="card-body">
        {title && <h6 className="fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>}
        {children}
      </div>
    </div>
  );
}

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const s = data.cohort_statistics || {};
  const feats = data.cohort_summary_features || [];

  return (
    <div>
      {/* Gene header */}
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `5px solid ${COLOR}` }}>
        <h5 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 UQCRQ — Ubiquinol-Cytochrome C Reductase, Complex III Subunit VII (QCR8) / CIII Qo-Site Structural Subunit
        </h5>
        <p className="mb-1 small">
          <strong>OMIM Gene:</strong> *{data.omim_gene} &nbsp;|&nbsp;
          <strong>Chr:</strong> {data.chromosome} &nbsp;|&nbsp;
          <strong>Protein:</strong> {data.protein_size} &nbsp;|&nbsp;
          <strong>Alias:</strong> {data.alias}
        </p>
        <p className="mb-1 small">
          <strong>Inheritance:</strong> {data.inheritance} &nbsp;|&nbsp;
          <strong>Cohort:</strong> {data.cohort_n} patients (seed {data.seed})
        </p>
        <p className="mb-0 small fw-semibold" style={{ color: COLOR }}>
          🟩 UQCRQ: structural subunit at CIII Qo-site periphery — stabilises Rieske ISP (RISP) stalk.
          <span style={{ color: COLOR4 }}> Dystonia (75%) CARDINAL</span> + <span style={{ color: COLOR4 }}>Optic atrophy (42%) DISTINGUISHING</span>.
          Isolated CIII deficiency (8–22% residual). NO cataracts (DDx CYC1). NO GRACILE triad (DDx BCS1L). NO psychiatric features (DDx TTC19).
          BN-PAGE: CIII reduced (sub-complexes present) — unlike UQCC2 (CIII absent).
          KD / Metformin / VPA / Linezolid / Propofol: ABSOLUTE CONTRAINDICATIONS.
        </p>
      </div>

      {/* Clinical alerts */}
      <div className="mb-4">
        {(data.key_clinical_alerts || []).map((a, i) => (
          <div key={i} className={`alert py-1 px-2 mb-1 small ${a.startsWith('🚫') ? 'alert-danger' : a.startsWith('⚠️') ? 'alert-warning' : 'alert-success'}`}>
            {a}
          </div>
        ))}
      </div>

      {/* KPIs row 1 — phenotypes */}
      <h6 className="fw-semibold mb-2" style={{ color: COLOR }}>Phenotype Distribution (n={data.cohort_n})</h6>
      <div className="row mb-3">
        <KPI label="Psychomotor retardation" value={`${s.psychomotor_retardation_pct}%`} color={COLOR3} />
        <KPI label="Hypotonia" value={`${s.hypotonia_pct}%`} color={COLOR} />
        <KPI label="Dystonia (CARDINAL)" value={`${s.dystonia_pct}%`} color={COLOR4} />
        <KPI label="Optic atrophy (DDx)" value={`${s.optic_atrophy_pct}%`} color={COLOR4} />
        <KPI label="Lactic acidosis" value={`${s.lactic_acidosis_pct}%`} color={COLOR3} />
        <KPI label="Deceased (any)" value={`${s.deceased_pct}%`} color={COLOR3} />
      </div>
      <div className="row mb-3">
        <KPI label="Leigh-like MRI" value={`${s.leigh_like_mri_pct}%`} color={COLOR} />
        <KPI label="Seizures" value={`${s.seizures_pct}%`} color={COLOR2} />
        <KPI label="Encephalopathy" value={`${s.encephalopathy_pct}%`} color={COLOR2} />
        <KPI label="Cardiomyopathy (<12%)" value={`${s.cardiomyopathy_pct}%`} color="#388e3c" />
        <KPI label="Avg CIII activity" value={`${s.avg_ciii_activity_pct}%`} color={COLOR3} />
        <KPI label="Avg lactate (mM)" value={`${s.avg_lactic_acid_mmolL}`} color={COLOR3} />
      </div>

      {/* Feature bars */}
      <SectionCard title="Phenotype Frequency — UQCRQ Cohort (CIII Structural Subunit Deficiency)" borderColor={COLOR}>
        <div className="row">
          <div className="col-md-6">
            {feats.slice(0, Math.ceil(feats.length / 2)).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct}
                color={f.pct === 0 ? '#43a047' : f.feature.includes('Dystonia') || f.feature.includes('Optic') ? COLOR4 : COLOR} />
            ))}
          </div>
          <div className="col-md-6">
            {feats.slice(Math.ceil(feats.length / 2)).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct}
                color={f.pct === 0 ? '#43a047' : COLOR2} />
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Sample patients */}
      <SectionCard title="Sample Patients (first 10)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                <th>ID</th><th>Phenotype</th><th>Onset</th>
                <th>Variant 1</th><th>Variant 2</th><th>Zygosity</th>
                <th>CIII%</th><th>Lactate</th><th>Dystonia</th><th>Optic</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map((p, i) => (
                <tr key={i}>
                  <td>{p.patient_id}</td>
                  <td className="small text-muted">{p.phenotype.split('—')[0].trim()}</td>
                  <td>{p.onset_weeks === 0 ? 'Birth' : `${p.onset_weeks}w`}</td>
                  <td><code className="small">{p.variant_1}</code></td>
                  <td><code className="small">{p.variant_2}</code></td>
                  <td><span className="badge" style={{ background: p.zygosity === 'Homozygous' ? COLOR3 : COLOR, fontSize: '0.7em' }}>{p.zygosity.replace('Compound ', 'Comp. ')}</span></td>
                  <td><span style={{ color: p.ciii_activity_pct < 12 ? COLOR3 : COLOR }}>{p.ciii_activity_pct}%</span></td>
                  <td><span style={{ color: p.lactic_acid_mmolL > 12 ? COLOR3 : COLOR2 }}>{p.lactic_acid_mmolL}</span></td>
                  <td>{p.dystonia ? <span style={{ color: COLOR4 }}>✓</span> : '—'}</td>
                  <td>{p.optic_atrophy ? <span style={{ color: COLOR4 }}>✓</span> : '—'}</td>
                  <td><small style={{ color: p.outcome.includes('Deceased') ? COLOR3 : '#388e3c' }}>{p.outcome}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Variants & Biochemistry ──────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  const biochem = data.biochemistry_distribution || {};
  const outcomes = data.outcome_distribution || [];
  const immunoblot = data.immunoblot_pattern || {};
  const bnPage = data.bn_page_pattern || {};

  return (
    <div>
      <SectionCard title="Pathogenic Variants in UQCRQ (OMIM *612080)" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                <th>Protein change</th><th>cDNA</th><th>Domain</th>
                <th>Type</th><th>Severity</th><th>Penetrance</th><th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {(data.all_variants || []).map((v, i) => (
                <tr key={i}>
                  <td><code>{v.protein}</code></td>
                  <td><code className="text-muted">{v.cdna}</code></td>
                  <td>{v.domain}</td>
                  <td><span className="badge bg-secondary">{v.type}</span></td>
                  <td><span className="badge" style={{ background: v.severity === 'Severe' ? COLOR3 : v.severity === 'Moderate' ? '#f57c00' : '#388e3c' }}>{v.severity}</span></td>
                  <td><strong style={{ color: v.penetrance_pct >= 85 ? COLOR3 : COLOR }}>{v.penetrance_pct}%</strong></td>
                  <td className="text-muted small">{v.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="CIII Activity Distribution" borderColor={COLOR}>
            <p className="small text-muted mb-2">Average CIII residual: <strong>{biochem.avg_ciii_activity_pct}%</strong> (8–22% range; higher than UQCC2 5–18%)</p>
            <Bar label="CIII ≤12%" value={biochem.ciii_8to12_pct} color={COLOR3} />
            <Bar label="CIII 12–17%" value={biochem.ciii_12to17_pct} color={COLOR2} />
            <Bar label="CIII >17%" value={biochem.ciii_17to22_pct} color={COLOR} />
            <p className="small text-muted mt-2">Average lactate: <strong>{biochem.avg_lactic_acid_mmolL} mM</strong></p>
            <Bar label="Lactate >10 mM" value={biochem.lactic_above_10_pct} color={COLOR3} />
            <Bar label="Lactate 5–10 mM" value={biochem.lactic_5_to_10_pct} color={COLOR2} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Outcome Distribution" borderColor={COLOR3}>
            {outcomes.map((o, i) => (
              <div key={i} className="d-flex justify-content-between align-items-center mb-1">
                <span className="small">{o.outcome}</span>
                <span className="badge" style={{ background: o.outcome.includes('Deceased') ? COLOR3 : '#388e3c' }}>{o.count}</span>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="BN-PAGE Pattern (UQCRQ)" borderColor={COLOR4}>
            <p className="fw-bold small mb-1" style={{ color: COLOR3 }}>{bnPage.finding}</p>
            <p className="small mb-2 text-muted">{bnPage.interpretation}</p>
            <div className="alert alert-info py-1 px-2 small">{bnPage.ddx_value}</div>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Immunoblot Pattern (UQCRQ)" borderColor={COLOR2}>
            {Object.entries(immunoblot).map(([k, v], i) => (
              <div key={i} className="d-flex justify-content-between align-items-start mb-1 border-bottom pb-1">
                <span className="small fw-semibold" style={{ minWidth: 140 }}>{k.replace(/_/g, ' ')}</span>
                <span className="small text-muted ms-2">{v}</span>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Treatment uptake */}
      <SectionCard title="Treatment Uptake (n=40)" borderColor={COLOR}>
        <div className="row">
          {Object.entries(data.treatment_uptake || {}).map(([k, v], i) => (
            <div key={i} className="col-md-6 mb-1">
              <div className="d-flex justify-content-between small">
                <span>{k}</span>
                <span className="badge" style={{ background: COLOR2 }}>{v}</span>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: DDx & Treatment ──────────────────────────────────────────────────────
function DDxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ddx = data.key_ddx || [];
  const abs_ci = data.absolute_contraindications || [];
  const rx = data.recommended_treatments || [];
  const gc = data.genetic_counselling || {};

  return (
    <div>
      <div className="alert alert-warning mb-4 small">
        <strong>⚠️ Key distinguishing features of UQCRQ:</strong> (1) <strong style={{ color: COLOR4 }}>Dystonia (75%) — CARDINAL</strong> — more prominent than UQCC2/UQCC3/LYRM7; basal ganglia vulnerability to Qo-site ROS.
        (2) <strong style={{ color: COLOR4 }}>Optic atrophy (42%) — DISTINGUISHING</strong> — ABSENT in UQCC2/UQCC3/LYRM7/TTC19; retinal ganglion cells selectively vulnerable.
        (3) BN-PAGE: CIII reduced with sub-complexes (NOT absent as in UQCC2) — WES mandatory to distinguish from UQCRC2.
        (4) NO cataracts (DDx CYC1 35%). NO GRACILE triad (DDx BCS1L). NO psychiatric features (DDx TTC19).
      </div>

      <SectionCard title="Differential Diagnosis — UQCRQ vs Other CIII/OXPHOS Diseases" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                <th>Condition</th><th>Distinguishing feature(s)</th>
              </tr>
            </thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ color: COLOR2, whiteSpace: 'nowrap' }}>{d.condition}</td>
                  <td>{d.distinguishing}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Absolute Contraindications" borderColor={COLOR3}>
            {abs_ci.map((ci, i) => (
              <div key={i} className="alert alert-danger py-1 px-2 mb-1 small">🚫 {ci}</div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Recommended Treatments" borderColor="#388e3c">
            {rx.map((r, i) => (
              <div key={i} className="alert alert-success py-1 px-2 mb-1 small">✅ {r}</div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Genetic Counselling" borderColor={COLOR2}>
        <div className="row">
          {Object.entries(gc).map(([k, v], i) => (
            <div key={i} className="col-md-6 mb-2">
              <div className="fw-semibold small text-capitalize mb-1" style={{ color: COLOR2 }}>{k.replace(/_/g, ' ')}</div>
              <div className="small text-muted">{v}</div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const terms = data.terms || [];
  const refs = data.key_references || [];
  const biochem = data.key_biochemical_features || [];
  const prot = data.protein || {};

  return (
    <div>
      <div className="row mb-4">
        <div className="col-md-6">
          <SectionCard title="Gene & Protein Summary" borderColor={COLOR}>
            <table className="table table-sm small mb-0">
              <tbody>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>Gene</td><td>{data.gene} ({data.alias})</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>Full name</td><td>{data.full_name}</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>OMIM Gene</td><td>*{data.omim_gene}</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>Chromosome</td><td>{data.chromosome}</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>Inheritance</td><td>{data.inheritance}</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>Protein size</td><td>{prot.size_aa} aa, {prot.kDa} kDa</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>TM helices</td><td>{prot.tm_helices} (N-terminal, aa 1–21)</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>Localization</td><td>{prot.localization}</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>Function</td><td>{prot.function}</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>CIII step</td><td>{data.ciii_assembly_step}</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>BN-PAGE</td><td>{data.bn_page}</td></tr>
              </tbody>
            </table>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Key Biochemical Features" borderColor={COLOR2}>
            {biochem.map((b, i) => (
              <div key={i} className="small mb-1 pb-1 border-bottom">• {b}</div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Glossary" borderColor={COLOR4}>
        <div className="row">
          {terms.map((t, i) => (
            <div key={i} className="col-md-6 mb-3">
              <div className="fw-bold small" style={{ color: COLOR4 }}>{t.term}</div>
              <div className="small text-muted">{t.definition}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Key References" borderColor={COLOR}>
        {refs.map((r, i) => (
          <div key={i} className="small mb-2 pb-2 border-bottom">📄 {r}</div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function UQCRQPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/uqcrq/overview`).then(r => r.json()).then(setOverview).catch(() => setError('overview failed'));
    fetch(`${API}/api/uqcrq/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => setError('breakdown failed'));
    fetch(`${API}/api/uqcrq/definitions`).then(r => r.json()).then(setDefs).catch(() => setError('definitions failed'));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2" style={{ borderBottom: `3px solid ${COLOR}`, paddingBottom: 8 }}>
        <span style={{ fontSize: 28 }}>🧬</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>UQCRQ — Complex III Structural Subunit Deficiency</h4>
          <small className="text-muted">QCR8 / Subunit VII · Qo-site periphery · Isolated CIII Deficiency · OMIM *612080 · 5q31.1 · AR biallelic · 40-patient cohort seed-729</small>
        </div>
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderColor: `${COLOR} ${COLOR} #fff` } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <VariantsTab data={breakdown} />}
      {tab === 2 && <DDxTab data={defs} />}
      {tab === 3 && <DefinitionsTab data={defs} />}
    </div>
  );
}

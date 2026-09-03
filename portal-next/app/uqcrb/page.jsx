'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Biochemistry', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#1a237e';   // deep indigo — CIII Qi-site structural / matrix face
const LIGHT  = '#e8eaf6';
const COLOR2 = '#283593';
const COLOR3 = '#b71c1c';   // red — severe / mortality
const COLOR4 = '#e65100';   // deep orange — hypoglycaemia / hepatopathy highlight
const COLOR5 = '#2e7d32';   // green — absent features / treatment

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
          🧬 UQCRB — Ubiquinol-Cytochrome C Reductase Binding Protein (QCR7 / QP-C) / CIII Qi-site Peripheral Structural Subunit
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
          🟦 UQCRB: peripheral Qi-site structural subunit (NO TM helix, matrix face) — stabilises UQCRC1/UQCRC2 scaffold at Qi site.
          <span style={{ color: COLOR4 }}> Hypoglycaemia (65%) DISTINGUISHING</span> + <span style={{ color: COLOR4 }}>Hepatopathy (55%) — more than UQCRQ</span>.
          Isolated CIII deficiency (5–15% residual; more severe than UQCRQ 8–22%). NO dystonia as cardinal (DDx UQCRQ 75%). NO optic atrophy as distinguishing (DDx UQCRQ 42%).
          BN-PAGE: CIII severely reduced — RISP relatively preserved (Qi-side defect; Qo-side RISP insert intact).
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

      {/* KPIs row 1 */}
      <h6 className="fw-semibold mb-2" style={{ color: COLOR }}>Phenotype Distribution (n={data.cohort_n})</h6>
      <div className="row mb-3">
        <KPI label="Lactic acidosis" value={`${s.lactic_acidosis_pct}%`} color={COLOR3} />
        <KPI label="Hypotonia" value={`${s.hypotonia_pct}%`} color={COLOR} />
        <KPI label="Hypoglycaemia (DDx)" value={`${s.hypoglycemia_pct}%`} color={COLOR4} />
        <KPI label="Hepatopathy" value={`${s.hepatopathy_pct}%`} color={COLOR4} />
        <KPI label="Avg CIII activity" value={`${s.avg_ciii_activity_pct}%`} color={COLOR3} />
        <KPI label="Deceased (any)" value={`${s.deceased_pct}%`} color={COLOR3} />
      </div>
      <div className="row mb-3">
        <KPI label="Encephalopathy" value={`${s.encephalopathy_pct}%`} color={COLOR2} />
        <KPI label="Leigh-like MRI" value={`${s.leigh_like_mri_pct}%`} color={COLOR} />
        <KPI label="Seizures" value={`${s.seizures_pct}%`} color={COLOR2} />
        <KPI label="Dystonia (<25%)" value={`${s.dystonia_pct}%`} color={COLOR5} />
        <KPI label="Optic atrophy (<10%)" value={`${s.optic_atrophy_pct}%`} color={COLOR5} />
        <KPI label="Avg lactate (mM)" value={`${s.avg_lactic_acid_mmolL}`} color={COLOR3} />
      </div>

      {/* Feature bars */}
      <SectionCard title="Phenotype Frequency — UQCRB Cohort (CIII Qi-site Structural Subunit Deficiency)" borderColor={COLOR}>
        <div className="row">
          <div className="col-md-6">
            {feats.slice(0, Math.ceil(feats.length / 2)).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct}
                color={f.pct <= 10 ? COLOR5 : f.feature.includes('Hypoglycaemia') || f.feature.includes('Hepatopathy') ? COLOR4 : COLOR} />
            ))}
          </div>
          <div className="col-md-6">
            {feats.slice(Math.ceil(feats.length / 2)).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct}
                color={f.pct <= 10 ? COLOR5 : COLOR2} />
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
                <th>CIII%</th><th>Lactate</th><th>Hypogly</th><th>Hepato</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map((p, i) => (
                <tr key={i}>
                  <td>{p.patient_id}</td>
                  <td className="small text-muted">{(p.phenotype || '').split('—')[0].trim()}</td>
                  <td>{p.onset_weeks === 0 ? 'Birth' : `${p.onset_weeks}w`}</td>
                  <td><code className="small">{p.variant_1}</code></td>
                  <td><code className="small">{p.variant_2}</code></td>
                  <td><span className="badge" style={{ background: p.zygosity === 'Homozygous' ? COLOR3 : COLOR, fontSize: '0.7em' }}>{(p.zygosity || '').replace('Compound ', 'Comp. ')}</span></td>
                  <td><span style={{ color: p.ciii_activity_pct < 10 ? COLOR3 : COLOR2 }}>{p.ciii_activity_pct}%</span></td>
                  <td><span style={{ color: p.lactic_acid_mmolL > 10 ? COLOR3 : COLOR2 }}>{p.lactic_acid_mmolL}</span></td>
                  <td>{p.hypoglycemia ? <span style={{ color: COLOR4 }}>✓</span> : '—'}</td>
                  <td>{p.hepatopathy ? <span style={{ color: COLOR4 }}>✓</span> : '—'}</td>
                  <td><small style={{ color: (p.outcome || '').includes('Deceased') ? COLOR3 : COLOR5 }}>{p.outcome}</small></td>
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
      <SectionCard title="Pathogenic Variants in UQCRB (OMIM *191330)" borderColor={COLOR}>
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
                  <td><span className="badge" style={{ background: v.severity === 'Severe' ? COLOR3 : v.severity === 'Moderate' ? '#f57c00' : COLOR5 }}>{v.severity}</span></td>
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
            <p className="small text-muted mb-2">Average CIII residual: <strong>{biochem.avg_ciii_activity_pct}%</strong> (5–15% range; more severe than UQCRQ 8–22%)</p>
            <Bar label="CIII ≤10%" value={biochem.ciii_5to10_pct} color={COLOR3} />
            <Bar label="CIII 10–12%" value={biochem.ciii_10to12_pct} color={COLOR4} />
            <Bar label="CIII 12–15%" value={biochem.ciii_12to15_pct} color={COLOR2} />
            <Bar label="CIII >15%" value={biochem.ciii_above15_pct} color={COLOR} />
            <p className="small text-muted mt-2">Average lactate: <strong>{biochem.avg_lactic_acid_mmolL} mM</strong></p>
            <Bar label="Lactate >10 mM" value={biochem.lactic_above_10_pct} color={COLOR3} />
            <Bar label="Lactate 5–10 mM" value={biochem.lactic_5_to_10_pct} color={COLOR2} />
            <Bar label="Lactate <5 mM" value={biochem.lactic_below5_pct} color={COLOR} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Outcome Distribution" borderColor={COLOR3}>
            {outcomes.map((o, i) => (
              <div key={i} className="d-flex justify-content-between align-items-center mb-1">
                <span className="small">{o.outcome}</span>
                <div>
                  <span className="badge me-1" style={{ background: (o.outcome || '').includes('Deceased') ? COLOR3 : COLOR5 }}>{o.count}</span>
                  <span className="text-muted small">{o.pct}%</span>
                </div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="BN-PAGE Pattern (UQCRB)" borderColor={COLOR4}>
            <p className="fw-bold small mb-1" style={{ color: COLOR3 }}>{bnPage.finding}</p>
            <p className="small mb-2 text-muted">{bnPage.interpretation}</p>
            <div className="alert alert-info py-1 px-2 small">{bnPage.ddx_value}</div>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Immunoblot Pattern (UQCRB)" borderColor={COLOR2}>
            {Object.entries(immunoblot).map(([k, v], i) => (
              <div key={i} className="d-flex justify-content-between align-items-start mb-1 border-bottom pb-1">
                <span className="small fw-semibold" style={{ minWidth: 160 }}>{k.replace(/_/g, ' ')}</span>
                <span className="small text-muted ms-2">{v}</span>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

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
        <strong>⚠️ Key distinguishing features of UQCRB:</strong>{' '}
        (1) <strong style={{ color: COLOR4 }}>Hypoglycaemia (65%) — DISTINGUISHING</strong> — Qi-site CIII block impairs gluconeogenesis/β-oxidation; absent in UQCRQ; NEVER fast; GIR 6–8 mandatory.
        (2) <strong style={{ color: COLOR4 }}>Hepatopathy (55%)</strong> — more prominent than UQCRQ; monitor LFTs.
        (3) <strong style={{ color: COLOR5 }}>NO dystonia as cardinal</strong> (DDx UQCRQ 75%) — Qi-site ROS burden lower than Qo-site.
        (4) <strong style={{ color: COLOR5 }}>NO optic atrophy as distinguishing</strong> (DDx UQCRQ 42%).
        (5) BN-PAGE: CIII severely reduced (5–15%); RISP relatively preserved (Qi-side defect; Qo-side insert intact).
        (6) UQCRB at 8q22.1; CYC1 at 8q24.1 — same chromosome arm; WES mandatory.
      </div>

      <SectionCard title="Differential Diagnosis — UQCRB vs Other CIII/OXPHOS Diseases" borderColor={COLOR}>
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
          <SectionCard title="Recommended Treatments" borderColor={COLOR5}>
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
                <tr><td className="fw-semibold" style={{ color: COLOR }}>Protein size</td><td>{prot.size_aa} aa, {prot.kDa} kDa (MTS-cleaved mature ~96 aa)</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>TM helices</td><td>{prot.tm_helices} — peripheral subunit, matrix face (NO TM helix)</td></tr>
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
export default function UQCRBPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/uqcrb/overview`).then(r => r.json()).then(setOverview).catch(() => setError('overview failed'));
    fetch(`${API}/api/uqcrb/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => setError('breakdown failed'));
    fetch(`${API}/api/uqcrb/definitions`).then(r => r.json()).then(setDefs).catch(() => setError('definitions failed'));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2" style={{ borderBottom: `3px solid ${COLOR}`, paddingBottom: 8 }}>
        <span style={{ fontSize: 28 }}>🧬</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>UQCRB — Complex III Qi-site Structural Subunit Deficiency</h4>
          <small className="text-muted">QCR7 / QP-C / Subunit VI · Qi-site matrix face · NO TM helix · Isolated CIII Deficiency · OMIM *191330 · 8q22.1 · AR biallelic · 40-patient cohort seed-733</small>
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

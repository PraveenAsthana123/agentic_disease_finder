'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Biochemistry', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#1a237e';   // deep indigo — CIII early assembly / neonatal
const LIGHT  = '#e8eaf6';
const COLOR2 = '#283593';
const COLOR3 = '#b71c1c';   // red — severe neonatal / mortality highlight
const COLOR4 = '#4a148c';   // purple — key DDx highlight

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
          🧬 UQCC2 — Ubiquinol-Cytochrome C Reductase Complex Assembly Factor 2 / CIII Early Assembly Factor
        </h5>
        <p className="mb-1 small">
          <strong>OMIM Gene:</strong> *{data.omim_gene} &nbsp;|&nbsp;
          <strong>Disease:</strong> CIII-D7 #{data.omim_disease} &nbsp;|&nbsp;
          <strong>Chr:</strong> {data.chromosome} &nbsp;|&nbsp;
          <strong>Protein:</strong> {data.protein_size} &nbsp;|&nbsp;
          <strong>Alias:</strong> {data.alias}
        </p>
        <p className="mb-1 small">
          <strong>Inheritance:</strong> {data.inheritance} &nbsp;|&nbsp;
          <strong>Cohort:</strong> {data.cohort_n} patients (seed {data.seed})
        </p>
        <p className="mb-0 small fw-semibold" style={{ color: COLOR }}>
          🟦 UQCC2: earliest CIII assembly factor — UQCC1-UQCC2 heterodimer stabilises MT-CYB (CIII*).
          Severe <span style={{ color: COLOR3 }}>neonatal/early infantile</span> onset. Lactic acidosis + hypotonia + feeding failure + respiratory failure.
          NO psychiatric features (DDx TTC19). NO GRACILE triad (DDx BCS1L).
          BN-PAGE: CIII absent, NO sub-complexes. KD / Metformin / VPA / Linezolid / Propofol: ABSOLUTE CONTRAINDICATIONS.
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
        <KPI label="Neonatal onset" value={`${s.neonatal_onset_pct}%`} color={COLOR3} />
        <KPI label="Hypotonia" value={`${s.hypotonia_pct}%`} color={COLOR} />
        <KPI label="Feeding difficulties" value={`${s.feeding_difficulties_pct}%`} color={COLOR2} />
        <KPI label="Encephalopathy" value={`${s.encephalopathy_pct}%`} color={COLOR4} />
        <KPI label="Resp. failure" value={`${s.respiratory_failure_pct}%`} color={COLOR3} />
        <KPI label="Deceased (any)" value={`${s.deceased_pct}%`} color={COLOR3} />
      </div>
      <div className="row mb-3">
        <KPI label="Leigh-like MRI" value={`${s.leigh_like_mri_pct}%`} color={COLOR} />
        <KPI label="Seizures" value={`${s.seizures_pct}%`} color={COLOR2} />
        <KPI label="Growth restriction" value={`${s.growth_restriction_pct}%`} color={COLOR2} />
        <KPI label="Cardiomyopathy" value={`${s.cardiomyopathy_pct}%`} color="#388e3c" />
        <KPI label="Avg CIII activity" value={`${s.avg_ciii_activity_pct}%`} color={COLOR3} />
        <KPI label="Avg lactate (mM)" value={`${s.avg_lactic_acid_mmolL}`} color={COLOR3} />
      </div>

      {/* Feature bars */}
      <SectionCard title="Phenotype Frequency — UQCC2 Cohort" borderColor={COLOR}>
        <div className="row">
          <div className="col-md-6">
            {feats.slice(0, Math.ceil(feats.length / 2)).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct} color={f.pct === 0 ? '#43a047' : COLOR} />
            ))}
          </div>
          <div className="col-md-6">
            {feats.slice(Math.ceil(feats.length / 2)).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct} color={f.pct === 0 ? '#43a047' : COLOR2} />
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
                <th>ID</th><th>Phenotype</th><th>Onset (wks)</th>
                <th>Variant 1</th><th>Variant 2</th><th>Zygosity</th>
                <th>CIII %</th><th>Lactate</th><th>Leigh MRI</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map((p, i) => (
                <tr key={i}>
                  <td>{p.patient_id}</td>
                  <td>{p.phenotype}</td>
                  <td>{p.onset_weeks === 0 ? 'Birth' : `${p.onset_weeks}w`}</td>
                  <td><code className="small">{p.variant_1}</code></td>
                  <td><code className="small">{p.variant_2}</code></td>
                  <td><span className="badge" style={{ background: p.zygosity === 'Homozygous' ? COLOR3 : COLOR, fontSize: '0.7em' }}>{p.zygosity.replace('Compound ', 'Comp. ')}</span></td>
                  <td><span style={{ color: p.ciii_activity_pct < 10 ? COLOR3 : COLOR }}>{p.ciii_activity_pct}%</span></td>
                  <td><span style={{ color: p.lactic_acid_mmolL > 15 ? COLOR3 : COLOR2 }}>{p.lactic_acid_mmolL}</span></td>
                  <td>{p.leigh_like_mri ? '✓' : '—'}</td>
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
      <SectionCard title="Pathogenic Variants in UQCC2 (OMIM *614461)" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                <th>Protein change</th><th>cDNA</th><th>Domain</th>
                <th>Type</th><th>Severity</th><th>Penetrance</th><th>Mechanism / Notes</th>
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
                  <td className="text-muted">{v.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="CIII Activity Distribution" borderColor={COLOR}>
            <p className="small text-muted mb-2">Average CIII residual: <strong>{biochem.avg_ciii_activity_pct}%</strong> (5-18% range; severe deficiency)</p>
            <Bar label="CIII ≤10%" value={biochem.ciii_5to10_pct} color={COLOR3} />
            <Bar label="CIII 10-15%" value={biochem.ciii_10to15_pct} color={COLOR2} />
            <Bar label="CIII >15%" value={biochem.ciii_15to20_pct} color={COLOR} />
            <p className="small text-muted mt-2">Average lactate: <strong>{biochem.avg_lactic_acid_mmolL} mM</strong></p>
            <Bar label="Lactate >15 mM" value={biochem.lactic_above_15_pct} color={COLOR3} />
            <Bar label="Lactate 10-15 mM" value={biochem.lactic_10_to_15_pct} color={COLOR2} />
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
          <SectionCard title="BN-PAGE Pattern (UQCC2)" borderColor={COLOR4}>
            <p className="fw-bold small mb-1" style={{ color: COLOR3 }}>{bnPage.finding}</p>
            <p className="small mb-2 text-muted">{bnPage.interpretation}</p>
            <div className="alert alert-info py-1 px-2 small">{bnPage.ddx_value}</div>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Immunoblot Pattern (UQCC2)" borderColor={COLOR2}>
            {Object.entries(immunoblot).map(([k, v], i) => (
              <div key={i} className="d-flex justify-content-between mb-1">
                <span className="small fw-semibold">{k}</span>
                <span className="small text-muted">{v}</span>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="All Patients Table" borderColor={COLOR2}>
        <div className="table-responsive" style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table className="table table-sm table-hover small">
            <thead className="sticky-top bg-white">
              <tr>
                <th>ID</th><th>Onset</th><th>CIII%</th><th>Lactate</th>
                <th>Hypotonia</th><th>Resp.</th><th>Seizures</th><th>HCM</th><th>Leigh MRI</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.all_patients || []).map((p, i) => (
                <tr key={i}>
                  <td>{p.patient_id}</td>
                  <td>{p.onset_weeks === 0 ? 'Birth' : `${p.onset_weeks}w`}</td>
                  <td style={{ color: p.ciii_activity_pct < 10 ? COLOR3 : COLOR2 }}>{p.ciii_activity_pct}%</td>
                  <td style={{ color: p.lactic_acid_mmolL > 15 ? COLOR3 : 'inherit' }}>{p.lactic_acid_mmolL}</td>
                  <td>{p.hypotonia ? '✓' : '—'}</td>
                  <td>{p.respiratory_failure ? '✓' : '—'}</td>
                  <td>{p.seizures ? '✓' : '—'}</td>
                  <td>{p.cardiomyopathy ? '⚠️' : '—'}</td>
                  <td>{p.leigh_like_mri ? '✓' : '—'}</td>
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

// ── Tab: DDx & Treatment ──────────────────────────────────────────────────────
function DDxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  const ddx = data.key_ddx || [];
  const ci = data.absolute_contraindications || [];
  const tx = data.recommended_treatments || [];
  const gc = data.genetic_counselling || {};
  const treatment = data.treatment_uptake || {};

  return (
    <div>
      <SectionCard title="Key Differential Diagnoses" borderColor={COLOR4}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr><th>Condition</th><th>How to distinguish from UQCC2</th></tr>
            </thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i}>
                  <td><strong style={{ color: COLOR4 }}>{d.condition}</strong></td>
                  <td className="text-muted">{d.distinguishing}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="🚫 Absolute Contraindications" borderColor={COLOR3}>
            {ci.map((c, i) => (
              <div key={i} className="alert alert-danger py-1 px-2 mb-1 small">🚫 {c}</div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="✅ Recommended Treatments" borderColor="#388e3c">
            {tx.map((t, i) => (
              <div key={i} className="alert alert-success py-1 px-2 mb-1 small">✅ {t}</div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Treatment Uptake (Cohort)" borderColor={COLOR2}>
        <div className="row">
          {Object.entries(treatment).map(([k, v], i) => (
            <div key={i} className="col-md-6 mb-1">
              <div className="d-flex justify-content-between small">
                <span>{k}</span><strong>{v} pts</strong>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Genetic Counselling" borderColor={COLOR}>
        <div className="row">
          {Object.entries(gc).map(([k, v], i) => (
            <div key={i} className="col-md-6 mb-2">
              <div className="fw-semibold small" style={{ color: COLOR }}>{k.replace(/_/g, ' ')}</div>
              <div className="text-muted small">{v}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="CIII Assembly — UQCC2 Role" borderColor={COLOR2}>
        <ol className="small mb-0">
          <li className="mb-1"><strong>MT-CYB synthesis (matrix)</strong> — sole mitochondrially-encoded CIII subunit; scaffold for CIII assembly</li>
          <li className="mb-1"><strong>UQCC1-UQCC2 bind nascent MT-CYB</strong> → forms <strong>CIII* (earliest assembly intermediate)</strong>; UQCC2 loss = MT-CYB immediately degraded by m-AAA protease; no CIII assembly possible</li>
          <li className="mb-1"><strong>Early subunits join CIII*</strong> — UQCRB, UQCRQ add to growing core; UQCC1-UQCC2 dissociate</li>
          <li className="mb-1"><strong>UQCRC1, UQCRC2, CYC1, UQCRH join</strong> — expanding early-core intermediate</li>
          <li className="mb-1"><strong>TTC19 stabilises later intermediate</strong> — distinct step from UQCC2</li>
          <li className="mb-1"><strong>BCS1L inserts RISP (UQCRFS1)</strong> → catalytically active CIII holocomplex → Q-cycle begins → CoQH2 oxidised</li>
        </ol>
        <div className="alert alert-info py-1 px-2 mt-2 small">
          <strong>Key:</strong> UQCC2 acts at the earliest step — before UQCRC1/UQCRC2 join, before TTC19, before BCS1L.
          Loss → zero CIII assembly. Contrast BCS1L (late step, precomplex accumulates) and TTC19 (intermediate stabilisation).
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  return (
    <div>
      <SectionCard title="Gene / Disease Reference" borderColor={COLOR}>
        <div className="row">
          {[
            ['Gene', data.gene],
            ['Full name', data.full_name],
            ['Alias', data.alias],
            ['OMIM Gene', `*${data.omim_gene}`],
            ['OMIM Disease', `#${data.omim_disease}`],
            ['Disease name', data.disease_name],
            ['Chromosome', data.chromosome],
            ['Inheritance', data.inheritance],
            ['Protein size', `${data.protein?.size_aa} aa ${data.protein?.kDa} kDa`],
            ['TM helices', data.protein?.tm_helices],
            ['Localization', data.protein?.localization],
            ['Partner protein', data.protein?.partner],
            ['CIII assembly step', data.ciii_assembly_step],
            ['BN-PAGE', data.bn_page],
          ].map(([k, v], i) => (
            <div key={i} className="col-md-6 mb-2">
              <div className="fw-semibold small" style={{ color: COLOR }}>{k}</div>
              <div className="text-muted small">{v}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Key Biochemical Features" borderColor={COLOR2}>
            <ul className="small mb-0 ps-3">
              {(data.key_biochemical_features || []).map((f, i) => <li key={i} className="mb-1">{f}</li>)}
            </ul>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Key References" borderColor={COLOR4}>
            <ul className="small mb-0 ps-3">
              {(data.key_references || []).map((r, i) => <li key={i} className="mb-1 text-muted">{r}</li>)}
            </ul>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Clinical / Genetic Terms" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead>
              <tr><th>Term</th><th>Definition</th></tr>
            </thead>
            <tbody>
              {(data.terms || []).map((t, i) => (
                <tr key={i}>
                  <td><strong style={{ color: COLOR }}>{t.term}</strong></td>
                  <td className="text-muted">{t.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function UQCC2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/uqcc2/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(() => setError('Failed to load overview'));
    fetch(`${API}/api/uqcc2/breakdown`)
      .then(r => r.json()).then(setBreakdown)
      .catch(() => setError('Failed to load breakdown'));
    fetch(`${API}/api/uqcc2/definitions`)
      .then(r => r.json()).then(setDefinitions)
      .catch(() => setError('Failed to load definitions'));
  }, []);

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
        🧬 UQCC2 — Complex III Deficiency Nuclear Type 7 (CIII-D7)
      </h4>
      <p className="text-muted small mb-3">
        OMIM Gene *614461 &nbsp;|&nbsp; Disease #615824 &nbsp;|&nbsp; 6p21.2 &nbsp;|&nbsp;
        UQCC1-UQCC2 heterodimer — earliest CIII assembly factor (CIII* stabilisation) &nbsp;|&nbsp;
        AR biallelic &nbsp;|&nbsp; Neonatal onset &nbsp;|&nbsp; Seed {717}
      </p>
      {error && <div className="alert alert-danger">{error}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { borderBottom: `3px solid ${COLOR}`, color: COLOR } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <VariantsTab data={breakdown} />}
      {tab === 2 && <DDxTab data={{ ...breakdown, ...definitions }} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}

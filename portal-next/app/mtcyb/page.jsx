'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Biochemistry', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#4a148c';   // deep purple — mitochondrial / maternal inheritance
const LIGHT  = '#f3e5f5';
const COLOR2 = '#6a1b9a';
const COLOR3 = '#b71c1c';   // red — severe / mortality
const COLOR4 = '#e65100';   // deep orange — exercise intolerance / myoglobinuria highlight
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
  const pheno = data.phenotype_distribution || {};

  return (
    <div>
      {/* Gene header */}
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `5px solid ${COLOR}` }}>
        <h5 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 MT-CYB — Mitochondrially Encoded Cytochrome b / ONLY mtDNA-Encoded CIII Structural Subunit
        </h5>
        <p className="mb-1 small">
          <strong>OMIM Gene:</strong> *{data.omim_gene} &nbsp;|&nbsp;
          <strong>Genome:</strong> {data.chromosome} &nbsp;|&nbsp;
          <strong>Protein:</strong> {data.protein_size} &nbsp;|&nbsp;
          <strong>Alias:</strong> {data.alias}
        </p>
        <p className="mb-1 small">
          <strong>Inheritance:</strong> {data.inheritance} &nbsp;|&nbsp;
          <strong>Cohort:</strong> {data.cohort_n} patients (seed {data.seed})
        </p>
        <p className="mb-0 small fw-semibold" style={{ color: COLOR }}>
          🟣 MT-CYB: ONLY mtDNA-encoded CIII subunit — catalytic core (Qo + Qi sites, 8 TM helices, heme bL + bH).
          <span style={{ color: COLOR4 }}> MATERNAL inheritance + HETEROPLASMY</span> — key DDx from ALL nuclear CIII defects (AR biallelic).
          <span style={{ color: COLOR4 }}> Exercise intolerance/myopathy ({pheno.adult_exercise_intolerance_pct}%)</span> DISTINGUISHING (rare in nuclear CIII).
          Myoglobinuria PATHOGNOMONIC for mtDNA myopathy. Isolated CIII ({s.avg_ciii_activity_pct}% avg residual).
          VPA / Metformin / Linezolid / Chloramphenicol / Propofol: ABSOLUTE CONTRAINDICATIONS.
        </p>
      </div>

      {/* Phenotype banner */}
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card text-center shadow-sm" style={{ borderTop: `3px solid ${COLOR4}` }}>
            <div className="card-body py-2">
              <div className="fw-bold fs-4" style={{ color: COLOR4 }}>{pheno.adult_exercise_intolerance_pct}%</div>
              <div className="small text-muted">Adult Exercise Intolerance / Myopathy (MILD)</div>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card text-center shadow-sm" style={{ borderTop: `3px solid ${COLOR3}` }}>
            <div className="card-body py-2">
              <div className="fw-bold fs-4" style={{ color: COLOR3 }}>{pheno.infantile_severe_pct}%</div>
              <div className="small text-muted">Infantile / Severe (Leigh syndrome)</div>
            </div>
          </div>
        </div>
      </div>

      {/* Clinical alerts */}
      <div className="mb-4">
        {(data.key_clinical_alerts || []).map((a, i) => (
          <div key={i} className={`alert py-1 px-2 mb-1 small ${a.startsWith('🚫') ? 'alert-danger' : a.startsWith('⚠️') ? 'alert-warning' : 'alert-success'}`}>
            {a}
          </div>
        ))}
      </div>

      {/* KPIs */}
      <h6 className="fw-semibold mb-2" style={{ color: COLOR }}>Phenotype Distribution (n={data.cohort_n})</h6>
      <div className="row mb-3">
        <KPI label="Exercise intolerance (mild)" value={`${s.exercise_intolerance_pct}%`} color={COLOR4} />
        <KPI label="CK elevation (attacks)" value={`${s.ck_elevated_pct}%`} color={COLOR4} />
        <KPI label="Myoglobinuria" value={`${s.myoglobinuria_pct}%`} color={COLOR4} />
        <KPI label="Maternal family hx" value={`${s.maternal_family_affected_pct}%`} color={COLOR} />
        <KPI label="Avg CIII activity" value={`${s.avg_ciii_activity_pct}%`} color={COLOR3} />
        <KPI label="Deceased (any)" value={`${s.deceased_pct}%`} color={COLOR3} />
      </div>
      <div className="row mb-3">
        <KPI label="Lactic acidosis (severe)" value={`${s.lactic_acidosis_pct}%`} color={COLOR3} />
        <KPI label="Hypotonia (severe)" value={`${s.hypotonia_pct}%`} color={COLOR2} />
        <KPI label="Dev. delay (severe)" value={`${s.developmental_delay_pct}%`} color={COLOR2} />
        <KPI label="Leigh-like MRI" value={`${s.leigh_like_mri_pct}%`} color={COLOR} />
        <KPI label="Ragged-red fibres" value={`${s.ragged_red_fibers_pct}%`} color={COLOR} />
        <KPI label="Avg lactate (mM)" value={`${s.avg_lactic_acid_mmolL}`} color={COLOR3} />
      </div>

      {/* Feature bars */}
      <SectionCard title="Phenotype Frequency — MT-CYB Cohort (mtDNA CIII Deficiency: Exercise Intolerance + Severe spectrum)" borderColor={COLOR}>
        <div className="row">
          <div className="col-md-6">
            {feats.slice(0, Math.ceil(feats.length / 2)).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct}
                color={
                  f.feature.includes('Exercise') || f.feature.includes('Myalgia') || f.feature.includes('CK') || f.feature.includes('Myoglobin') ? COLOR4
                  : f.feature.includes('Leigh') || f.feature.includes('Lactic') || f.feature.includes('Enceph') ? COLOR3
                  : COLOR
                } />
            ))}
          </div>
          <div className="col-md-6">
            {feats.slice(Math.ceil(feats.length / 2)).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct}
                color={
                  f.feature.includes('Cardiomyo') || f.feature.includes('Seiz') ? COLOR3
                  : f.feature.includes('Maternal') ? COLOR
                  : COLOR2
                } />
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
                <th>Variant</th><th>Heteroplasmy (muscle)</th>
                <th>CIII%</th><th>Lactate</th><th>Ex.Int</th><th>Myoglob</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map((p, i) => (
                <tr key={i}>
                  <td>{p.patient_id}</td>
                  <td className="small text-muted">{(p.phenotype || '').substring(0, 30)}…</td>
                  <td>{p.onset_weeks <= 4 ? 'Neonatal' : p.onset_weeks <= 52 ? `${p.onset_weeks}w` : `${Math.round(p.onset_weeks/52)}yr`}</td>
                  <td><code className="small">{p.variant}</code></td>
                  <td><span style={{ color: p.heteroplasmy_pct_muscle > 85 ? COLOR3 : COLOR4 }}>{p.heteroplasmy_pct_muscle}%</span></td>
                  <td><span style={{ color: p.ciii_activity_pct < 5 ? COLOR3 : COLOR2 }}>{p.ciii_activity_pct}%</span></td>
                  <td><span style={{ color: p.lactic_acid_mmolL > 8 ? COLOR3 : COLOR2 }}>{p.lactic_acid_mmolL}</span></td>
                  <td>{p.exercise_intolerance ? <span style={{ color: COLOR4 }}>✓</span> : '—'}</td>
                  <td>{p.myoglobinuria ? <span style={{ color: COLOR3 }}>✓</span> : '—'}</td>
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
  const phenoDist = data.phenotype_distribution || [];
  const immunoblot = data.immunoblot_pattern || {};
  const bnPage = data.bn_page_pattern || {};

  return (
    <div>
      <div className="alert alert-info small mb-3">
        <strong>🧬 mtDNA Variants — Important Note:</strong> MT-CYB is mitochondrially encoded — variants described as protein changes (p.Xxx#Xxx) or mtDNA nucleotide positions (m.XXXX). Heteroplasmy level in muscle tissue is the key determinant of phenotype severity.
        <strong> WES does NOT reliably detect mtDNA point mutations</strong> — dedicated mtDNA sequencing on muscle DNA is required.
      </div>

      <SectionCard title="Pathogenic Variants in MT-CYB (OMIM *516020) — mtDNA Encoded" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                <th>Protein change</th><th>cDNA</th><th>Domain</th>
                <th>Type</th><th>Severity</th><th>Phenotype</th><th>Heteroplasmy (muscle)</th><th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {(data.all_variants || []).map((v, i) => (
                <tr key={i}>
                  <td><code>{v.change}</code></td>
                  <td><code className="text-muted">{v.cdna}</code></td>
                  <td>{v.domain}</td>
                  <td><span className="badge bg-secondary">{v.type}</span></td>
                  <td><span className="badge" style={{ background: v.severity === 'Severe' ? COLOR3 : v.severity === 'Moderate–Severe' ? '#f57c00' : COLOR5 }}>{v.severity}</span></td>
                  <td className="small text-muted">{v.phenotype}</td>
                  <td><span className="badge" style={{ background: COLOR4 }}>{v.heteroplasmy_range}</span></td>
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
            <p className="small text-muted mb-2">Average CIII residual: <strong>{biochem.avg_ciii_activity_pct}%</strong> (2–25% range; correlates with muscle heteroplasmy)</p>
            <Bar label="CIII <5% (near absent)" value={biochem.ciii_below5_pct} color={COLOR3} />
            <Bar label="CIII 5–10%" value={biochem.ciii_5to10_pct} color={COLOR4} />
            <Bar label="CIII 10–15%" value={biochem.ciii_10to15_pct} color={COLOR2} />
            <Bar label="CIII 15–25% (higher heteroplasmy)" value={biochem.ciii_15to25_pct} color={COLOR} />
            <hr />
            <p className="small text-muted mb-1">Muscle heteroplasmy (avg: <strong>{biochem.avg_heteroplasmy_muscle_pct}%</strong>)</p>
            <Bar label="Heteroplasmy >85% (severe range)" value={biochem.heteroplasmy_above85_pct} color={COLOR3} />
            <Bar label="Heteroplasmy 70–85% (moderate-severe)" value={biochem.heteroplasmy_70to85_pct} color={COLOR4} />
            <Bar label="Heteroplasmy <70% (mild range)" value={biochem.heteroplasmy_below70_pct} color={COLOR} />
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
          <SectionCard title="Phenotype Distribution" borderColor={COLOR4}>
            {phenoDist.map((p, i) => (
              <div key={i} className="d-flex justify-content-between align-items-center mb-1">
                <span className="small text-muted">{(p.phenotype || '').substring(0, 40)}</span>
                <div>
                  <span className="badge me-1" style={{ background: COLOR4 }}>{p.count}</span>
                  <span className="text-muted small">{p.pct}%</span>
                </div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="BN-PAGE Pattern (MT-CYB)" borderColor={COLOR4}>
            <p className="fw-bold small mb-1" style={{ color: COLOR3 }}>{bnPage.finding}</p>
            <p className="small mb-2 text-muted">{bnPage.interpretation}</p>
            <div className="alert alert-info py-1 px-2 small">{bnPage.ddx_value}</div>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Immunoblot Pattern (MT-CYB)" borderColor={COLOR2}>
            {Object.entries(immunoblot).map(([k, v], i) => (
              <div key={i} className="d-flex justify-content-between align-items-start mb-1 border-bottom pb-1">
                <span className="small fw-semibold" style={{ minWidth: 160 }}>{k}</span>
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
        <strong>⚠️ Key distinguishing features of MT-CYB:</strong>{' '}
        (1) <strong style={{ color: COLOR }}>MATERNAL inheritance + HETEROPLASMY</strong> — DDx from ALL nuclear CIII defects (AR biallelic); test maternal relatives.
        (2) <strong style={{ color: COLOR4 }}>Exercise intolerance / myopathy (60%)</strong> — DISTINGUISHING from nuclear CIII defects (which are usually severe infantile).
        (3) <strong style={{ color: COLOR3 }}>Myoglobinuria (32%)</strong> — PATHOGNOMONIC for mtDNA myopathy in CIII context; dark urine post-exercise.
        (4) <strong style={{ color: COLOR5 }}>WES misses mtDNA mutations</strong> — dedicated mtDNA sequencing on MUSCLE DNA required; blood unreliable.
        (5) ISOLATED CIII deficiency (CI, CII, CIV normal) — DDx from MTFMT (combined CI+CIII+CIV) and FASTKD2 (combined CI+CIV).
        (6) <strong>Linezolid blocks MT-CYB translation directly</strong> — double-hit: inhibiting the gene that is the disease target.
      </div>

      <SectionCard title="Differential Diagnosis — MT-CYB vs Other CIII/OXPHOS Diseases" borderColor={COLOR}>
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

      <SectionCard title="Genetic Counselling (Maternal Inheritance)" borderColor={COLOR2}>
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
                <tr><td className="fw-semibold" style={{ color: COLOR }}>Genome location</td><td>{data.chromosome}</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>Inheritance</td><td><strong style={{ color: COLOR4 }}>{data.inheritance}</strong></td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>Protein size</td><td>{prot.size_aa} aa, {prot.kDa} kDa</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>TM helices</td><td>{prot.tm_helices} — spans IMM fully; heme bL (Qi) + heme bH (Qo)</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>Localization</td><td>{prot.localization}</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>Function</td><td>{prot.function}</td></tr>
                <tr><td className="fw-semibold" style={{ color: COLOR }}>CIII assembly role</td><td>{data.ciii_assembly_step}</td></tr>
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
export default function MTCYBPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/mtcyb/overview`).then(r => r.json()).then(setOverview).catch(() => setError('overview failed'));
    fetch(`${API}/api/mtcyb/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => setError('breakdown failed'));
    fetch(`${API}/api/mtcyb/definitions`).then(r => r.json()).then(setDefs).catch(() => setError('definitions failed'));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2" style={{ borderBottom: `3px solid ${COLOR}`, paddingBottom: 8 }}>
        <span style={{ fontSize: 28 }}>🧬</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>MT-CYB — Mitochondrially Encoded Cytochrome b / Isolated CIII Deficiency</h4>
          <small className="text-muted">ONLY mtDNA-Encoded CIII Subunit · Qo + Qi Sites · 8 TM Helices · Exercise Intolerance / Myopathy → Leigh Syndrome · MATERNAL Inheritance · OMIM *516020 · mtDNA rCRS 14747–15887 · 40-patient cohort seed-735</small>
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

'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#1b5e20';   // deep green — AR / peripheral scaffold / no TM helix / PP-module ND3-ND4L
const LIGHT = '#e8f5e9';

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

function Alert({ variant, text }) {
  const bg     = variant === 'danger'  ? '#ffebee' : variant === 'warning' ? '#fff8e1'
               : variant === 'success' ? '#e8f5e9' : LIGHT;
  const border = variant === 'danger'  ? '#c62828' : variant === 'warning' ? '#f57f17'
               : variant === 'success' ? '#2e7d32' : COLOR;
  return (
    <div className="mb-2 p-2 rounded small" style={{ background: bg, borderLeft: `4px solid ${border}` }}>
      {text}
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

// ── Tab: Overview ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ff = data.feature_frequencies_pct || {};
  const bf = data.biochemical_fingerprint || {};
  const p  = data.protein || {};

  return (
    <>
      <div className="row g-3 mb-4">
        <KPI label="Gene"              value={data.gene}                         color={COLOR} />
        <KPI label="Also known as"     value="B9 (bovine)"                       color={COLOR} />
        <KPI label="OMIM Gene"         value={`*${data.omim_gene}`}              color={COLOR} />
        <KPI label="Chromosome"        value={data.chromosome}                   color={COLOR} />
        <KPI label="Inheritance"       value="AR"                                color="#2e7d32" />
        <KPI label="Protein"           value={`${p.size_kda} kDa / ${p.size_aa} aa`} color={COLOR} />
      </div>

      <Alert variant="warning" text="⚠️ AR inheritance — both sexes equally affected; consanguinity common. 25% recurrence risk per pregnancy. WES mandatory to distinguish NDUFA3 (19q13.42) from NDUFB9/AQDQ (8q24.13) — superficially similar names, entirely different genes and CI face zones." />
      <Alert variant="success" text="🔬 PERIPHERAL scaffold — NO canonical TM helix (unique among PP-module CI subunits of similar size). B9 designation (bovine CI proteomics) ≠ NDUFB9 (AQDQ). PP-module ND3–ND4L boundary matrix-face anchor." />

      <SectionCard title="🧬 Gene & Protein">
        <p className="small mb-1"><strong>Full name:</strong> {data.gene_full_name}</p>
        <p className="small mb-1"><strong>Also known as:</strong> {data.also_known_as}</p>
        <p className="small mb-1"><strong>Fold / domain:</strong> {p.fold}</p>
        <p className="small mb-1"><strong>Module:</strong> {p.module}</p>
        <p className="small mb-1"><strong>TM helices:</strong> {p.tm_helices} — PERIPHERAL subunit (no TM helix; matrix-face scaffold)</p>
        <p className="small mb-1"><strong>Size:</strong> {p.size_aa} aa / {p.size_kda} kDa</p>
        <p className="small mb-0"><strong>Function:</strong> {p.function}</p>
      </SectionCard>

      <SectionCard title="🔑 Pathway Note">
        <p className="small mb-0">{data.key_pathway_note}</p>
      </SectionCard>

      <SectionCard title="🧪 Biochemical Fingerprint">
        {Object.entries(bf).map(([k, v]) => (
          <p key={k} className="small mb-1">
            <strong>{k.replace(/_/g, ' ')}:</strong>{' '}
            <span style={{ color: v.includes('NORMAL') ? '#2e7d32' : v.includes('ELEVATED') || v.includes('SEVERELY') ? '#c62828' : undefined }}>
              {v}
            </span>
          </p>
        ))}
      </SectionCard>

      <SectionCard title="📊 Feature Frequencies (40-patient cohort, seed-655)">
        {Object.entries(ff).map(([k, v]) => (
          <Bar key={k} label={k} value={v} />
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Features ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const sd = data.sex_distribution || {};
  const ab = data.onset_age_buckets || {};
  const od = data.outcome_distribution || [];
  const rd = data.region_distribution || [];
  const md = data.mutation_distribution || [];
  const ps = data.patients_sample || [];

  return (
    <>
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <SectionCard title="👥 Sex Distribution (AR — equal)">
            <p className="small mb-1"><strong>Males:</strong> {sd.males} ({sd.male_pct}%)</p>
            <p className="small mb-1"><strong>Females:</strong> {sd.females} ({sd.female_pct}%)</p>
            <p className="small mb-2 text-muted">{sd.note}</p>
            <Bar label="Males" value={sd.male_pct} color={COLOR} />
            <Bar label="Females" value={sd.female_pct} color="#6a1b9a" />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="⏱️ Age at Onset">
            <p className="small mb-1"><strong>Neonatal (&lt;6 mo):</strong> {ab.neonatal_under_6mo}</p>
            <p className="small mb-1"><strong>Infantile (6 mo–2 yr):</strong> {ab.infantile_6mo_to_2yr}</p>
            <p className="small mb-1"><strong>Childhood (2–10 yr):</strong> {ab.childhood_2_to_10yr}</p>
            <p className="small mb-0"><strong>Juvenile (&gt;10 yr):</strong> {ab.juvenile_over_10yr}</p>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🧬 Mutation Distribution">
        {md.map((m, i) => (
          <div key={i} className="mb-1">
            <Bar label={m.mutation} value={m.pct} />
          </div>
        ))}
      </SectionCard>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <SectionCard title="🌍 Region Distribution">
            {rd.map((r, i) => <Bar key={i} label={r.region} value={r.pct} />)}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="📋 Outcomes">
            {od.map((o, i) => (
              <p key={i} className="small mb-1">
                <strong>{o.outcome}:</strong> {o.count} ({o.pct}%)
              </p>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🗂️ Patient Sample (first 15)">
        <div className="table-responsive">
          <table className="table table-sm table-striped small">
            <thead>
              <tr>
                <th>#</th><th>Sex</th><th>Onset (yr)</th><th>Mutation</th>
                <th>Region</th><th>CI%</th><th>Leigh MRI</th><th>Lactic Ac.</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {ps.map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td><span style={{ color: COLOR, fontWeight: 'bold' }}>{p.sex}</span></td>
                  <td>{p.age_onset_years}</td>
                  <td style={{ maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={p.mutation}>{p.mutation}</td>
                  <td>{p.region}</td>
                  <td>{p.ci_activity_pct_control}%</td>
                  <td>{p.leigh_mri ? '✅' : '—'}</td>
                  <td>{p.lactic_acidosis ? '✅' : '—'}</td>
                  <td>{p.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments & DDx ──────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const tx  = data.treatments || {};
  const ddx = data.ddx_key_negatives || [];

  return (
    <>
      <SectionCard title="🚫 ABSOLUTE Contraindications" borderColor="#c62828">
        {(tx.absolute_contraindicated || []).map((d, i) => (
          <Alert key={i} variant="danger" text={`${d.drug}: ${d.reason}`} />
        ))}
      </SectionCard>

      <SectionCard title="⛔ Contraindicated" borderColor="#e65100">
        {(tx.contraindicated || []).map((d, i) => (
          <Alert key={i} variant="warning" text={`${d.drug}: ${d.reason}`} />
        ))}
      </SectionCard>

      <SectionCard title="⚠️ Avoid / High Caution" borderColor="#f57f17">
        {(tx.avoid_caution || []).map((d, i) => (
          <Alert key={i} variant="warning" text={`${d.drug}: ${d.reason}`} />
        ))}
      </SectionCard>

      <SectionCard title="💊 Level C Cofactors (supportive)" borderColor="#2e7d32">
        {(tx.level_c_cofactors || []).map((a, i) => (
          <div key={i} className="mb-2 p-2 rounded small" style={{ background: '#e8f5e9', borderLeft: '4px solid #2e7d32' }}>
            <strong>{a.agent}</strong> ({a.dose}) — {a.note}
          </div>
        ))}
      </SectionCard>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <SectionCard title="💉 Supportive Protocols" borderColor="#1565c0">
            <p className="small mb-1"><strong>Preferred AED:</strong> {tx.preferred_aed}</p>
            <p className="small mb-1"><strong>Glucose:</strong> {tx.glucose_protocol}</p>
            <p className="small mb-1"><strong>Anaesthesia:</strong> {tx.anaesthesia}</p>
            <p className="small mb-0"><strong>Genetic counselling:</strong> {tx.genetic_counselling}</p>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="🔍 DDx Key Negatives" borderColor="#4a148c">
            {ddx.map((d, i) => (
              <div key={i} className="mb-2 small">
                <span className="fw-bold" style={{ color: '#2e7d32' }}>✅ {d.finding}</span>
                <br />
                <span className="text-muted">→ Excludes: {d.ddx_excluded}</span>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const refs    = data.references || [];
  const entries = Object.entries(data).filter(([k]) => k !== 'references');

  return (
    <>
      {entries.map(([k, v]) => (
        <SectionCard key={k} title={`📖 ${k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}`}>
          <p className="small mb-0">{v}</p>
        </SectionCard>
      ))}
      <SectionCard title="📚 References">
        <ul className="small mb-0">
          {refs.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      </SectionCard>
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function NDUFA3Page() {
  const [tab, setTab]               = useState(0);
  const [overview, setOverview]     = useState(null);
  const [breakdown, setBreakdown]   = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError]           = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ndufa3/overview`).then(r => r.json()),
      fetch(`${API}/api/ndufa3/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ndufa3/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return (
    <div className="container py-4">
      <div className="alert alert-danger">Failed to load NDUFA3 data: {error}</div>
    </div>
  );

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 NDUFA3 — Leigh Syndrome CI-Leigh (B9 / PP-Module ND3–ND4L Peripheral Scaffold, AR)
        </h4>
        <p className="text-muted small mb-0">
          NDUFA3 (B9) · 19q13.42 · OMIM *603837 / #256000 (Leigh) ·
          Autosomal recessive · Isolated CI 5–20% · CII/CIII/CIV NORMAL ·
          40-patient cohort seed-655 · Peripheral scaffold — NO TM helix · B9 ≠ NDUFB9/AQDQ
        </p>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <TreatmentsTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}

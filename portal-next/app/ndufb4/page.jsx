'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#4a148c';   // deep purple — PD-module membrane arm / B15 scaffold
const LIGHT = '#f3e5f5';

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
        <KPI label="Gene"              value={data.gene}         color={COLOR} />
        <KPI label="Also known as"     value="B15 / PD-module"   color={COLOR} />
        <KPI label="OMIM Gene"         value={`*${data.omim_gene}`}  color={COLOR} />
        <KPI label="Chromosome"        value={data.chromosome}   color={COLOR} />
        <KPI label="Inheritance"       value={data.inheritance}  color={COLOR} />
        <KPI label="Protein (mature)"  value={`${p.size_kda} kDa`} color={COLOR} />
      </div>

      <SectionCard title="🧬 Gene & Protein">
        <p className="small mb-1"><strong>Full name:</strong> {data.gene_full_name}</p>
        <p className="small mb-1"><strong>Also known as:</strong> {data.also_known_as}</p>
        <p className="small mb-1"><strong>Fold / domain:</strong> {p.fold}</p>
        <p className="small mb-1"><strong>Module:</strong> {p.module}</p>
        <p className="small mb-0"><strong>Function:</strong> {p.function}</p>
      </SectionCard>

      <SectionCard title="⚡ Key Pathway Note — PD-Module Membrane Arm Scaffold">
        <p className="small mb-0">{data.key_pathway_note}</p>
      </SectionCard>

      <SectionCard title="🔬 Biochemical Fingerprint">
        {Object.entries(bf).map(([k, v]) => (
          <div key={k} className="d-flex justify-content-between small mb-1">
            <span className="fw-semibold">{k.replace(/_/g, ' ').toUpperCase()}</span>
            <span className={v.includes('NORMAL') ? 'text-success' : 'text-danger fw-bold'}>{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📊 Feature Frequencies (40-patient cohort, seed-641)">
        {Object.entries(ff).map(([k, v]) => (
          <Bar key={k} label={k.replace(/_/g, ' ')} value={v} />
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Features ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const pts = data.patients || [];
  const hist = data.ci_activity_histogram || {};
  const out  = data.outcome_distribution || {};
  const mut  = data.mutation_distribution || {};
  const reg  = data.region_distribution || {};
  const sex  = data.sex_distribution || {};

  return (
    <>
      <div className="row g-3 mb-4">
        <KPI label="Total patients"  value={data.n}  color={COLOR} />
        <KPI label="Male"            value={sex.M}   color={COLOR} />
        <KPI label="Female"          value={sex.F}   color={COLOR} />
      </div>

      <div className="row g-3">
        <div className="col-md-6">
          <SectionCard title="🧬 Mutation Distribution">
            {Object.entries(mut).sort((a,b) => b[1]-a[1]).map(([k,v]) => (
              <div key={k} className="d-flex justify-content-between small mb-1 border-bottom pb-1">
                <span style={{ maxWidth: '75%' }}>{k}</span>
                <span className="fw-bold">{v}</span>
              </div>
            ))}
          </SectionCard>
          <SectionCard title="🌍 Geographic Distribution">
            {Object.entries(reg).sort((a,b) => b[1]-a[1]).map(([k,v]) => (
              <Bar key={k} label={k} value={Math.round(v / data.n * 100)} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="📈 Outcome Distribution">
            {Object.entries(out).sort((a,b) => b[1]-a[1]).map(([k,v]) => (
              <div key={k} className="d-flex justify-content-between small mb-2">
                <span>{k}</span>
                <span className="badge" style={{ background: COLOR }}>{v}</span>
              </div>
            ))}
          </SectionCard>
          <SectionCard title="📉 CI Activity Histogram">
            {(hist.bins || []).map((bin, i) => (
              <Bar key={bin} label={bin} value={Math.round((hist.counts?.[i] || 0) / data.n * 100)} />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🗂 Patient Cohort (40 patients, seed-641)">
        <div style={{ overflowX: 'auto' }}>
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                <th>#</th><th>Age onset (mo)</th><th>Sex</th><th>CI %</th>
                <th>Mutation</th><th>Region</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {pts.map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.age_onset_months}</td>
                  <td>{p.sex}</td>
                  <td>{p.ci_activity_pct}</td>
                  <td style={{ maxWidth: 240, wordBreak: 'break-word' }}>{p.mutation}</td>
                  <td>{p.region}</td>
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

// ── Tab: Treatments & DDx ─────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ddx = data.key_ddx || [];
  const abs = data.absolute_contraindications || [];
  const con = data.contraindicated || [];
  const pref = data.preferred_treatments || [];

  return (
    <>
      <SectionCard title="🚫 Absolute Contraindications">
        {abs.map((t, i) => <Alert key={i} variant="danger" text={t} />)}
      </SectionCard>

      <SectionCard title="⛔ Contraindicated">
        {con.map((t, i) => <Alert key={i} variant="warning" text={t} />)}
      </SectionCard>

      <SectionCard title="✅ Preferred Treatments / Cofactors">
        {pref.map((t, i) => <Alert key={i} variant="success" text={t} />)}
      </SectionCard>

      <SectionCard title="🔍 Key Differential Diagnosis Points">
        {ddx.map((d, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: LIGHT }}>
            <div className="fw-semibold small mb-1" style={{ color: COLOR }}>{d.feature}</div>
            <div className="small text-muted">{d.significance}</div>
            {d.target_gene && (
              <div className="small mt-1">
                <span className="badge bg-secondary me-1">{d.target_gene}</span>
                {d.target_freq_pct > 0 && <span className="text-danger">{d.target_freq_pct}% prevalence</span>}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📚 Key References">
        {(data.key_references || []).map((r, i) => (
          <div key={i} className="small mb-1 border-bottom pb-1">{r}</div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const sections = [
    { key: 'prescribing_safety', label: '💊 Prescribing Safety' },
    { key: 'gene_concepts',      label: '🧬 Gene Concepts' },
    { key: 'disease_concepts',   label: '🏥 Disease Concepts' },
    { key: 'pharmacology',       label: '⚗️ Pharmacology' },
  ];
  const catColor = {
    absolute_contraindication: '#c62828',
    contraindicated: '#e65100',
    caution: '#f57f17',
    treatment: '#2e7d32',
    gene_concept: COLOR,
    disease_concept: '#1565c0',
    prescribing_safety: '#6a1b9a',
  };

  return (
    <>
      {sections.map(({ key, label }) => (
        <SectionCard key={key} title={label}>
          {(data[key] || []).map((item, i) => (
            <div key={i} className="mb-3 p-2 rounded border-start border-3"
              style={{ borderColor: catColor[item.category] || COLOR, background: '#fafafa' }}>
              <div className="fw-semibold small mb-1"
                style={{ color: catColor[item.category] || COLOR }}>{item.term}</div>
              <div className="small text-muted" style={{ whiteSpace: 'pre-line' }}>{item.detail}</div>
            </div>
          ))}
        </SectionCard>
      ))}
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function NDUFB4Page() {
  const [tab,    setTab]    = useState(0);
  const [over,   setOver]   = useState(null);
  const [bdown,  setBdown]  = useState(null);
  const [defs,   setDefs]   = useState(null);
  const [error,  setError]  = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ndufb4/overview`).then(r => r.json()),
      fetch(`${API}/api/ndufb4/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ndufb4/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOver(o); setBdown(b); setDefs(d); })
      .catch(e => setError(String(e)));
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="mb-4 p-3 rounded" style={{ background: COLOR, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">&#x1f9ec; NDUFB4 — Leigh Syndrome Isolated Complex I Deficiency</h4>
        <div className="small opacity-90">
          CI-Leigh · PD-Module B15 2-TM-Helix Membrane Arm Structural Scaffold · MT-ND4 Face ·
          15kDa · 129aa precursor · 3q13.33 · OMIM Gene *603840 · AR biallelic · Isolated CI 5–20% · CII/CIII/CIV NORMAL
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab    data={over}  />}
      {tab === 1 && <PatientsTab   data={bdown} />}
      {tab === 2 && <TreatmentsTab data={over}  />}
      {tab === 3 && <DefinitionsTab data={defs} />}
    </div>
  );
}

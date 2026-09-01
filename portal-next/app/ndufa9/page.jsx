'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#1b5e20';   // deep forest green — Q/membrane-arm junction, I-gamma assembly checkpoint
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
  const pt = data.protein || {};
  const co = data.cohort || {};
  return (
    <>
      <div className="row g-2 mb-3">
        <KPI label="Gene"          value={data.gene}                           color={COLOR} />
        <KPI label="Also Known As" value={data.also_known_as || 'NUCM 39kDa'} color={COLOR} />
        <KPI label="Size"          value={`${pt.size_kda} kDa`}               color={COLOR} />
        <KPI label="Fe-S Cluster"  value="None"                               color="#455a64" />
        <KPI label="Fold"          value="SDR (structural)"                   color="#2e7d32" />
        <KPI label="Module"        value="Q/Mem-Arm Jxn"                      color="#1b5e20" />
        <KPI label="Inheritance"   value="AR biallelic"                       color={COLOR} />
        <KPI label="Chromosome"    value={data.chromosome}                    color={COLOR} />
        <KPI label="OMIM Gene"     value={`*${data.omim_gene}`}               color="#388e3c" />
        <KPI label="OMIM Disease"  value={`#${data.omim_disease}`}            color="#388e3c" />
        <KPI label="Cohort N"      value={co.n}                               color={COLOR} />
        <KPI label="Mean CI Act"   value={`${co.ci_activity_mean_pct}%`}     color="#e53935" />
      </div>

      <SectionCard title="Key Pathway Note">
        <p className="small mb-0">{data.key_pathway_note}</p>
      </SectionCard>

      <SectionCard title="Biochemical Fingerprint — Isolated CI Deficiency">
        <div className="row g-2">
          {Object.entries(bf).map(([k, v]) => (
            <div key={k} className="col-md-6">
              <div className="p-2 rounded small" style={{
                background: k === 'complex_I' ? '#ffebee' : '#e8f5e9',
                borderLeft: `4px solid ${k === 'complex_I' ? '#c62828' : '#2e7d32'}`,
              }}>
                <strong>{k.replace('_', ' ').toUpperCase()}</strong>: {v}
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Clinical Feature Frequencies (40-patient cohort, seed-631)">
        <div className="row g-3">
          <div className="col-md-6">
            {['psychomotor_regression','lactic_acidosis','hypotonia','leigh_mri'].map(k => (
              <Bar key={k} label={k.replace(/_/g,' ')} value={ff[k] || 0} />
            ))}
          </div>
          <div className="col-md-6">
            {['seizures','respiratory_compromise','ataxia','dystonia'].map(k => (
              <Bar key={k} label={k.replace(/_/g,' ')} value={ff[k] || 0} />
            ))}
          </div>
        </div>
        <hr />
        <div className="row g-3">
          <div className="col-md-12">
            <p className="small fw-bold mb-2">Key DDx Negative Markers:</p>
            {['hcm','peripheral_neuropathy','olfactory_bulb_lesions','leukodystrophy','hepatopathy'].map(k => (
              <Bar key={k} label={k.replace(/_/g,' ')} value={ff[k] || 0} color="#e53935" />
            ))}
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Key Differential Diagnosis Points">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>Feature</th><th>NDUFA9</th><th>Comparator</th><th>Significance</th>
              </tr>
            </thead>
            <tbody>
              {(data.key_ddx || []).map((d, i) => (
                <tr key={i}>
                  <td>{d.feature}</td>
                  <td className="text-success fw-bold">Absent</td>
                  <td>{d.target_gene} {d.target_freq_pct > 0 ? `(${d.target_freq_pct}%)` : ''}</td>
                  <td>{d.significance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Key References">
        <ul className="small mb-0">
          {(data.key_references || []).map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Features ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const pts = data.patients || [];
  const od = data.outcome_distribution || {};
  const md = data.mutation_distribution || {};
  const rd = data.region_distribution || {};
  const sd = data.sex_distribution || {};
  return (
    <>
      <div className="row g-3 mb-3">
        <div className="col-md-6">
          <SectionCard title="Outcome Distribution">
            {Object.entries(od).map(([k, v]) => (
              <div key={k} className="d-flex justify-content-between small mb-1">
                <span>{k}</span><span className="badge" style={{ background: COLOR }}>{v}</span>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Sex / Region">
            <div className="mb-2 small"><strong>M:</strong> {sd.M} &nbsp; <strong>F:</strong> {sd.F}</div>
            {Object.entries(rd).map(([k, v]) => (
              <div key={k} className="d-flex justify-content-between small mb-1">
                <span>{k}</span><span className="text-muted">{v}</span>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Mutation Distribution">
        {Object.entries(md).map(([k, v]) => (
          <div key={k} className="d-flex justify-content-between small mb-1">
            <span className="text-break me-2">{k}</span>
            <span className="badge" style={{ background: '#388e3c' }}>{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="CI Activity Histogram">
        <div className="row">
          {(data.ci_activity_histogram?.bins || []).map((bin, i) => (
            <div key={i} className="col-6 col-md-3 text-center mb-2">
              <div className="fw-bold" style={{ color: COLOR }}>{data.ci_activity_histogram.counts[i]}</div>
              <div className="small text-muted">{bin}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title={`All ${pts.length} Patients`}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>#</th><th>Sex</th><th>Onset (mo)</th><th>CI%</th>
                <th>Region</th><th>Mutation</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {pts.map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.age_onset_months}</td>
                  <td>{p.ci_activity_pct}</td>
                  <td>{p.region}</td>
                  <td className="text-break" style={{ maxWidth: 200 }}>{p.mutation}</td>
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
  return (
    <>
      <SectionCard title="Absolute Contraindications" borderColor="#c62828">
        {(data.absolute_contraindications || []).map((t, i) => (
          <Alert key={i} variant="danger" text={t} />
        ))}
      </SectionCard>

      <SectionCard title="Contraindicated" borderColor="#e65100">
        {(data.contraindicated || []).map((t, i) => (
          <Alert key={i} variant="warning" text={t} />
        ))}
      </SectionCard>

      <SectionCard title="Preferred Treatments (Level C / Supportive)" borderColor="#2e7d32">
        {(data.preferred_treatments || []).map((t, i) => (
          <Alert key={i} variant="success" text={t} />
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const cats = {
    absolute_contraindication: { label: 'Absolute Contraindications', color: '#c62828', bg: '#ffebee' },
    contraindicated:           { label: 'Contraindicated',             color: '#e65100', bg: '#fff3e0' },
    caution:                   { label: 'Caution / Avoid',             color: '#f57f17', bg: '#fff8e1' },
    treatment:                 { label: 'Treatments',                  color: '#2e7d32', bg: '#e8f5e9' },
    gene_concept:              { label: 'Gene Concepts',               color: COLOR,     bg: LIGHT    },
    disease_concept:           { label: 'Disease Concepts',            color: '#37474f', bg: '#eceff1' },
    prescribing_safety:        { label: 'Prescribing Safety',          color: '#4a148c', bg: '#f3e5f5' },
  };
  const all = [
    ...(data.pharmacology || []),
    ...(data.gene_concepts || []),
    ...(data.disease_concepts || []),
    ...(data.prescribing_safety || []),
  ];
  return (
    <>
      {Object.entries(cats).map(([cat, meta]) => {
        const items = all.filter(d => d.category === cat);
        if (!items.length) return null;
        return (
          <SectionCard key={cat} title={meta.label} borderColor={meta.color}>
            {items.map((d, i) => (
              <div key={i} className="mb-3 p-3 rounded small"
                   style={{ background: meta.bg, borderLeft: `4px solid ${meta.color}` }}>
                <div className="fw-bold mb-1">{d.term}</div>
                <div style={{ whiteSpace: 'pre-line' }}>{d.detail}</div>
              </div>
            ))}
          </SectionCard>
        );
      })}
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function NDUFA9Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/ndufa9/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
    fetch(`${API}/api/ndufa9/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/ndufa9/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3" style={{ borderBottom: `3px solid ${COLOR}` }}>
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>
          NDUFA9 — Leigh Syndrome
        </h4>
        <p className="text-muted small mb-1">
          Isolated Complex I Deficiency · Q-Module/Membrane-Arm Junction · SDR-Fold 39kDa · I-gamma Sub-Assembly ·
          OMIM *603834 / #256000 / #618245 · 12q24.31 · AR
        </p>
        {error && <div className="alert alert-danger py-1 small">{error}</div>}
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
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
      {tab === 2 && <TreatmentsTab data={overview} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}

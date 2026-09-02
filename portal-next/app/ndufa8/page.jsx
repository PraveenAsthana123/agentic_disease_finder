'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#1b5e20';   // deep green — AR / N-Q boundary peripheral / B14.5b / NDUFS3-contact / no TM helix
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
  if (!data) return null;
  const k = data.kpis || {};
  return (
    <div>
      <SectionCard title="Gene & Disease Identity">
        <div className="row g-2 small">
          {[
            ['Gene', data.gene],
            ['Protein', 'NADH:Ubiquinone Oxidoreductase Subunit A8 (B14.5b)'],
            ['Module', data.module],
            ['Size', data.protein_size],
            ['Chromosome', data.chromosome],
            ['OMIM Gene', `*${data.omim_gene}`],
            ['OMIM Disease', `#${data.omim_disease}`],
            ['Inheritance', data.inheritance],
            ['Disease', data.disease],
          ].map(([l, v]) => (
            <div className="col-md-6" key={l}>
              <span className="fw-semibold">{l}:</span> <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Cohort KPIs — {data.cohort_n} patients (seed {data.seed})</h6>
      <div className="row mb-4">
        <KPI label="Seizures"         value={`${k.seizures_pct}%`}        color={COLOR} />
        <KPI label="Hypotonia"        value={`${k.hypotonia_pct}%`}       color={COLOR} />
        <KPI label="Lactic Acidosis"  value={`${k.lactic_acidosis_pct}%`} color={COLOR} />
        <KPI label="Leigh MRI"        value={`${k.leigh_mri_pct}%`}       color="#b71c1c" />
        <KPI label="Resp. Compromise" value={`${k.respiratory_pct}%`}     color="#e65100" />
        <KPI label="Median Onset"     value={`${k.median_onset_mo} mo`}   color="#4a148c" />
        <KPI label="Mean CI %"        value={`${k.mean_ci_pct}%`}         color="#006064" />
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Phenotype Distribution">
            {(data.phenotype_distribution || []).map(p => (
              <Bar key={p.class} label={`${p.class} (n=${p.n})`} value={p.pct} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Seizure Types">
            {(data.seizure_types || []).map(s => (
              <Bar key={s.type} label={s.type} value={s.pct} color="#4a148c" />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Seizure Triggers">
        <div className="row">
          {(data.triggers || []).map(t => (
            <div className="col-md-6" key={t.trigger}>
              <Bar label={t.trigger} value={t.pct} color="#bf360c" />
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Key Molecular Concepts">
        {(data.key_concepts || []).map(c => (
          <Alert key={c.concept} variant="info" text={<><strong>{c.concept}:</strong> {c.detail}</>} />
        ))}
      </SectionCard>

      <SectionCard title="References">
        <ul className="small mb-0">
          {(data.references || []).map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Tab: Patients & Features ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return null;
  const pts = data.patients || [];
  return (
    <div>
      <SectionCard title="Variant Distribution">
        {(data.variants || []).map(v => (
          <div key={v.variant} className="mb-3 p-3 rounded" style={{ background: LIGHT }}>
            <div className="d-flex justify-content-between align-items-start">
              <div>
                <span className="fw-bold" style={{ color: COLOR }}>{v.variant}</span>
                <span className="text-muted ms-2 small">({v.cDNA})</span>
                <span className="badge ms-2" style={{ background: COLOR, color: '#fff' }}>
                  {v.freq_pct}% freq · n={v.n_in_cohort}
                </span>
              </div>
              <span className="badge bg-secondary">{v.modal_phenotype}</span>
            </div>
            <div className="small text-muted mt-1"><strong>Structural:</strong> {v.structural_impact}</div>
            <div className="small mt-1">{v.detail}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (40 patients, seed 663)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr style={{ background: COLOR, color: '#fff' }}>
                <th>ID</th><th>Phenotype</th><th>Onset (mo)</th>
                <th>Variant</th><th>CI%</th>
                <th>Sz</th><th>Hyp</th><th>LA</th><th>Leigh</th>
              </tr>
            </thead>
            <tbody>
              {pts.map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.phenotype.split(' (')[0]}</td>
                  <td>{p.onset_mo}</td>
                  <td>{p.variant}</td>
                  <td>{p.ci_pct}%</td>
                  <td>{p.has_seizure ? '✓' : '—'}</td>
                  <td>{p.has_hypotonia ? '✓' : '—'}</td>
                  <td>{p.has_lactic_acidosis ? '✓' : '—'}</td>
                  <td>{p.has_leigh_mri ? '✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Treatments & DDx ─────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return null;
  return (
    <div>
      <SectionCard title="Treatments (Evidence Level)">
        {(data.treatments || []).map(t => (
          <div key={t.name} className="mb-2 p-2 rounded" style={{ background: LIGHT }}>
            <div className="d-flex justify-content-between">
              <span className="fw-semibold small">{t.name}</span>
              <span className="badge" style={{
                background: t.evidence === 'A' ? '#1b5e20' : t.evidence === 'B' ? '#e65100' : '#01579b',
                color: '#fff'
              }}>Level {t.evidence}</span>
            </div>
            <div className="small text-muted mt-1">{t.rationale}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications">
        {(data.contraindications || []).map(c => (
          <Alert
            key={c.drug}
            variant={c.class === 'ABSOLUTE' ? 'danger' : c.class === 'CONTRAINDICATED' ? 'warning' : 'info'}
            text={<><strong>{c.drug}</strong> [{c.class}] — {c.reason}</>}
          />
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Protocol">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr style={{ background: LIGHT }}>
              <th>Parameter</th><th>Protocol</th>
            </tr></thead>
            <tbody>
              {(data.monitoring || []).map(m => (
                <tr key={m.parameter}>
                  <td className="fw-semibold">{m.parameter}</td>
                  <td className="text-muted">{m.protocol}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return null;
  return (
    <div>
      <SectionCard title="Key Concepts">
        {(data.key_concepts || []).map(c => (
          <Alert key={c.concept} variant="info" text={<><strong>{c.concept}:</strong> {c.detail}</>} />
        ))}
      </SectionCard>
      <SectionCard title="Glossary">
        {(data.glossary || []).map(g => (
          <div key={g.term} className="mb-2">
            <span className="fw-semibold" style={{ color: COLOR }}>{g.term}:</span>{' '}
            <span className="small text-muted">{g.definition}</span>
          </div>
        ))}
      </SectionCard>
      <SectionCard title="References">
        <ul className="small mb-0">
          {(data.references || []).map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Root component ────────────────────────────────────────────────────────────
export default function NDUFA8Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOv]     = useState(null);
  const [breakdown, setBd]    = useState(null);
  const [definitions, setDef] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/ndufa8/overview`).then(r => r.json()),
      fetch(`${API}/api/ndufa8/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ndufa8/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, def]) => { setOv(ov); setBd(bd); setDef(def); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const tabContent = [
    <OverviewTab    key="ov"  data={overview} />,
    <PatientsTab    key="bd"  data={breakdown} />,
    <TreatmentsTab  key="tx"  data={breakdown} />,
    <DefinitionsTab key="def" data={definitions} />,
  ];

  return (
    <div className="container-fluid py-3">
      <div className="mb-3 p-3 rounded" style={{ background: COLOR, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">🧬 NDUFA8 — Leigh Syndrome / Isolated Complex I Deficiency</h4>
        <div className="small opacity-75">
          B14.5b · N-Q Module Boundary Peripheral Stabiliser · NDUFS3 Contact · No TM Helix · AR · 9q33.2 ·
          OMIM Gene *603649 · Disease #256000
        </div>
      </div>

      {loading && <div className="alert alert-info">Loading NDUFA8 dashboard…</div>}
      {error   && <div className="alert alert-danger">Error: {error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {!loading && !error && tabContent[tab]}
    </div>
  );
}

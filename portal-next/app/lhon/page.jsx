'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Genetics', 'Treatments & DDx', 'Definitions'];
const COLOR = '#b71c1c';   // deep red — LHON/Complex I (optic neuropathy, red-green colour defect, mtDNA)
const LIGHT = '#ffebee';

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
  const bg = variant === 'danger' ? '#ffebee' : variant === 'warning' ? '#fff8e1' : variant === 'success' ? '#e8f5e9' : LIGHT;
  const border = variant === 'danger' ? '#c62828' : variant === 'warning' ? '#f57f17' : variant === 'success' ? '#2e7d32' : COLOR;
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
        <h6 className="card-title fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function Spinner() {
  return <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /></div>;
}

// ── Tab 1: Overview ──────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="Gene & Disease Identity">
        <div className="row g-2 small">
          {[
            ['Gene', data.gene],
            ['Protein', data.protein],
            ['Disease', data.disease],
            ['OMIM Gene', data.omim_gene],
            ['OMIM Disease', data.omim_disease],
            ['Locus', data.chromosome],
            ['Inheritance', data.inheritance],
            ['Onset', data.onset],
          ].map(([k, v]) => (
            <div key={k} className="col-12 col-md-6">
              <span className="fw-semibold">{k}:</span> <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
        <div className="mt-3 p-2 rounded small" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          <strong>Mechanism:</strong> {data.mechanism}
        </div>
        <div className="mt-2 p-2 rounded small" style={{ background: '#fce4ec', borderLeft: `4px solid #880e4f` }}>
          <strong>mtDNA Pattern:</strong> {data.mtdna_pattern}
        </div>
      </SectionCard>

      <SectionCard title="Clinical Feature Prevalence (40-Patient Cohort)">
        <div className="row g-3 mb-3">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
        {(data.feature_bars || []).map(b => (
          <Bar key={b.label} label={b.label} value={b.pct} />
        ))}
      </SectionCard>

      <SectionCard title="DDx: LHON vs OPA1/ADOA vs NAION vs Toxic Optic Neuropathy">
        <Alert variant="danger" text="⛔ LHON HALLMARKS: maternal inheritance + subacute onset (days-weeks) + sequential bilateral (second eye 6-8 wk) + red-green dyschromatopsia + peripapillary telangiectasia with NO FFA leak + young adult male (80-90%). DDx from OPA1: OPA1 = AD nuclear + childhood onset + insidious + tritanopia (blue-yellow) + simultaneous bilateral + NOT maternal." />
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>Feature</th>
                <th>LHON</th>
                <th>OPA1/ADOA</th>
                <th>NAION</th>
                <th>Toxic Optic N.</th>
              </tr>
            </thead>
            <tbody>
              {(data.ddx_comparison || []).map(r => (
                <tr key={r.feature}>
                  <td className="fw-semibold">{r.feature}</td>
                  <td style={{ color: COLOR }}>{r.lhon}</td>
                  <td>{r.opa1_adoa}</td>
                  <td>{r.naion}</td>
                  <td>{r.toxic_optic}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Genetics ──────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <Spinner />;
  const mut = data.mutations_summary || {};
  const mutColors = { 'MT-ND4': '#b71c1c', 'MT-ND1': '#880e4f', 'MT-ND6': '#1a237e' };

  return (
    <div>
      <SectionCard title="Mutation-Specific Statistics">
        <Alert variant="info" text="🧬 PROGNOSIS by mutation: MT-ND6 m.14484T>C — best (50% spontaneous recovery) | MT-ND1 m.3460G>A — intermediate (~22%) | MT-ND4 m.11778G>A — worst (<4%); counsel patients on mutation-specific outlook at diagnosis." />
        <div className="row g-3">
          {Object.entries(mut).map(([gene, m]) => (
            <div key={gene} className="col-12 col-md-4">
              <div className="card h-100" style={{ borderTop: `3px solid ${mutColors[gene] || COLOR}` }}>
                <div className="card-body small">
                  <h6 className="fw-bold" style={{ color: mutColors[gene] || COLOR }}>{gene}</h6>
                  <div>n = {m.n} patients ({m.pct}%)</div>
                  <div>Avg onset: {m.avg_onset} yr</div>
                  <div>Males: {m.n_male}/{m.n}</div>
                  <div>Recovery: {m.n_recovered}/{m.n} ({m.recovery_pct}%)</div>
                  <div>Tobacco: {m.n_tobacco}/{m.n}</div>
                  <div>Idebenone: {m.n_idebenone}/{m.n}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Visual Acuity Distribution (Current)">
        {Object.entries(data.va_distribution || {}).map(([va, n]) => (
          <Bar key={va} label={`VA ${va}`} value={Math.round(100 * n / (data.n_patients || 40))} />
        ))}
      </SectionCard>

      <SectionCard title="Disease Phase Distribution">
        {Object.entries(data.phase_distribution || {}).map(([ph, n]) => (
          <Bar key={ph} label={ph.replace(/-/g, ' ').replace(/_/g, ' ')} value={Math.round(100 * n / (data.n_patients || 40))} />
        ))}
      </SectionCard>

      <SectionCard title="Per-Patient Table (40 patients, seed-583)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>ID</th><th>Mutation</th><th>Sex</th><th>Onset (yr)</th>
                <th>Phase</th><th>VA Nadir</th><th>VA Current</th>
                <th>Recovered</th><th>Idebenone</th><th>Tobacco</th><th>Heteropl.</th>
              </tr>
            </thead>
            <tbody>
              {(data.patient_table || []).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold" style={{ color: COLOR }}>{p.id}</td>
                  <td className="small">{p.mutation}</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_yr}</td>
                  <td><span className="badge" style={{ background: p.phase === 'acute' ? '#b71c1c' : p.phase === 'subacute' ? '#e65100' : p.phase === 'established' ? '#4a148c' : '#37474f', fontSize: '0.7rem' }}>{p.phase}</span></td>
                  <td>{p.va_nadir}</td>
                  <td>{p.va_current}</td>
                  <td style={{ color: p.recovered === 'Yes' ? '#1b5e20' : '#c62828' }}>{p.recovered}</td>
                  <td>{p.idebenone}</td>
                  <td style={{ color: p.tobacco === 'Yes' ? '#c62828' : undefined }}>{p.tobacco}</td>
                  <td>{p.heteroplasmic}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="LHON vs OPA1/ADOA Head-to-Head Comparison">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ background: LIGHT }}>
              <tr><th>Feature</th><th>LHON</th><th>OPA1/ADOA</th></tr>
            </thead>
            <tbody>
              {(data.ddx_opa1_table || []).map(r => (
                <tr key={r.feature}>
                  <td className="fw-semibold">{r.feature}</td>
                  <td style={{ color: COLOR }}>{r.lhon}</td>
                  <td style={{ color: '#1b5e20' }}>{r.opa1}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Treatments & DDx ──────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="Contraindications — LHON Prescribing Safety">
        <Alert variant="danger" text="⛔ ABSOLUTE: Tobacco (ABSOLUTE environmental CI) · Ethambutol (ABSOLUTE CI, ALL genotypes) · Linezolid (ABSOLUTE CI, ALL genotypes) | ⚠ AVOID: Alcohol · Amiodarone (CAUTION, ophthalmology co-management required)" />
        {(data.contraindications || []).map(c => (
          <div key={c.drug} className="mb-3 p-3 rounded" style={{
            background: c.class === 'ABSOLUTE' ? '#ffebee' : c.class === 'AVOID' ? '#fff8e1' : '#e3f2fd',
            borderLeft: `4px solid ${c.class === 'ABSOLUTE' ? '#c62828' : c.class === 'AVOID' ? '#f57f17' : '#1565c0'}`
          }}>
            <div className="fw-bold small" style={{ color: c.class === 'ABSOLUTE' ? '#c62828' : c.class === 'AVOID' ? '#e65100' : '#1565c0' }}>
              {c.class === 'ABSOLUTE' ? '⛔ ABSOLUTE CI' : c.class === 'AVOID' ? '⚠ AVOID' : '⚠ CAUTION'}: {c.drug}
            </div>
            <div className="small mt-1 text-muted">{c.mechanism}</div>
            <div className="small mt-1"><strong>Alternative:</strong> {c.alternative}</div>
            {c.notes && <div className="small mt-1 fst-italic">{c.notes}</div>}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Treatments">
        {(data.treatments || []).map(t => (
          <div key={t.name} className="mb-4 p-3 rounded" style={{ background: '#fafafa', border: `1px solid #e0e0e0` }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <div className="fw-bold small" style={{ color: COLOR }}>{t.name}</div>
              <span className="badge" style={{ background: t.evidence.includes('A') ? '#1b5e20' : t.evidence.includes('B') ? '#0d47a1' : '#5d4037' }}>
                {t.evidence}
              </span>
            </div>
            <div className="small text-muted mb-1"><strong>Mechanism:</strong> {t.mechanism}</div>
            <div className="small text-muted mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small mb-1"><strong>Safety:</strong> {t.safety}</div>
            <div className="small mb-1"><strong>Monitoring:</strong> {t.monitoring}</div>
            <div className="small mb-1"><strong>Start window:</strong> {t.start_window}</div>
            {t.caution && (
              <div className="p-2 rounded mt-1" style={{ background: '#fff8e1', borderLeft: '3px solid #f57f17' }}>
                <span className="small">⚠ {t.caution}</span>
              </div>
            )}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Definitions ──────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Spinner />;
  const Section = ({ title, items }) => (
    <SectionCard title={title}>
      {(items || []).map(d => (
        <div key={d.term} className="mb-3">
          <div className="fw-semibold small" style={{ color: COLOR }}>{d.term}</div>
          <div className="small text-muted">{d.definition}</div>
        </div>
      ))}
    </SectionCard>
  );
  return (
    <div>
      <Section title="Gene Biology" items={data.gene_biology} />
      <Section title="Disease Concepts" items={data.disease_concepts} />
      <Section title="Prescribing Safety" items={data.prescribing_safety} />
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────
export default function LHONPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/lhon/overview`).then(r => r.json()),
      fetch(`${API}/api/lhon/breakdown`).then(r => r.json()),
      fetch(`${API}/api/lhon/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefs(df); })
      .catch(e => setError(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          LHON — Leber Hereditary Optic Neuropathy
        </h4>
        <p className="text-muted small mb-2">
          MT-ND4 / MT-ND1 / MT-ND6 · Complex I (NADH:ubiquinone oxidoreductase) ·
          Mitochondrial DNA (maternal inheritance) ·
          OMIM #535000 · Wallace 1988 Science (m.11778G>A discovery) ·
          40-patient cohort (seed-583)
        </p>
        <div className="mt-2 p-2 rounded small fw-bold" style={{ background: LIGHT, border: `1px solid ${COLOR}`, color: COLOR }}>
          ⛔ Tobacco ABSOLUTE CI · ⛔ Ethambutol ABSOLUTE CI · ⛔ Linezolid ABSOLUTE CI ·
          ⚠ Alcohol AVOID · ⚠ Amiodarone CAUTION ·
          ✅ Idebenone 900mg/day Level B (RHODOS trial, EU approved 2015) ·
          ✅ Lenadogene nolparvovec EU conditional approval 2021 (m.11778G>A) ·
          🧬 MATERNAL inheritance (NOT AD like OPA1) · 👁 Red-green dyschromatopsia (NOT blue-yellow like OPA1) ·
          ⚡ Subacute onset (NOT insidious like OPA1) · Sequential bilateral (second eye 6–8 wk) ·
          MT-ND4 m.11778G>A ~70% (worst prognosis <4% recovery) · MT-ND6 m.14484T>C ~15% (50% recovery)
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR, fontWeight: 600 } : {}}
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
      {tab === 3 && <DefinitionsTab data={defs} />}
    </div>
  );
}

'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Heteroplasmy', 'Treatments & DDx', 'Definitions'];
const COLOR = '#e65100';   // deep orange — NARP/Complex V (energy metabolism, retina orange, warm)
const LIGHT = '#fff3e0';

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
      </SectionCard>

      <SectionCard title="Mechanism">
        <p className="small text-muted mb-0">{data.mechanism}</p>
      </SectionCard>

      <SectionCard title="mtDNA Pattern">
        <p className="small text-muted mb-0">{data.mtdna_pattern}</p>
      </SectionCard>

      <SectionCard title={`KPIs — ${data.cohort}`}>
        <div className="row g-2">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Clinical Feature Prevalence">
        {(data.feature_bars || []).map(b => (
          <Bar key={b.label} label={b.label} value={b.pct} />
        ))}
      </SectionCard>

      <SectionCard title="Key Prescribing Alerts">
        <Alert variant="danger" text="⛔ VPA ABSOLUTE CONTRAINDICATION — ALL MT-ATP6 (NARP and MILS): valproate inhibits beta-oxidation + depletes CoA + impairs Complex V substrate → documented fatal metabolic decompensation in MILS. NEVER prescribe VPA in any MT-ATP6 patient regardless of phenotype." />
        <Alert variant="danger" text="⛔ VGB ABSOLUTE CONTRAINDICATION in NARP (RP present) — vigabatrin causes irreversible retinal toxicity (bitemporal VF constriction) ADDITIVE to existing retinitis pigmentosa. ERG monitoring does NOT prevent this in NARP. Do not prescribe vigabatrin for epilepsy in NARP." />
        <Alert variant="danger" text="⛔ KETOGENIC DIET CONTRAINDICATED — KD forces fatty acid beta-oxidation requiring intact Complex V for ATP synthesis. Complex V deficiency (NARP/MILS) cannot utilise the proton gradient efficiently → KD worsens energy failure." />
        <Alert variant="warning" text="⚠ PROPOFOL — AVOID PRIS RISK: propofol uncouples OXPHOS and inhibits Complex I/IV. Standard anaesthetic doses tolerated in healthy patients can cause acute energy failure in mitochondrial disease. Prefer volatile agents for NARP/MILS anaesthesia." />
        <Alert variant="warning" text="⚠ LINEZOLID — ABSOLUTE CI in all mitochondrial disease: inhibits mitochondrial 23S rRNA → blocks synthesis of all 13 mtDNA-encoded OXPHOS subunits including MT-ATP6 → acute energy crisis on top of pre-existing Complex V defect." />
        <Alert variant="success" text="✅ LEV PREFERRED AED: renal excretion, no CYP450 induction, no CoA sequestration, no mito toxicity. IV formulation available for acute seizure management. First-line for all epilepsy in NARP/MILS." />
        <Alert variant="success" text="✅ ACUTE CRISIS: IV dextrose 10% (GIR 6-8 mg/kg/min) + IV thiamine 100-200 mg + sodium bicarbonate + L-carnitine + avoid fasting. Glucose bypasses FA oxidation and directly supports glycolytic ATP without requiring Complex V efficiency." />
      </SectionCard>

      <SectionCard title="DDx — NARP vs LHON vs OPA1 vs Friedreich">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: COLOR, color: '#fff' }}>
              <tr>
                <th>Feature</th>
                <th>NARP (MT-ATP6)</th>
                <th>LHON (MT-ND4/1/6)</th>
                <th>OPA1 / ADOA</th>
                <th>Friedreich (FRDA)</th>
              </tr>
            </thead>
            <tbody>
              {(data.ddx_comparison || []).map(r => (
                <tr key={r.feature}>
                  <td className="fw-semibold">{r.feature}</td>
                  <td>{r.narp}</td>
                  <td>{r.lhon}</td>
                  <td>{r.opa1_adoa}</td>
                  <td>{r.friedreich}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Heteroplasmy ───────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <Spinner />;
  const [sort, setSort] = useState('het_pct');
  const [dir, setDir] = useState(-1);
  const [filter, setFilter] = useState('');

  const cols = [
    { key: 'pid', label: 'ID' },
    { key: 'mutation', label: 'Mutation' },
    { key: 'het_pct', label: 'Het %' },
    { key: 'phenotype', label: 'Phenotype' },
    { key: 'sex', label: 'Sex' },
    { key: 'onset_yr', label: 'Onset yr' },
    { key: 'va', label: 'VA' },
    { key: 'flags', label: 'Features / Alerts' },
  ];

  const sorted = [...(data.patients || [])]
    .filter(p => !filter || Object.values(p).join(' ').toLowerCase().includes(filter.toLowerCase()))
    .sort((a, b) => {
      const av = a[sort]; const bv = b[sort];
      return typeof av === 'number' ? (av - bv) * dir : String(av).localeCompare(String(bv)) * dir;
    });

  const phenoColors = { NARP: '#e65100', MILS: '#b71c1c', Carrier: '#2e7d32' };

  return (
    <div>
      <SectionCard title="Phenotype Breakdown by Heteroplasmy">
        <div className="row g-3">
          {Object.entries(data.phenotype_breakdown || {}).map(([ph, info]) => (
            <div key={ph} className="col-12 col-md-4">
              <div className="card h-100" style={{ borderLeft: `4px solid ${phenoColors[ph] || COLOR}` }}>
                <div className="card-body p-3 small">
                  <div className="fw-bold mb-1" style={{ color: phenoColors[ph] || COLOR }}>{ph}</div>
                  <div>n={info.n} ({info.pct}%) · avg Het: {info.avg_heteroplasmy}%</div>
                  <div>RP: {info.pct_rp}% · Ataxia: {info.pct_ataxia}%</div>
                  <div>Neuropathy: {info.pct_neuro}% · Epilepsy: {info.pct_epil}%</div>
                  <div>Leigh MRI: {info.pct_leigh}%</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Heteroplasmy Distribution">
        <div className="row g-2">
          {Object.entries(data.heteroplasmy_distribution || {}).map(([bin, n]) => (
            <div key={bin} className="col text-center">
              <div className="fw-bold" style={{ color: COLOR }}>{n}</div>
              <div className="small text-muted">{bin}</div>
            </div>
          ))}
        </div>
        <p className="small text-muted mt-2 mb-0">
          Threshold: &lt;70% → subclinical/carrier · 70-90% → NARP · &gt;90% → MILS
        </p>
      </SectionCard>

      <SectionCard title="Mutation Summary">
        {Object.entries(data.mutation_summary || {}).map(([mut, info]) => (
          <div key={mut} className="mb-3 p-2 rounded" style={{ background: LIGHT }}>
            <div className="fw-semibold small">{mut}</div>
            <div className="small text-muted">
              n={info.n} ({info.pct}%) · NARP: {info.n_narp} · MILS: {info.n_mils} · avg Het: {info.avg_het}% · avg onset: {info.avg_onset} yr
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Visual Acuity Distribution">
        <div className="row g-2">
          {Object.entries(data.va_distribution || {}).map(([va, n]) => (
            <div key={va} className="col-auto text-center">
              <div className="fw-bold" style={{ color: COLOR }}>{n}</div>
              <div className="small text-muted">{va}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Treatment Usage">
        {Object.entries(data.treatment_summary || {}).map(([tx, pct]) => (
          <Bar key={tx} label={tx} value={pct} color={tx.includes('ADVERSE') ? '#c62828' : COLOR} />
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort — 40 Patients (seed-585)">
        <div className="mb-2">
          <input className="form-control form-control-sm" placeholder="Filter patients…"
            value={filter} onChange={e => setFilter(e.target.value)} style={{ maxWidth: 320 }} />
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-hover table-bordered small">
            <thead style={{ backgroundColor: COLOR, color: '#fff' }}>
              <tr>
                {cols.map(c => (
                  <th key={c.key} style={{ cursor: 'pointer' }}
                    onClick={() => { setSort(c.key); setDir(sort === c.key ? -dir : -1); }}>
                    {c.label} {sort === c.key ? (dir === -1 ? '▼' : '▲') : ''}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map(p => (
                <tr key={p.pid}
                  style={{ backgroundColor: p.phenotype === 'MILS' ? '#ffebee' : p.phenotype === 'NARP' ? '#fff3e0' : '#e8f5e9' }}>
                  {cols.map(c => (
                    <td key={c.key}>{p[c.key]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Treatments & DDx ───────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="Feature Frequencies">
        {(data.feature_frequencies || []) && Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar key={feat} label={feat} value={pct} />
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Spinner />;
  const sections = [
    { key: 'pharmacology',     label: 'Pharmacology & Safety' },
    { key: 'gene_concepts',    label: 'Gene & Molecular Concepts' },
    { key: 'disease_concepts', label: 'Disease Concepts' },
    { key: 'prescribing_safety', label: 'Prescribing Safety (Extended)' },
  ];
  return (
    <div>
      {sections.map(sec => (
        data[sec.key] && (
          <SectionCard key={sec.key} title={sec.label}>
            {data[sec.key].map(d => (
              <div key={d.term} className="mb-4">
                <div className="fw-semibold small mb-1" style={{ color: COLOR }}>{d.term}</div>
                <p className="small text-muted mb-0" style={{ whiteSpace: 'pre-wrap' }}>{d.definition}</p>
              </div>
            ))}
          </SectionCard>
        )
      ))}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function NARPPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/narp/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1) fetch(`${API}/api/narp/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 2) fetch(`${API}/api/narp/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 3) fetch(`${API}/api/narp/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, [tab]);

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      <div className="d-flex align-items-center mb-3 gap-3">
        <div style={{
          width: 48, height: 48, borderRadius: '50%',
          background: COLOR, display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: '#fff', fontWeight: 'bold', fontSize: 22, flexShrink: 0,
        }}>⚡</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            NARP — Neuropathy, Ataxia, Retinitis Pigmentosa
          </h4>
          <div className="text-muted small">
            MT-ATP6 · Complex V (F0-ATP synthase) · Primary mtDNA Mutation · OMIM #551500 (NARP) / #500017 (MILS) · Maternal Inheritance · Heteroplasmy Threshold: 70-90% NARP / &gt;90% MILS
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}>{t}</button>
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

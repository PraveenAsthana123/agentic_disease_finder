'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#e65100';   // deep amber/orange — pyruvate metabolism / energy gateway
const LIGHT  = '#fff3e0';
const COLOR2 = '#1565c0';   // X-linked (PDHA1)
const COLOR3 = '#2e7d32';   // E1-beta
const COLOR4 = '#6a1b9a';   // E2 acetyltransferase
const COLOR5 = '#b71c1c';   // E3 triple-complex (DLD)
const COLOR6 = '#0277bd';   // E3BP
const COLOR7 = '#37474f';   // phosphatase genes
const COLOR8 = '#c62828';   // danger / CI
const COLOR9 = '#1b5e20';   // treatment / KD

const CLASS_COLORS = {
  e1_alpha_xlinked:         '#1565c0',   // PDHA1 — X-linked
  e1_beta_ar:               '#2e7d32',   // PDHB
  e2_ar:                    '#6a1b9a',   // DLAT
  e3_triple_complex_ar:     '#b71c1c',   // DLD — triple complex
  e3bp_ar:                  '#0277bd',   // PDHX
  phosphatase_regulatory_ar: '#37474f', // PDP1
  phosphatase_catalytic_ar:  '#546e7a',  // PDP2
};

const CLASS_LABELS = {
  e1_alpha_xlinked:         'E1-alpha (PDHA1, X-linked, most common)',
  e1_beta_ar:               'E1-beta (PDHB, AR)',
  e2_ar:                    'E2 Acetyltransferase (DLAT, AR)',
  e3_triple_complex_ar:     'E3 — Triple Complex (DLD: PDC+2-OGDC+GCS, AR)',
  e3bp_ar:                  'E3-binding Protein (PDHX, PDC-specific, AR)',
  phosphatase_regulatory_ar: 'PDH Phosphatase Regulatory (PDP1, AR)',
  phosphatase_catalytic_ar:  'PDH Phosphatase Catalytic (PDP2, AR, very rare)',
};

function KPI({ label, value, color = COLOR }) {
  return (
    <div className="col-6 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function BarRow({ label, pct, color = COLOR }) {
  return (
    <div className="mb-1">
      <div className="d-flex justify-content-between mb-0" style={{ fontSize: '0.78rem' }}>
        <span>{label}</span><span className="fw-semibold">{pct}%</span>
      </div>
      <div className="progress" style={{ height: '7px' }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function RuleCard({ title, text, color = COLOR8 }) {
  return (
    <div className="card mb-2 border-0 shadow-sm">
      <div className="card-body py-2 px-3">
        <div className="fw-semibold small" style={{ color }}>{title}</div>
        <div style={{ fontSize: '0.78rem' }}>{text}</div>
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading overview…</div>;
  const cl = data.aggregate_clinical || {};
  const ci = data.drug_contraindications || {};
  const st = data.special_therapies || {};
  const kr = data.key_rules || {};
  const arch = data.complex_architecture || {};
  const wu = data.wes_utility || {};

  return (
    <div>
      {/* Atlas info */}
      <div className="alert mb-3" style={{ backgroundColor: LIGHT, borderColor: COLOR, borderLeft: `4px solid ${COLOR}` }}>
        <div className="fw-bold mb-1" style={{ color: COLOR }}>{data.atlas_subtitle}</div>
        <div style={{ fontSize: '0.83rem' }}>{data.description}</div>
      </div>

      {/* CRITICAL: KD is the TREATMENT banner */}
      <div className="alert alert-success mb-3 py-2">
        <div className="fw-bold">&#x1f957; KD (Ketogenic Diet) = PRIMARY TREATMENT — OPPOSITE of OXPHOS disorders</div>
        <div className="small mt-1">
          PDC deficiency: KD bypasses the pyruvate block (fat → acetyl-CoA directly into TCA) ·
          OXPHOS disorders: KD often CI or CAUTION · Know the metabolic block before prescribing
        </div>
      </div>

      {/* L/P ratio teaching */}
      <div className="alert alert-warning mb-3 py-2">
        <div className="fw-bold">&#x1f9ea; L/P Ratio &lt;25 = PDC deficiency (pyruvate type) · L/P &gt;25 = OXPHOS (citric-acid type)</div>
        <div className="small mt-1">
          Measure blood + CSF lactate + pyruvate SIMULTANEOUSLY on ice · Delayed/improperly handled samples → false L/P elevation
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Genes" value={data.n_genes} color={COLOR} />
        <KPI label="Patients" value={data.n_patients} color={COLOR} />
        <KPI label="Encephalopathy" value={`${cl.encephalopathy_pct}%`} color={COLOR} />
        <KPI label="Lactic Acidosis" value={`${cl.lactic_acidosis_pct}%`} color={COLOR} />
        <KPI label="Epilepsy" value={`${cl.epilepsy_pct}%`} color={COLOR} />
        <KPI label="Ataxia" value={`${cl.ataxia_pct}%`} color={COLOR2} />
        <KPI label="Cognitive" value={`${cl.cognitive_pct}%`} color={COLOR8} />
        <KPI label="Hepatopathy" value={`${cl.hepatopathy_pct}%`} color={COLOR5} />
        <KPI label="Myopathy" value={`${cl.myopathy_pct}%`} color={COLOR3} />
        <KPI label="Resp. Failure" value={`${cl.respiratory_failure_pct}%`} color={COLOR8} />
      </div>

      {/* Aggregate phenotype bars */}
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header py-1 fw-semibold small" style={{ background: LIGHT }}>Aggregate Clinical Features (280 patients)</div>
            <div className="card-body py-2">
              <BarRow label="Encephalopathy" pct={cl.encephalopathy_pct} color={COLOR} />
              <BarRow label="Lactic Acidosis" pct={cl.lactic_acidosis_pct} color={COLOR} />
              <BarRow label="Epilepsy" pct={cl.epilepsy_pct} color={COLOR2} />
              <BarRow label="Cognitive impairment" pct={cl.cognitive_pct} color={COLOR8} />
              <BarRow label="Ataxia" pct={cl.ataxia_pct} color={COLOR2} />
              <BarRow label="Myopathy" pct={cl.myopathy_pct} color={COLOR3} />
              <BarRow label="Respiratory failure" pct={cl.respiratory_failure_pct} color={COLOR8} />
              <BarRow label="Hepatopathy" pct={cl.hepatopathy_pct} color={COLOR5} />
              <BarRow label="HCM" pct={cl.hcm_pct} color={COLOR5} />
              <BarRow label="SNHL" pct={cl.snhl_pct} color={COLOR4} />
              <BarRow label="Renal" pct={cl.renal_pct} color={COLOR7} />
              <BarRow label="CPEO" pct={cl.cpeo_pct} color={COLOR6} />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          {/* Complex architecture */}
          <div className="card shadow-sm mb-3">
            <div className="card-header py-1 fw-semibold small" style={{ background: LIGHT }}>PDC Complex Architecture</div>
            <div className="card-body py-2">
              {Object.entries(arch).map(([k, v]) => (
                <div key={k} className="mb-1" style={{ fontSize: '0.76rem' }}>
                  <span className="fw-semibold text-capitalize" style={{ color: COLOR }}>{k.replace(/_/g,' ')}: </span>{v}
                </div>
              ))}
            </div>
          </div>
          {/* WES utility */}
          <div className="card shadow-sm">
            <div className="card-header py-1 fw-semibold small" style={{ background: LIGHT }}>Diagnostic Utility</div>
            <div className="card-body py-2">
              {Object.entries(wu).map(([k, v]) => (
                <div key={k} className="mb-1" style={{ fontSize: '0.76rem' }}>
                  <span className="fw-semibold" style={{ color: COLOR2 }}>{k.replace(/_/g,' ')}: </span>{v}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Special therapies */}
      <div className="row mb-3">
        <div className="col-12">
          <div className="card shadow-sm">
            <div className="card-header py-1 fw-semibold small" style={{ background: '#e8f5e9', color: COLOR9 }}>&#x1f957; Special Therapies</div>
            <div className="card-body py-2">
              <div className="row">
                {Object.entries(st).map(([k, v]) => (
                  <div key={k} className="col-md-4 mb-2">
                    <div className="fw-semibold small" style={{ color: COLOR9 }}>{v.therapy || v.rule || k}</div>
                    <div style={{ fontSize: '0.75rem' }} className="text-muted">
                      {v.genes || v.gene_primary || ''} · {v.mechanism || ''} · {v.status || ''}
                      {v.limitation ? ' ⚠ ' + v.limitation : ''}
                      {v.not_effective ? ' 🚫 NOT in: ' + v.not_effective : ''}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Drug CI */}
      <div className="row mb-3">
        <div className="col-12">
          <div className="card shadow-sm">
            <div className="card-header py-1 fw-semibold small" style={{ background: '#ffebee', color: COLOR8 }}>&#x26a0; Drug Contraindications &amp; Rules</div>
            <div className="card-body py-2">
              <div className="row">
                {Object.entries(ci).map(([k, v]) => (
                  <div key={k} className="col-md-4 mb-2">
                    <div className="fw-semibold small" style={{ color: COLOR8 }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                    <div style={{ fontSize: '0.75rem' }}>
                      {typeof v === 'string' ? v : (v.rule || JSON.stringify(v).slice(0, 120) + '…')}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Key rules */}
      <div className="row">
        <div className="col-12">
          <div className="card shadow-sm">
            <div className="card-header py-1 fw-semibold small" style={{ background: LIGHT }}>Key Clinical Rules</div>
            <div className="card-body py-2">
              <div className="row">
                {Object.entries(kr).map(([k, v]) => (
                  <div key={k} className="col-md-6 mb-2">
                    <RuleCard title={k.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase())} text={v} color={COLOR} />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data?.genes) return <div className="text-muted">Loading gene table…</div>;
  const genes = data.genes;

  return (
    <div>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-light">
            <tr>
              <th>Gene</th>
              <th>Alias / Component</th>
              <th>Locus</th>
              <th>Inheritance</th>
              <th>Size</th>
              <th>OMIM</th>
              <th>KD Tx</th>
              <th>Thiamine-Responsive</th>
              <th>DCA</th>
              <th>Triple Complex</th>
              <th>X-linked</th>
              <th>VPA Risk</th>
            </tr>
          </thead>
          <tbody>
            {genes.map(g => (
              <tr key={g.gene}>
                <td>
                  <span className="fw-bold px-2 py-1 rounded text-white" style={{ backgroundColor: CLASS_COLORS[g.gene_class] || COLOR, fontSize: '0.8rem' }}>
                    {g.gene}
                  </span>
                </td>
                <td style={{ fontSize: '0.75rem', maxWidth: '200px' }}>{g.alias}</td>
                <td><code style={{ fontSize: '0.72rem' }}>{g.locus}</code></td>
                <td style={{ fontSize: '0.75rem' }}>{g.inheritance?.split(';')[0]?.split('(')[0]?.trim()}</td>
                <td style={{ fontSize: '0.75rem' }}>{g.aa}</td>
                <td><a href={`https://omim.org/entry/${g.omim_gene}`} target="_blank" rel="noreferrer" style={{ fontSize: '0.73rem' }}>{g.omim_gene}</a></td>
                <td>{g.kd_treatment ? <span className="badge bg-success">&#10003; KD</span> : <span className="badge bg-secondary">No</span>}</td>
                <td>{g.thiamine_responsive ? <span className="badge bg-primary">Responsive</span> : <span className="badge bg-light text-dark border">Trial</span>}</td>
                <td>{g.dca_used ? <span className="badge bg-info text-dark">Used</span> : <span className="badge bg-danger">Not effective</span>}</td>
                <td>{g.dld_triple_complex ? <span className="badge bg-danger">Triple!</span> : <span className="badge bg-light text-dark border">PDC only</span>}</td>
                <td>{g.xlinked ? <span className="badge bg-primary">X-linked</span> : <span className="badge bg-light text-dark border">AR</span>}</td>
                <td style={{ fontSize: '0.72rem', maxWidth: '180px' }}>{g.vpa_ci?.split(';')[0]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* Gene details */}
      {genes.map(g => (
        <div key={g.gene} className="card mb-3 shadow-sm">
          <div className="card-header py-2 d-flex align-items-center gap-2" style={{ background: CLASS_COLORS[g.gene_class] || COLOR }}>
            <span className="fw-bold text-white fs-6">{g.gene}</span>
            <span className="text-white opacity-75 small">{g.alias}</span>
            <span className="ms-auto text-white opacity-75 small">{g.locus} · {g.omim_gene}</span>
          </div>
          <div className="card-body py-2">
            <div className="row">
              <div className="col-md-6">
                <div style={{ fontSize: '0.75rem' }}>
                  <div className="mb-1"><span className="fw-semibold">Phenotype: </span>{g.phenotype}</div>
                  <div className="mb-1"><span className="fw-semibold">Inheritance: </span>{g.inheritance}</div>
                  <div className="mb-1"><span className="fw-semibold">Onset: </span>{g.onset_pattern}</div>
                  <div className="mb-1"><span className="fw-semibold">MRI: </span>{g.mri_pattern}</div>
                  <div className="mb-1"><span className="fw-semibold">Founder: </span>{g.founder_variant}</div>
                </div>
              </div>
              <div className="col-md-6">
                <div style={{ fontSize: '0.75rem' }}>
                  <div className="mb-1"><span className="fw-semibold">Hallmark: </span>{g.hallmark?.slice(0, 220)}…</div>
                  <div className="mb-1"><span className="fw-semibold">DDx: </span>{g.key_ddx?.slice(0, 180)}…</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data?.genes) return <div className="text-muted">Loading clinical atlas…</div>;
  const genes = data.genes;
  const phenotypes = [
    { key: 'encephalopathy_pct', label: 'Encephalopathy', color: COLOR },
    { key: 'lactic_ac_pct',      label: 'Lactic Acidosis', color: COLOR },
    { key: 'epilepsy_pct',       label: 'Epilepsy',         color: COLOR2 },
    { key: 'cognitive_pct',      label: 'Cognitive',        color: COLOR8 },
    { key: 'ataxia_pct',         label: 'Ataxia',           color: COLOR2 },
    { key: 'myopathy_pct',       label: 'Myopathy',         color: COLOR3 },
    { key: 'hepatopathy_pct',    label: 'Hepatopathy',      color: COLOR5 },
    { key: 'respiratory_pct',    label: 'Respiratory',      color: COLOR8 },
    { key: 'hcm_pct',            label: 'HCM',              color: COLOR5 },
    { key: 'snhl_pct',           label: 'SNHL',             color: COLOR4 },
    { key: 'renal_pct',          label: 'Renal',            color: COLOR7 },
    { key: 'cpeo_pct',           label: 'CPEO',             color: COLOR6 },
  ];

  return (
    <div>
      {/* Per-gene phenotype heatmap */}
      <div className="table-responsive mb-3">
        <table className="table table-sm table-bordered align-middle" style={{ fontSize: '0.74rem' }}>
          <thead className="table-light">
            <tr>
              <th style={{ minWidth: '80px' }}>Gene</th>
              <th>Component</th>
              {phenotypes.map(p => <th key={p.key} className="text-center">{p.label}</th>)}
              <th>Cohort N</th>
            </tr>
          </thead>
          <tbody>
            {genes.map(g => (
              <tr key={g.gene}>
                <td>
                  <span className="fw-bold px-1 rounded text-white" style={{ backgroundColor: CLASS_COLORS[g.gene_class] || COLOR, fontSize: '0.78rem' }}>
                    {g.gene}
                  </span>
                </td>
                <td style={{ fontSize: '0.7rem', color: '#555' }}>{CLASS_LABELS[g.gene_class]?.split('(')[0]?.trim()}</td>
                {phenotypes.map(p => {
                  const val = g[p.key] || 0;
                  const opacity = Math.min(1, val / 100);
                  return (
                    <td key={p.key} className="text-center" style={{ background: `rgba(${hexToRgb(p.color)},${opacity * 0.55 + 0.08})` }}>
                      <strong>{val}%</strong>
                    </td>
                  );
                })}
                <td className="text-center">{g.cohort_n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Per-gene phenotype charts */}
      <div className="row">
        {genes.map(g => (
          <div key={g.gene} className="col-md-6 col-xl-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header py-1" style={{ background: CLASS_COLORS[g.gene_class] || COLOR }}>
                <span className="fw-bold text-white">{g.gene}</span>
                <span className="text-white opacity-75 small ms-2">{CLASS_LABELS[g.gene_class]?.split(' (')[0]}</span>
                {g.dld_triple_complex && <span className="badge bg-light text-danger ms-2" style={{ fontSize: '0.65rem' }}>TRIPLE COMPLEX</span>}
                {g.xlinked && <span className="badge bg-light text-primary ms-2" style={{ fontSize: '0.65rem' }}>X-LINKED</span>}
              </div>
              <div className="card-body py-2">
                {phenotypes.map(p => (
                  <BarRow key={p.key} label={p.label} pct={g[p.key] || 0} color={CLASS_COLORS[g.gene_class] || COLOR} />
                ))}
                <div className="mt-2 small">
                  <div><span className="fw-semibold">KD Tx: </span><span className="badge bg-success" style={{ fontSize: '0.65rem' }}>YES</span></div>
                  <div><span className="fw-semibold">Thiamine: </span>{g.thiamine_responsive ? <span className="badge bg-primary" style={{ fontSize: '0.65rem' }}>RESPONSIVE</span> : <span className="badge bg-secondary" style={{ fontSize: '0.65rem' }}>TRIAL</span>}</div>
                  <div><span className="fw-semibold">DCA: </span>{g.dca_used ? <span className="badge bg-info text-dark" style={{ fontSize: '0.65rem' }}>Used</span> : <span className="badge bg-danger" style={{ fontSize: '0.65rem' }}>Not effective</span>}</div>
                  <div className="text-muted mt-1" style={{ fontSize: '0.68rem' }}>{g.onset_pattern}</div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data || !Array.isArray(data)) return <div className="text-muted">Loading definitions…</div>;
  return (
    <div>
      <div className="row">
        {data.map((d, i) => (
          <div key={i} className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header py-2" style={{ background: LIGHT }}>
                <span className="fw-bold small" style={{ color: COLOR }}>{d.term}</span>
              </div>
              <div className="card-body py-2">
                <div style={{ fontSize: '0.78rem' }}>{d.definition}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Utility: hex → r,g,b string for rgba()
function hexToRgb(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r},${g},${b}`;
}

export default function PDCDeficiencyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/pdc-deficiency-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
    fetch(`${API}/api/pdc-deficiency-atlas/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/pdc-deficiency-atlas/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  if (error) return (
    <div className="container py-4">
      <div className="alert alert-danger">Error loading PDC atlas: {error}</div>
      <Link href="/" className="btn btn-sm btn-outline-secondary">&#8592; Back</Link>
    </div>
  );

  return (
    <div className="container-fluid py-3" style={{ maxWidth: '1400px' }}>
      {/* Header */}
      <div className="d-flex align-items-center mb-3 gap-3">
        <Link href="/" className="btn btn-sm btn-outline-secondary">&#8592; Back</Link>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            &#x1f525; PDC-Deficiency-Atlas
          </h4>
          <div className="text-muted small">
            Complete 7-Gene Pyruvate Dehydrogenase Complex Deficiency Atlas
            &middot; {overview?.n_genes || 7} genes &middot; {overview?.n_patients || 280} patients
            &middot; <span className="fw-semibold" style={{ color: COLOR9 }}>&#x1f957; KD = PRIMARY TREATMENT (opposite of OXPHOS)</span>
            &middot; <span className="fw-semibold" style={{ color: COLOR2 }}>PDHA1 X-linked · L/P &lt;25 fingerprint</span>
            &middot; <span className="fw-semibold" style={{ color: COLOR5 }}>DLD = Triple Complex (PDC+2-OGDC+GCS)</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active fw-bold' : ''}`}
              style={tab === t ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* Tab content */}
      {tab === 'Overview'       && <OverviewTab      data={overview}    />}
      {tab === 'Gene Table'     && <GeneTableTab     data={breakdown}   />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown}   />}
      {tab === 'Definitions'    && <DefinitionsTab   data={definitions} />}
    </div>
  );
}

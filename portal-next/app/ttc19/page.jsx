'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Features', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#1b5e20';   // deep green — CIII neurological
const LIGHT  = '#e8f5e9';
const COLOR2 = '#2e7d32';
const COLOR3 = '#6a1b9a';   // purple — psychiatric highlight

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
          🧬 TTC19 — Tetratricopeptide Repeat Domain 19 / CIII Assembly Factor
        </h5>
        <p className="mb-1 small">
          <strong>OMIM Gene:</strong> *{data.omim_gene} &nbsp;|&nbsp;
          <strong>Disease:</strong> CIII-D2 #{data.omim_disease} &nbsp;|&nbsp;
          <strong>Chr:</strong> {data.chromosome} &nbsp;|&nbsp;
          <strong>Protein:</strong> {data.protein_size} &nbsp;|&nbsp;
          <strong>Complex:</strong> {data.complex}
        </p>
        <p className="mb-1 small">
          <strong>Inheritance:</strong> {data.inheritance} &nbsp;|&nbsp;
          <strong>Cohort:</strong> {data.cohort_n} patients (seed {data.seed})
        </p>
        <p className="mb-0 small fw-semibold" style={{ color: COLOR }}>
          🟢 TTC19: CIII assembly factor — TPR scaffold stabilises CIII intermediate.
          Phenotype: spinocerebellar ataxia + spasticity + <span style={{ color: COLOR3 }}>psychiatric features</span> (psychosis/depression ~40%).
          NO aminoaciduria / NO iron overload / NO cholestasis (key DDx from BCS1L-GRACILE).
          BN-PAGE: CIII absent WITHOUT precomplex (distinguishes from BCS1L).
          KD / Metformin / VPA / Linezolid: ABSOLUTE CONTRAINDICATIONS.
        </p>
      </div>

      {/* Clinical alerts */}
      <div className="mb-4">
        {(data.key_clinical_alerts || []).map((a, i) => (
          <div key={i} className="alert alert-warning py-1 px-2 mb-1 small">{a}</div>
        ))}
      </div>

      {/* KPIs row 1 — phenotypes */}
      <h6 className="fw-semibold mb-2" style={{ color: COLOR }}>Phenotype Distribution</h6>
      <div className="row mb-3">
        <KPI label="SCA-like" value={`${s.sca_phenotype_pct}%`} color={COLOR} />
        <KPI label="Encephalomyopathy" value={`${s.encephalomyopathy_pct}%`} color={COLOR2} />
        <KPI label="Psychiatric-predominant" value={`${s.psychiatric_predominant_pct}%`} color={COLOR3} />
        <KPI label="Ovario-Leukodystrophy" value={`${s.ovario_leukodystrophy_pct}%`} color="#7b1fa2" />
        <KPI label="Neonatal-onset" value={`${s.neonatal_onset_pct}%`} color="#b71c1c" />
        <KPI label="Avg Onset (yrs)" value={`${s.avg_onset_age_years}`} color={COLOR} />
      </div>

      {/* KPIs row 2 — biochemistry */}
      <h6 className="fw-semibold mb-2" style={{ color: COLOR }}>Biochemical Profile</h6>
      <div className="row mb-4">
        <KPI label="Avg CIII Residual" value={`${s.avg_ciii_residual_pct}%`} color="#b71c1c" />
        <KPI label="Avg Lactate (mmol/L)" value={`${s.avg_lactate_mmol}`} color="#e65100" />
        <KPI label="Lactic Acidosis" value={`${s.lactic_acidosis_pct}%`} color="#e65100" />
        <KPI label="Cerebellar MRI" value={`${s.cerebellar_mri_pct}%`} color={COLOR} />
        <KPI label="Compound Het" value={`${s.compound_het_pct}%`} color={COLOR2} />
        <KPI label="Cohort N" value={data.cohort_n} color={COLOR} />
      </div>

      {/* Clinical feature bars */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Neurological Features">
            {feats.filter(f => ['Ataxia (any)', 'Spasticity', 'Epilepsy', 'Dystonia', 'Peripheral neuropathy'].includes(f.feature)).map(f => (
              <Bar key={f.feature} label={f.feature} value={f.pct} color={COLOR} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Psychiatric & Imaging" borderColor={COLOR3}>
            {feats.filter(f => ['Psychiatric symptoms', 'Cerebellar MRI atrophy', 'Lactic acidosis'].includes(f.feature)).map(f => (
              <Bar key={f.feature} label={f.feature} value={f.pct} color={COLOR3} />
            ))}
            <div className="mt-3 p-2 rounded small" style={{ background: '#f3e5f5' }}>
              <strong style={{ color: COLOR3 }}>Psychiatric features in TTC19:</strong><br />
              ~37–40% of patients develop psychosis (hallucinations/delusions) or depression/behavioural change.
              May precede neurological signs by years. Initially misdiagnosed as primary psychiatric disorder.
              Atypical antipsychotics (quetiapine/risperidone) preferred; avoid first-generation.
            </div>
          </SectionCard>
        </div>
      </div>

      {/* BN-PAGE key feature */}
      <SectionCard title="BN-PAGE Diagnostic Signature" borderColor="#37474f">
        <div className="row">
          <div className="col-md-6">
            <div className="p-2 rounded border" style={{ background: '#fafafa' }}>
              <p className="small mb-1"><strong>TTC19 pattern:</strong></p>
              <p className="small mb-0 text-danger fw-semibold">
                CIII band ABSENT — NO precomplex accumulation<br />
                Assembly intermediates degraded (not arrested)
              </p>
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-2 rounded border" style={{ background: '#fafafa' }}>
              <p className="small mb-1"><strong>BCS1L contrast:</strong></p>
              <p className="small mb-0" style={{ color: '#1a237e' }}>
                CIII band absent + <strong>precomplex ACCUMULATES</strong><br />
                RISP-free intermediate visible on BN-PAGE
              </p>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Patient table */}
      <SectionCard title={`Patient Cohort (${data.cohort_n} patients, seed ${data.seed})`}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Sex</th><th>Onset</th><th>Phenotype</th>
                <th>CIII%</th><th>Lactate</th><th>Ataxia</th><th>Psych</th><th>Epilepsy</th><th>Variant 1</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_age}y</td>
                  <td><span className="badge" style={{ background: p.phenotype.includes('Psychiatric') ? COLOR3 : COLOR, fontSize: '0.65rem' }}>
                    {p.phenotype.replace('Spinocerebellar ataxia (SCA-like)','SCA').replace('Encephalomyopathy','Encephalo').replace('Psychiatric-predominant','Psychiatric').replace('Ovario-leukodystrophy','Ovario-LD').replace('Neonatal-onset severe','Neonatal')}
                  </span></td>
                  <td className={p.ciii_pct < 10 ? 'text-danger fw-bold' : 'text-warning fw-semibold'}>{p.ciii_pct}%</td>
                  <td className={p.lactate_mmol > 5 ? 'text-danger' : ''}>{p.lactate_mmol}</td>
                  <td>{p.ataxia ? '✓' : '–'}</td>
                  <td>{p.psychiatric ? <span style={{ color: COLOR3 }}>✓</span> : '–'}</td>
                  <td>{p.epilepsy ? '✓' : '–'}</td>
                  <td className="text-muted">{p.variant1}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Variants & Features ──────────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const vars = data.all_variants || [];
  const pheno = data.phenotype_breakdown || [];
  const varBd = data.variant_breakdown || [];
  const ciiiBd = data.ciii_residual_by_phenotype || [];
  const matrix = data.feature_matrix_by_phenotype || [];
  const pop = data.population_breakdown || [];
  const treat = data.treatment_uptake || {};

  return (
    <div>
      {/* Phenotype breakdown */}
      <SectionCard title="Phenotype Distribution">
        <div className="row">
          {pheno.map(p => (
            <div key={p.phenotype} className="col-md-4 mb-3">
              <div className="card text-center h-100 shadow-sm">
                <div className="card-body py-2">
                  <div className="fw-bold fs-4" style={{ color: COLOR }}>{p.pct}%</div>
                  <div className="small">{p.phenotype}</div>
                  <div className="text-muted small">n = {p.count}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* All variants */}
      <SectionCard title="Pathogenic Variants in TTC19">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-light">
              <tr>
                <th>cDNA</th><th>Protein</th><th>Location</th><th>Severity</th>
                <th>CIII%</th><th>Phenotype</th>
              </tr>
            </thead>
            <tbody>
              {vars.map((v, i) => (
                <tr key={i}>
                  <td className="fw-semibold text-monospace">{v.cDNA}</td>
                  <td className="fw-semibold">{v.protein}</td>
                  <td className="text-muted small">{v.location}</td>
                  <td>
                    <span className={`badge ${v.severity === 'severe' ? 'bg-danger' : v.severity === 'intermediate' ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                      {v.severity} ({v.severity_pct}%)
                    </span>
                  </td>
                  <td className={v.ciii_residual_pct < 8 ? 'text-danger fw-bold' : 'text-warning fw-semibold'}>{v.ciii_residual_pct}%</td>
                  <td className="small">{v.phenotype}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small text-muted mt-2">
          Most TTC19 pathogenic variants are null (frameshift/nonsense/splice/large deletion).
          Missense variants in conserved TPR residues (Arg145, Gly73, Leu297) cause intermediate severity.
          Compound heterozygosity is the rule ({data.cohort_n && `~93% of cohort`}).
        </p>
      </SectionCard>

      {/* Feature matrix */}
      <SectionCard title="Clinical Features by Phenotype">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead className="table-light">
              <tr>
                <th>Phenotype</th><th>N</th><th>Ataxia%</th><th>Spasticity%</th>
                <th>Psychiatric%</th><th>Epilepsy%</th><th>Avg Onset</th><th>Avg CIII%</th>
              </tr>
            </thead>
            <tbody>
              {matrix.map((m, i) => (
                <tr key={i}>
                  <td className="fw-semibold small">{m.phenotype}</td>
                  <td>{m.n}</td>
                  <td>{m.ataxia_pct}</td>
                  <td>{m.spasticity_pct}</td>
                  <td style={{ color: m.psychiatric_pct > 50 ? COLOR3 : 'inherit', fontWeight: m.psychiatric_pct > 50 ? 'bold' : 'normal' }}>{m.psychiatric_pct}</td>
                  <td>{m.epilepsy_pct}</td>
                  <td>{m.avg_onset}y</td>
                  <td className={m.avg_ciii_pct < 8 ? 'text-danger fw-bold' : ''}>{m.avg_ciii_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Allele frequency + population */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Variant Allele Frequency (Cohort)">
            {varBd.map((v, i) => (
              <Bar key={i} label={v.protein} value={v.allele_freq_pct} color={COLOR} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Population Distribution">
            {pop.map((p, i) => (
              <Bar key={i} label={p.population} value={p.pct} color={COLOR2} />
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Treatment uptake */}
      <SectionCard title="Treatment Uptake (Cohort)">
        <div className="row">
          {Object.entries(treat).map(([k, v]) => (
            <div key={k} className="col-6 col-md-4 mb-3">
              <div className="text-center">
                <div className="fw-bold" style={{ color: COLOR }}>{v}%</div>
                <div className="text-muted small">{k.replace(/_pct$/,'').replace(/_/g,' ')}</div>
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
  const treat = data.treatment_summary || {};
  const alerts = data.pharmacology_alerts || [];

  return (
    <div>
      {/* Pharmacology alerts */}
      <SectionCard title="⚠️ Pharmacology — Contraindications in CIII Deficiency" borderColor="#b71c1c">
        {alerts.map((a, i) => (
          <div key={i} className="alert alert-danger py-1 px-2 mb-1 small">{a}</div>
        ))}
      </SectionCard>

      {/* DDx table */}
      <SectionCard title="Differential Diagnosis">
        <div className="table-responsive">
          <table className="table table-sm small table-hover">
            <thead className="table-light">
              <tr>
                <th>Gene</th><th>Disease</th><th>Locus</th><th>DDx Point</th>
                <th>Inheritance</th><th>Residual CIII</th>
              </tr>
            </thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{ color: COLOR }}>{d.gene}</td>
                  <td>{d.disease}</td>
                  <td className="text-muted">{d.locus}</td>
                  <td className="small">{d.ddx_point}</td>
                  <td>{d.inheritance}</td>
                  <td>{d.residual_ciii}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Treatment */}
      <SectionCard title="Treatment Approach by Domain" borderColor={COLOR2}>
        {Object.entries(treat).map(([domain, text]) => (
          <div key={domain} className="mb-3">
            <div className="fw-semibold small mb-1" style={{ color: COLOR2 }}>{domain}</div>
            <div className="small text-muted">{text}</div>
            <hr className="my-2" />
          </div>
        ))}
      </SectionCard>

      {/* Imprinting note */}
      {data.imprinting_note && (
        <div className="alert alert-info small">{data.imprinting_note}</div>
      )}
    </div>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const defs = data.definitions || [];
  const stds = data.standards || [];
  const refs = data.references || [];
  const sf = data.structural_features || {};

  return (
    <div>
      {/* Structural features */}
      <SectionCard title="Structural Features — TTC19 TPR Scaffold">
        <div className="table-responsive">
          <table className="table table-sm small">
            <tbody>
              {Object.entries(sf).map(([k, v]) => (
                <tr key={k}>
                  <td className="fw-semibold text-nowrap" style={{ color: COLOR, width: '30%' }}>{k}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Definitions */}
      <SectionCard title="Clinical Definitions">
        {defs.map((d, i) => (
          <div key={i} className="mb-2 pb-2 border-bottom">
            <span className="fw-semibold small" style={{ color: COLOR }}>{d.term}:</span>{' '}
            <span className="small">{d.definition}</span>
          </div>
        ))}
      </SectionCard>

      {/* Standards */}
      <SectionCard title="Standards & Guidelines">
        <ul className="small mb-0">
          {stds.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
        </ul>
      </SectionCard>

      {/* References */}
      <SectionCard title="Key References">
        {refs.map((r, i) => (
          <div key={i} className="mb-3">
            <div className="small fw-semibold">{r.citation}</div>
            <div className="small text-muted mt-1">{r.significance}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function TTC19Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/ttc19/overview`).then(r => r.json()),
      fetch(`${API}/api/ttc19/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ttc19/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="container-fluid py-3">
      {/* Title bar */}
      <div className="mb-3 p-3 rounded" style={{ background: COLOR, color: '#fff' }}>
        <h4 className="mb-1">🧬 TTC19 — Tetratricopeptide Repeat Domain 19</h4>
        <p className="mb-0 small">
          Complex III Deficiency, Nuclear Type 2 (CIII-D2) · OMIM *613814 / #{overview?.omim_disease || '615157'} ·
          AR Biallelic · 17p12 · 380aa TPR Scaffold · No TM Helix · CIII Assembly Factor
        </p>
        <p className="mb-0 small mt-1" style={{ opacity: 0.9 }}>
          Spinocerebellar ataxia + Spasticity + <strong>Psychiatric features (psychosis ~40%)</strong> ·
          NO GRACILE triad (no aminoaciduria/iron overload/cholestasis) ·
          BN-PAGE: CIII absent WITHOUT precomplex (DDx BCS1L) ·
          KD / Metformin / VPA / Linezolid: ABSOLUTE CI
        </p>
      </div>

      {loading && <div className="alert alert-info">Loading TTC19 data…</div>}
      {error && <div className="alert alert-danger">Error: {error}</div>}

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* Tab content */}
      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <VariantsTab data={breakdown} />}
      {tab === 2 && <DDxTab data={definitions} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
